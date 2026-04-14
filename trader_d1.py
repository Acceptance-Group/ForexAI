import os
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import MetaTrader5 as mt5

from config import (
    DATA_CONFIG, RISK_CONFIG, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS,
)
from data_loader import build_dataset, load_raw_prices, compute_atr, compute_adx

SYMBOL = "EURUSD"
MAGIC_NUMBER = 123456
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

VOL_MODEL_PATH = "models/vol_model.json"
VOL_SCALER_PATH = "models/vol_scaler.pkl"
MEANREV_MODEL_PATH = "models/meanrev_model.json"
MEANREV_SCALER_PATH = "models/meanrev_scaler.pkl"

BREAKEVEN_PIPS = DATA_CONFIG.get("breakeven_pips", 15)
TRAIL_PIPS = DATA_CONFIG.get("trail_pips", 10)
MAX_HOLD_BARS = DATA_CONFIG.get("max_hold_bars", 10)
CHECK_INTERVAL_SEC = 60


def connect_mt5(path=MT5_PATH):
    if not mt5.initialize(path=path):
        print(f"MT5 init failed: {mt5.last_error()}")
        return False
    account = mt5.account_info()
    if account is None:
        print(f"MT5 account_info failed: {mt5.last_error()}")
        return False
    print(f"Connected: {account.server} | Login: {account.login} | Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")
    return True


def disconnect_mt5():
    mt5.shutdown()


def get_symbol_info(symbol=SYMBOL):
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Symbol {symbol} not found")
        return None
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return info


def get_current_price(symbol=SYMBOL):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, None
    return tick.ask, tick.bid


def get_open_position(symbol=SYMBOL):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return None
    return positions[0]


def close_position(position):
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY \
        else mt5.symbol_info_tick(position.symbol).ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": close_type,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "d1_ts_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close failed: {result.retcode} - {result.comment}")
        return False
    pnl = position.profit
    pos_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
    print(f"  Closed {pos_type} #{position.ticket} {position.volume:.2f} lots "
          f"@ {price:.5f} P&L=${pnl:.2f}")
    return True


def modify_sl(position, new_sl, symbol=SYMBOL):
    symbol_info = get_symbol_info(symbol)
    if symbol_info is None:
        return False
    digits = symbol_info.digits
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "volume": position.volume,
        "position": position.ticket,
        "sl": float(round(new_sl, digits)),
        "tp": float(round(position.tp, digits)),
        "magic": MAGIC_NUMBER,
        "comment": "d1_trail",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  SL modify failed: {result.retcode if result else 'None'}")
        return False
    return True


def build_signal():
    print("Building D1 signal (XGBoost + Vol Filter + Trailing Stop)...")
    feats_df = build_dataset(force_download=True)
    prices_df = load_raw_prices()
    common_idx = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)
    dir_model = xgb.XGBClassifier()
    dir_model.load_model(MODEL_SAVE_PATH.replace(".pth", ".json"))
    scaled = scaler.transform(feats_df.values)
    dir_prob = dir_model.predict_proba(scaled[-1:].reshape(1, -1))[0, 1]

    vol_model = xgb.XGBRegressor()
    vol_model.load_model(VOL_MODEL_PATH)
    with open(VOL_SCALER_PATH, "rb") as f:
        vol_scaler = pickle.load(f)
    vol_scaled = vol_scaler.transform(feats_df.values)
    vol_pred = vol_model.predict(vol_scaled)
    vol_median = np.median(vol_pred[:int(len(vol_pred) * 0.7)])
    current_vol = vol_pred[-1]
    vol_high = current_vol > vol_median

    close = prices_df["close"].values
    high = prices_df["high"].values
    low = prices_df["low"].values
    atr_series = compute_atr(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])
    atr_mean = atr_series.rolling(60, min_periods=1).mean()
    adx_series = compute_adx(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])

    current_atr = atr_series.iloc[-1]
    current_atr_mean = atr_mean.iloc[-1]
    current_adx = adx_series.iloc[-1]
    atr_pips = current_atr * 10000

    no_trade_buy_above = DATA_CONFIG["no_trade_buy_above"]
    no_trade_sell_below = DATA_CONFIG["no_trade_sell_below"]
    tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
    min_sl_pips = DATA_CONFIG.get("min_sl_pips", 40.0)
    min_adx = DATA_CONFIG.get("min_adx", 0.30)
    risk_per_trade = RISK_CONFIG["risk_per_trade"]
    pip_value = RISK_CONFIG["pip_value_per_lot"]
    contract_size = RISK_CONFIG["contract_size"]
    lot_step = RISK_CONFIG.get("lot_step", 0.01)
    min_lots = RISK_CONFIG.get("min_lots", 0.01)
    max_leverage = RISK_CONFIG["max_leverage"]

    sl_pips = max(atr_pips, min_sl_pips)
    vol_ratio = current_atr / current_atr_mean if not np.isnan(current_atr_mean) and current_atr_mean > 0 else 1.0
    dyn_tp_sl = tp_sl_ratio / max(vol_ratio, 0.5)
    dyn_tp_sl = min(dyn_tp_sl, 4.0)
    tp_pips = sl_pips * dyn_tp_sl
    adx_ok = current_adx >= min_adx

    if dir_prob > no_trade_buy_above:
        signal = "BUY"
    elif dir_prob < no_trade_sell_below:
        signal = "SELL"
    else:
        signal = "HOLD"

    if not adx_ok:
        signal = "HOLD"
        print(f"  ADX {current_adx:.2f} < {min_adx} - filtered")
    if signal != "HOLD" and not vol_high:
        signal = "HOLD"
        print(f"  Vol filter: {current_vol:.5f} <= median {vol_median:.5f} - filtered")

    print(f"\n{'='*60}")
    print(f"  D1 EURUSD Signal ({datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC)")
    print(f"{'='*60}")
    print(f"  P(UP)          : {dir_prob:.4f}")
    print(f"  Predicted ATR  : {current_vol:.5f} (median: {vol_median:.5f})")
    print(f"  Vol High       : {vol_high}")
    print(f"  ADX            : {current_adx:.2f}")
    print(f"  ATR/SL/TP      : {atr_pips:.1f} / {sl_pips:.1f} / {tp_pips:.1f} pips ({dyn_tp_sl:.1f}x)")
    print(f"  Trailing Stop  : BE={BREAKEVEN_PIPS}p / Trail={TRAIL_PIPS}p")
    print(f"  SIGNAL         : {signal}")
    print(f"{'='*60}")

    return {
        "signal": signal, "prob_up": dir_prob,
        "sl_pips": sl_pips, "tp_pips": tp_pips, "atr_pips": atr_pips,
        "vol_high": vol_high, "adx": current_adx,
    }


def place_order(signal_info, symbol=SYMBOL):
    signal = signal_info["signal"]
    if signal == "HOLD":
        print("  HOLD - no trade.")
        return None

    symbol_info = get_symbol_info(symbol)
    if symbol_info is None:
        return None
    ask, bid = get_current_price(symbol)
    if ask is None:
        return None

    account = mt5.account_info()
    if account is None:
        return None

    equity = account.equity
    risk_dollars = equity * RISK_CONFIG["risk_per_trade"]
    sl_pips = signal_info["sl_pips"]
    tp_pips = signal_info["tp_pips"]
    pip_value = RISK_CONFIG["pip_value_per_lot"]

    lots = risk_dollars / (sl_pips * pip_value)
    lots = round(lots / 0.01) * 0.01
    max_lots_val = (equity * RISK_CONFIG["max_leverage"]) / RISK_CONFIG["contract_size"]
    lots = min(lots, max_lots_val)
    lots = max(RISK_CONFIG["min_lots"], lots)

    digits = symbol_info.digits
    point = symbol_info.point
    sl_distance = sl_pips * point * 10
    tp_distance = tp_pips * point * 10

    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = ask
        sl_price = round(price - sl_distance, digits)
        tp_price = round(price + tp_distance, digits)
        original_sl = sl_price
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = bid
        sl_price = round(price + sl_distance, digits)
        tp_price = round(price - tp_distance, digits)
        original_sl = sl_price

    print(f"\n  Order: {signal} {lots:.2f} lots @ {price:.5f}")
    print(f"  SL: {sl_price:.5f} | TP: {tp_price:.5f}")
    print(f"  Risk: ${risk_dollars:.2f} ({RISK_CONFIG['risk_per_trade']*100:.1f}%)")
    print(f"  Trailing: BE={BREAKEVEN_PIPS}p, Trail={TRAIL_PIPS}p")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "price": float(price),
        "sl": float(sl_price),
        "tp": float(tp_price),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "d1_ts",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        print(f"  Order failed: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"  Order rejected: {result.retcode} - {result.comment}")
        return None

    print(f"  Executed! Ticket: {result.order}")
    return {
        "ticket": result.order,
        "signal": signal,
        "lots": lots,
        "entry_price": price,
        "original_sl": original_sl,
        "tp_price": tp_price,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
    }


def manage_trailing_stop(order_info, symbol=SYMBOL):
    is_buy = order_info["signal"] == "BUY"
    entry = order_info["entry_price"]
    original_sl = order_info["original_sl"]
    breakeven_pips_val = BREAKEVEN_PIPS / 10000
    trail_pips_val = TRAIL_PIPS / 10000
    max_hold_hours = MAX_HOLD_BARS * 24

    start_time = datetime.datetime.utcnow()
    be_triggered = False
    highest = entry if is_buy else entry
    lowest = entry if is_buy else entry

    print(f"\n  Managing trailing stop for ticket #{order_info['ticket']}")
    print(f"  {'BUY' if is_buy else 'SELL'} @ {entry:.5f} | SL={original_sl:.5f} | BE={BREAKEVEN_PIPS}p | Trail={TRAIL_PIPS}p")
    print(f"  Max hold: {MAX_HOLD_BARS} D1 bars ({max_hold_hours}h)")

    while True:
        position = get_open_position(symbol)
        if position is None or position.ticket != order_info["ticket"]:
            print("  Position closed (TP/SL hit or manual close)")
            break

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            time.sleep(CHECK_INTERVAL_SEC)
            continue

        current_bid = tick.bid
        current_ask = tick.ask
        current_price = current_bid if is_buy else current_ask
        current_pnl = position.profit
        elapsed = (datetime.datetime.utcnow() - start_time).total_seconds() / 3600

        if is_buy:
            highest = max(highest, current_bid)
            unrealized_pips = (highest - entry) * 10000
            if not be_triggered and unrealized_pips >= BREAKEVEN_PIPS:
                new_sl = entry + 0.0001
                if modify_sl(position, new_sl, symbol):
                    be_triggered = True
                    print(f"  [{elapsed:.1f}h] BE triggered! Unrealized={unrealized_pips:.1f}p SL->{new_sl:.5f}")

            if be_triggered:
                new_sl = highest - trail_pips_val
                current_sl = position.sl
                if new_sl > current_sl and new_sl > entry:
                    if modify_sl(position, new_sl, symbol):
                        print(f"  [{elapsed:.1f}h] Trail UP: SL={new_sl:.5f} (high={highest:.5f})")
        else:
            lowest = min(lowest, current_ask)
            unrealized_pips = (entry - lowest) * 10000
            if not be_triggered and unrealized_pips >= BREAKEVEN_PIPS:
                new_sl = entry - 0.0001
                if modify_sl(position, new_sl, symbol):
                    be_triggered = True
                    print(f"  [{elapsed:.1f}h] BE triggered! Unrealized={unrealized_pips:.1f}p SL->{new_sl:.5f}")

            if be_triggered:
                new_sl = lowest + trail_pips_val
                current_sl = position.sl
                if new_sl < current_sl and new_sl < entry:
                    if modify_sl(position, new_sl, symbol):
                        print(f"  [{elapsed:.1f}h] Trail DOWN: SL={new_sl:.5f} (low={lowest:.5f})")

        if elapsed >= max_hold_hours:
            print(f"  [{elapsed:.1f}h] Max hold reached. Closing position...")
            close_position(position)
            break

        if elapsed % 4 < CHECK_INTERVAL_SEC / 3600:
            pnl_str = f"${current_pnl:+.2f}" if current_pnl else "N/A"
            be_str = "BE" if be_triggered else "--"
            print(f"  [{elapsed:.1f}h] P&L={pnl_str} | {be_str} | High={'highest' if is_buy else lowest:.5f} | SL={position.sl:.5f}")

        time.sleep(CHECK_INTERVAL_SEC)


def run_trading_cycle():
    signal_info = build_signal()
    signal = signal_info["signal"]

    if signal == "HOLD":
        position = get_open_position()
        if position is not None:
            pos_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
            print(f"  Existing {pos_type}: {position.volume:.2f} lots P&L=${position.profit:.2f}")
        return None

    existing = get_open_position()
    if existing is not None:
        pos_type = "BUY" if existing.type == mt5.ORDER_TYPE_BUY else "SELL"
        new_signal = signal_info["signal"]
        same_dir = (pos_type == "BUY" and new_signal == "BUY") or \
                    (pos_type == "SELL" and new_signal == "SELL")
        if same_dir:
            print(f"  Already in {pos_type} matching {new_signal}. Holding. P&L: ${existing.profit:.2f}")
            return None
        print(f"  Closing {pos_type} (new signal: {new_signal})...")
        close_position(existing)
        time.sleep(1)

    order_info = place_order(signal_info)
    if order_info:
        manage_trailing_stop(order_info)
    return order_info


def run_bot_d1():
    print("=" * 60)
    print("  D1 XGBoost + Vol Filter + Trailing Stop Bot")
    print("=" * 60)
    print(f"  Strategy:  BUY>{DATA_CONFIG['no_trade_buy_above']}, SELL<{DATA_CONFIG['no_trade_sell_below']}")
    print(f"  Vol Filter: ATR_pred > median")
    print(f"  ADX Filter: >={DATA_CONFIG.get('min_adx', 0.30)}")
    print(f"  TP/SL:      {DATA_CONFIG['tp_sl_ratio']}x dynamic vol-scaled")
    print(f"  Trailing:   BE={BREAKEVEN_PIPS}p / Trail={TRAIL_PIPS}p / Max={MAX_HOLD_BARS} bars")
    print(f"  Risk:        {RISK_CONFIG['risk_per_trade']*100:.1f}% per trade")
    print("=" * 60)

    if not connect_mt5():
        print("Cannot connect to MT5!")
        return

    last_trade_date = None
    try:
        while True:
            now = datetime.datetime.utcnow()
            today = now.strftime("%Y-%m-%d")

            if now.hour == 0 and now.minute < 5 and today != last_trade_date:
                print(f"\n--- D1 bar close {now.strftime('%Y-%m-%d %H:%M')} UTC ---")
                run_trading_cycle()
                last_trade_date = today
                time.sleep(300)
            else:
                position = get_open_position()
                if position is not None:
                    pos_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                    manage_existing = False

                    if position.magic == MAGIC_NUMBER:
                        is_buy = position.type == mt5.ORDER_TYPE_BUY
                        entry = position.price_open
                        current_sl = position.sl
                        current_tp = position.tp
                        tick = mt5.symbol_info_tick(SYMBOL)
                        if tick:
                            be_price = entry + (1 / 10000 if is_buy else -1 / 10000)
                            be_triggered = (is_buy and current_sl >= be_price) or \
                                          (not is_buy and current_sl <= be_price)
                            if be_triggered and is_buy:
                                high_price = tick.bid
                                trail_sl = high_price - TRAIL_PIPS / 10000
                                if trail_sl > current_sl:
                                    modify_sl(position, trail_sl, SYMBOL)
                            elif be_triggered and not is_buy:
                                low_price = tick.ask
                                trail_sl = low_price + TRAIL_PIPS / 10000
                                if trail_sl < current_sl:
                                    modify_sl(position, trail_sl, SYMBOL)
                            elif not be_triggered:
                                unrealized = (tick.bid - entry if is_buy else entry - tick.ask) * 10000
                                if unrealized >= BREAKEVEN_PIPS:
                                    new_sl = entry + (1 / 10000 if is_buy else -1 / 10000)
                                    modify_sl(position, new_sl, SYMBOL)

                    pnl = position.profit
                    print(f"  [{now.strftime('%H:%M')}] Open {pos_type} P&L=${pnl:.2f}")
                else:
                    hours_left = (24 - now.hour) % 24
                    print(f"  [{now.strftime('%H:%M')}] No position. Next D1 close in ~{hours_left}h")

                time.sleep(CHECK_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        disconnect_mt5()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python trader_d1.py connect   - Test MT5 connection")
        print("  python trader_d1.py signal     - Get current D1 signal")
        print("  python trader_d1.py trade       - Execute one trade with trailing stop")
        print("  python trader_d1.py close       - Close current position")
        print("  python trader_d1.py bot          - Run D1 trading bot (24/7)")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "connect":
        connect_mt5()
        account = mt5.account_info()
        print(f"Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")
        disconnect_mt5()
    elif cmd == "signal":
        connect_mt5()
        build_signal()
        disconnect_mt5()
    elif cmd == "trade":
        connect_mt5()
        run_trading_cycle()
        disconnect_mt5()
    elif cmd == "close":
        connect_mt5()
        pos = get_open_position()
        if pos:
            close_position(pos)
        else:
            print("No position.")
        disconnect_mt5()
    elif cmd == "bot":
        run_bot_d1()