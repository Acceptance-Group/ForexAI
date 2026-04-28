import os
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    DATA_CONFIG, RISK_CONFIG, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS,
)
from data_loader import build_dataset, load_raw_prices, compute_atr, compute_adx
from broker import init_broker, get_broker, shutdown_broker
from ensemble import predict_direction_proba
import telegram_bot


def _utcnow():
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)

SYMBOL = "EURUSD"

VOL_MODEL_PATH = "models/vol_model.json"
VOL_SCALER_PATH = "models/vol_scaler.pkl"
MEANREV_MODEL_PATH = "models/meanrev_model.json"
MEANREV_SCALER_PATH = "models/meanrev_scaler.pkl"

BREAKEVEN_PIPS = DATA_CONFIG.get("breakeven_pips", 15)
TRAIL_PIPS = DATA_CONFIG.get("trail_pips", 10)
MAX_HOLD_BARS = DATA_CONFIG.get("max_hold_bars", 10)
CHECK_INTERVAL_SEC = 60


def connect_broker():
    return init_broker()


def disconnect_broker():
    shutdown_broker()


def get_symbol_info(symbol=SYMBOL):
    try:
        broker = get_broker()
        info = broker.symbol_info(symbol)
        if info is None:
            return None
        return info
    except Exception:
        return None


def get_current_price(symbol=SYMBOL):
    try:
        broker = get_broker()
        tick = broker.symbol_info_tick(symbol)
        if tick is None:
            return None, None
        return tick.ask, tick.bid
    except Exception:
        return None, None


def get_open_position(symbol=SYMBOL):
    try:
        broker = get_broker()
        positions = broker.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return None
        return positions[0]
    except Exception:
        return None


def close_position(position):
    try:
        broker = get_broker()
        result = broker.close_position(position.ticket, symbol=position.symbol)
        if result is None or (hasattr(result, 'retcode') and result.retcode != 1):
            print(f"Close failed: {result}")
            return False

        pnl = position.profit
        pos_type = "BUY" if position.type == 0 else "SELL"
        print(f"  Closed {pos_type} #{position.ticket} {position.volume:.2f} lots @ {position.price_current:.5f} P&L=${pnl:.2f}")
        return True
    except Exception as e:
        print(f"Close error: {e}")
        return False


def modify_sl(position, new_sl, symbol=SYMBOL):
    try:
        broker = get_broker()
        digits = broker.get_symbol_digits(symbol)
        result = broker.modify_position(
            position.ticket,
            sl=float(round(new_sl, digits)),
            tp=float(round(position.tp, digits)),
        )
        if result is None or (hasattr(result, 'retcode') and result.retcode != 1):
            print(f"  SL modify failed")
            return False
        return True
    except Exception as e:
        print(f"  SL modify error: {e}")
        return False


def build_signal():
    print("Building D1 signal (XGBoost + Vol Filter + Trailing Stop)...")
    prices_df = load_raw_prices()
    feats_df = build_dataset(force_download=True)
    common_idx = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    dir_prob, dir_probs = predict_direction_proba(feats_df.values[-1:].reshape(1, -1))
    dir_prob = float(dir_prob[0])
    prob_xgb = float(dir_probs["xgb"][0])
    prob_lgbm = float(dir_probs["lgbm"][0])
    prob_cb = float(dir_probs["cb"][0])

    vol_model = xgb.XGBRegressor()
    vol_model.load_model(VOL_MODEL_PATH)
    with open(VOL_SCALER_PATH, "rb") as f:
        vol_scaler = pickle.load(f)
    vol_scaled = vol_scaler.transform(feats_df.values)
    vol_pred = vol_model.predict(vol_scaled)
    vol_pct = pd.Series(vol_pred).rolling(252, min_periods=30).quantile(0.20).values
    vol_threshold = vol_pct[-1] if not pd.isna(vol_pct[-1]) else np.median(vol_pred)
    current_vol = vol_pred[-1]
    vol_high = current_vol > vol_threshold

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
        print(f"  Vol filter: {current_vol:.5f} <= threshold {vol_threshold:.5f} - filtered")

    print(f"\n{'='*60}")
    print(f"  D1 EURUSD Signal ({_utcnow().strftime('%Y-%m-%d %H:%M')} UTC)")
    print(f"{'='*60}")
    print(f"  P(UP)          : {dir_prob:.4f} (XGB={prob_xgb:.3f} LGBM={prob_lgbm:.3f} CB={prob_cb:.3f})")
    print(f"  Predicted ATR  : {current_vol:.5f} (threshold: {vol_threshold:.5f})")
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

    broker = get_broker()
    symbol_info = broker.symbol_info(symbol)
    if symbol_info is None:
        return None
    ask, bid = get_current_price(symbol)
    if ask is None:
        return None

    account = broker.account_info()
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

    digits = broker.get_symbol_digits(symbol)
    point = broker.get_symbol_point(symbol)
    sl_distance = sl_pips * point * 10
    tp_distance = tp_pips * point * 10

    if signal == "BUY":
        order_type = broker.ORDER_TYPE_BUY
        price = ask
        sl_price = round(price - sl_distance, digits)
        tp_price = round(price + tp_distance, digits)
        original_sl = sl_price
    else:
        order_type = broker.ORDER_TYPE_SELL
        price = bid
        sl_price = round(price + sl_distance, digits)
        tp_price = round(price - tp_distance, digits)
        original_sl = sl_price

    print(f"\n  Order: {signal} {lots:.2f} lots @ {price:.5f}")
    print(f"  SL: {sl_price:.5f} | TP: {tp_price:.5f}")
    print(f"  Risk: ${risk_dollars:.2f} ({RISK_CONFIG['risk_per_trade']*100:.1f}%)")
    print(f"  Trailing: BE={BREAKEVEN_PIPS}p, Trail={TRAIL_PIPS}p")

    request = {
        "symbol": symbol,
        "volume": float(lots),
        "type": order_type,
        "sl": float(sl_price),
        "tp": float(tp_price),
    }

    result = broker.order_send(request)
    if result is None:
        print(f"  Order failed")
        return None
    if hasattr(result, 'retcode') and result.retcode != 1:
        print(f"  Order rejected: {result.comment}")
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

    start_time = _utcnow()
    be_triggered = False
    highest = entry if is_buy else entry
    lowest = entry if is_buy else entry

    print(f"\n  Managing trailing stop for ticket #{order_info['ticket']}")
    print(f"  {'BUY' if is_buy else 'SELL'} @ {entry:.5f} | SL={original_sl:.5f} | BE={BREAKEVEN_PIPS}p | Trail={TRAIL_PIPS}p")
    print(f"  Max hold: {MAX_HOLD_BARS} D1 bars ({max_hold_hours}h)")

    last_pnl = 0
    close_reason = "TP / SL / Trailing"
    while True:
        position = get_open_position(symbol)
        if position is None or position.ticket != order_info["ticket"]:
            print("  Position closed (TP/SL hit or manual close)")
            break

        _, bid = get_current_price(symbol)
        ask, _ = get_current_price(symbol)
        current_price = bid if is_buy else ask
        current_pnl = position.profit
        last_pnl = current_pnl
        elapsed = (_utcnow() - start_time).total_seconds() / 3600

        if is_buy:
            highest = max(highest, bid)
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
            lowest = min(lowest, ask)
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

        if elapsed % 4 < CHECK_INTERVAL_SEC / 3600:
            pnl_str = f"${current_pnl:+.2f}" if current_pnl else "N/A"
            be_str = "BE" if be_triggered else "--"
            high_low_str = f"{highest:.5f}" if is_buy else f"{lowest:.5f}"
            print(f"  [{elapsed:.1f}h] P&L={pnl_str} | {be_str} | High/Low={high_low_str} | SL={position.sl:.5f}")

        time.sleep(CHECK_INTERVAL_SEC)

        if elapsed >= max_hold_hours:
            print(f"  [{elapsed:.1f}h] Max hold reached. Closing position...")
            close_reason = "Max Hold"
            close_position(position)
            break

    telegram_bot.send_trade_close({
        "side": order_info["signal"],
        "pnl": last_pnl,
        "reason": close_reason,
        "ticket": order_info["ticket"],
        "timestamp": _utcnow().isoformat(),
        "duration_hours": (_utcnow() - start_time).total_seconds() / 3600,
    })


def run_trading_cycle():
    signal_info = build_signal()
    signal = signal_info["signal"]

    if signal == "HOLD":
        position = get_open_position()
        if position is not None:
            pos_type = "BUY" if position.type == 0 else "SELL"
            print(f"  Existing {pos_type}: {position.volume:.2f} lots P&L=${position.profit:.2f}")
        return None

    existing = get_open_position()
    if existing is not None:
        pos_type = "BUY" if existing.type == 0 else "SELL"
        new_signal = signal_info["signal"]
        same_dir = (pos_type == "BUY" and new_signal == "BUY") or \
                    (pos_type == "SELL" and new_signal == "SELL")
        if same_dir:
            print(f"  Already in {pos_type} matching {new_signal}. Holding. P&L: ${existing.profit:.2f}")
            return None
        print(f"  Closing {pos_type} (new signal: {new_signal})...")
        closed_pnl = existing.profit
        closed_ticket = existing.ticket
        close_position(existing)
        time.sleep(1)
        telegram_bot.send_trade_close({
            "side": pos_type,
            "pnl": closed_pnl,
            "reason": "Signal Change",
            "ticket": closed_ticket,
            "timestamp": _utcnow().isoformat(),
            "duration_hours": 0,
        })

    order_info = place_order(signal_info)
    if order_info:
        telegram_bot.send_trade_open(signal_info, order_info)
        manage_trailing_stop(order_info)
    return order_info


def run_bot_d1():
    print("=" * 60)
    print(" /$$$$$$$$                                       /$$$$$$  /$$$$$$")
    print("| $$_____/                                      /$$__  $$|_  $$_/")
    print("| $$     /$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$| $$  \\ $$  | $$  ")
    print("| $$$$$ /$$__  $$ /$$__  $$ /$$__  $$|  $$ /$$/| $$$$$$$$  | $$  ")
    print("| $$__/| $$  \\ $$| $$  \\__/| $$$$$$$$ \\  $$$$/ | $$__  $$  | $$  ")
    print("| $$   | $$  | $$| $$      | $$_____/  >$$  $$ | $$  | $$  | $$  ")
    print("| $$   |  $$$$$$/| $$      |  $$$$$$$ /$$/\\  $$| $$  | $$ /$$$$$$")
    print("|__/    \\______/ |__/       \\_______/|__/  \\__/|__/  |__/|______/")
    print("  powered by moonway")
    print("=" * 60)
    print(f"  Strategy:  BUY>{DATA_CONFIG['no_trade_buy_above']}, SELL<{DATA_CONFIG['no_trade_sell_below']}")
    print(f"  Vol Filter: ATR_pred > 20th pct")
    print(f"  ADX Filter: >={DATA_CONFIG.get('min_adx', 0.30)}")
    print(f"  TP/SL:      {DATA_CONFIG['tp_sl_ratio']}x dynamic vol-scaled")
    print(f"  Trailing:   BE={BREAKEVEN_PIPS}p / Trail={TRAIL_PIPS}p / Max={MAX_HOLD_BARS} bars")
    print(f"  Risk:        {RISK_CONFIG['risk_per_trade']*100:.1f}% per trade")
    print("=" * 60)

    if not init_broker():
        print("Cannot connect to broker!")
        return

    telegram_bot.init_telegram()

    last_trade_date = None
    try:
        while True:
            now = _utcnow()
            today = now.strftime("%Y-%m-%d")

            if now.hour == 0 and now.minute < 5 and today != last_trade_date:
                print(f"\n--- D1 bar close {now.strftime('%Y-%m-%d %H:%M')} UTC ---")
                run_trading_cycle()
                last_trade_date = today
                time.sleep(300)
            else:
                position = get_open_position()
                if position is not None:
                    pos_type = "BUY" if position.type == 0 else "SELL"
                    is_buy = position.type == 0
                    entry = position.price_open
                    current_sl = position.sl
                    current_tp = position.tp
                    ask, bid = get_current_price()
                    if ask and bid:
                        tick_ask = ask
                        tick_bid = bid
                        be_price = entry + (1 / 10000 if is_buy else -1 / 10000)
                        be_triggered = (is_buy and current_sl >= be_price) or \
                                      (not is_buy and current_sl <= be_price)
                        if be_triggered and is_buy:
                            high_price = tick_bid
                            trail_sl = high_price - TRAIL_PIPS / 10000
                            if trail_sl > current_sl:
                                modify_sl(position, trail_sl)
                        elif be_triggered and not is_buy:
                            low_price = tick_ask
                            trail_sl = low_price + TRAIL_PIPS / 10000
                            if trail_sl < current_sl:
                                modify_sl(position, trail_sl)
                        elif not be_triggered:
                            unrealized = (tick_bid - entry if is_buy else entry - tick_ask) * 10000
                            if unrealized >= BREAKEVEN_PIPS:
                                new_sl = entry + (1 / 10000 if is_buy else -1 / 10000)
                                modify_sl(position, new_sl)

                    pnl = position.profit
                    print(f"  [{now.strftime('%H:%M')}] Open {pos_type} P&L=${pnl:.2f}")
                else:
                    hours_left = (24 - now.hour) % 24
                    print(f"  [{now.strftime('%H:%M')}] No position. Next D1 close in ~{hours_left}h")

                time.sleep(CHECK_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        shutdown_broker()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python trader_d1.py connect   - Test broker connection")
        print("  python trader_d1.py signal     - Get current D1 signal")
        print("  python trader_d1.py trade      - Execute one trade with trailing stop")
        print("  python trader_d1.py close      - Close current position")
        print("  python trader_d1.py bot         - Run D1 trading bot (24/7)")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "connect":
        init_broker()
        account = get_broker().account_info()
        print(f"Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")
        shutdown_broker()
    elif cmd == "signal":
        init_broker()
        build_signal()
        shutdown_broker()
    elif cmd == "trade":
        init_broker()
        run_trading_cycle()
        shutdown_broker()
    elif cmd == "close":
        init_broker()
        pos = get_open_position()
        if pos:
            close_position(pos)
        else:
            print("No position.")
        shutdown_broker()
    elif cmd == "bot":
        run_bot_d1()