import os
import time
import pickle
import datetime

import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt5

from config import (
    DATA_CONFIG, RISK_CONFIG, DEVICE, MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_WEIGHTS, FEATURE_COLUMNS,
)
from model import ForexClassifier
from data_loader import build_dataset, load_raw_prices, compute_atr


SYMBOL = "EURUSD"
MAGIC_NUMBER = 123456
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"


def connect_mt5(login=None, password=None, server=None, path=MT5_PATH):
    if not mt5.initialize(path=path):
        print(f"MT5 init failed: {mt5.last_error()}")
        return False

    if login and password and server:
        authorized = mt5.login(login=login, password=password, server=server)
        if not authorized:
            print(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

    account = mt5.account_info()
    if account is None:
        print(f"MT5 account_info failed: {mt5.last_error()}")
        return False

    print(f"Connected: {account.server} | Login: {account.login} | Balance: ${account.balance:,.2f} "
          f"| Equity: ${account.equity:,.2f} | Leverage: 1:{account.leverage}")
    return True


def disconnect_mt5():
    mt5.shutdown()


def get_symbol_info(symbol=SYMBOL):
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Symbol {symbol} not found")
        return None
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"Cannot select {symbol}")
            return None
    return info


def get_current_price(symbol=SYMBOL):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Cannot get price for {symbol}")
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
        "comment": "forex_model_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close failed: {result.retcode} - {result.comment}")
        return False
    print(f"Closed position #{position.ticket} {position.volume:.2f} lots @ {price:.5f}")
    return True


def build_signal(use_mt5_prices=True):
    print("Fetching fresh data up to today...")
    feats_df = build_dataset(force_download=True)
    prices_df = load_raw_prices()

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    lookback = DATA_CONFIG["lookback"]
    scaled = scaler.transform(feats_df.values)

    last_seq = scaled[-lookback:]
    X = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob_up = model.predict_proba(X).detach().cpu().item()

    no_trade_buy_above = DATA_CONFIG["no_trade_buy_above"]
    no_trade_sell_below = DATA_CONFIG["no_trade_sell_below"]
    tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
    min_sl_pips = DATA_CONFIG.get("min_sl_pips", 25.0)
    risk_per_trade = RISK_CONFIG["risk_per_trade"]
    pip_value = RISK_CONFIG["pip_value_per_lot"]
    contract_size = RISK_CONFIG["contract_size"]
    lot_step = RISK_CONFIG.get("lot_step", 0.01)
    min_lots = RISK_CONFIG.get("min_lots", 0.01)
    max_leverage = RISK_CONFIG["max_leverage"]
    compound = RISK_CONFIG.get("compound", False)

    close = prices_df["close"].values
    high = prices_df["high"].values
    low = prices_df["low"].values

    atr_series = compute_atr(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])
    atr_mean = atr_series.rolling(60, min_periods=1).mean()

    current_atr = atr_series.iloc[-1]
    current_atr_mean = atr_mean.iloc[-1]

    atr_pips = current_atr * 10000
    sl_pips = max(atr_pips, min_sl_pips)

    vol_ratio = current_atr / current_atr_mean if not np.isnan(current_atr_mean) and current_atr_mean > 0 else 1.0
    dyn_tp_sl = tp_sl_ratio / max(vol_ratio, 0.5)
    dyn_tp_sl = min(dyn_tp_sl, 4.0)
    tp_pips = sl_pips * dyn_tp_sl

    if prob_up > no_trade_buy_above:
        signal = "BUY"
    elif prob_up < no_trade_sell_below:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = abs(prob_up - 0.5) * 2.0

    model_price = close[-1]

    signal_info = {
        "prob_up": prob_up,
        "signal": signal,
        "confidence": confidence,
        "model_price": model_price,
        "atr_pips": atr_pips,
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "dyn_tp_sl": dyn_tp_sl,
        "vol_ratio": vol_ratio,
    }

    print(f"\n{'='*50}")
    print(f"  SIGNAL: {signal}")
    print(f"  P(UP)  : {prob_up:.4f}")
    print(f"  Model Price: {model_price:.5f}")
    print(f"  ATR    : {atr_pips:.1f} pips")
    print(f"  SL/TP  : {sl_pips:.1f} / {tp_pips:.1f} pips (ratio {dyn_tp_sl:.1f}x)")
    print(f"{'='*50}")

    return signal_info


def place_order(signal_info, symbol=SYMBOL):
    signal = signal_info["signal"]
    if signal == "HOLD":
        print("No-Trade zone. Skipping.")
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
    contract_size = RISK_CONFIG["contract_size"]
    lot_step = RISK_CONFIG.get("lot_step", 0.01)
    min_lots = RISK_CONFIG.get("min_lots", 0.01)
    max_leverage = RISK_CONFIG["max_leverage"]

    digits = symbol_info.digits
    point = symbol_info.point
    spread = ask - bid

    lots = risk_dollars / (sl_pips * pip_value)
    lots = round(lots / lot_step) * lot_step

    max_lots_val = (equity * max_leverage) / contract_size
    max_lots_val = round(max_lots_val / lot_step) * lot_step
    lots = min(lots, max_lots_val)
    lots = max(min_lots, lots)

    sl_distance = sl_pips * point * 10
    tp_distance = tp_pips * point * 10

    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = ask
        sl_price = round(price - sl_distance, digits)
        tp_price = round(price + tp_distance, digits)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = bid
        sl_price = round(price + sl_distance, digits)
        tp_price = round(price - tp_distance, digits)

    print(f"\nOrder Details:")
    print(f"  Symbol  : {symbol}")
    print(f"  Type    : {signal}")
    print(f"  Lots    : {lots:.2f}")
    print(f"  Price   : {price:.{digits}f}")
    print(f"  SL      : {sl_price:.{digits}f} ({sl_pips:.1f} pips)")
    print(f"  TP      : {tp_price:.{digits}f} ({tp_pips:.1f} pips)")
    print(f"  Spread  : {spread * 10000:.1f} pips")
    print(f"  Risk    : ${risk_dollars:.2f} ({RISK_CONFIG['risk_per_trade']*100:.1f}%)")
    print(f"  Equity  : ${equity:,.2f}")

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
        "comment": "forex_model_h4",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        print(f"Order send failed: {mt5.last_error()}")
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order rejected: {result.retcode} - {result.comment}")
        print(f"Request: {request}")
        return None

    print(f"\nOrder executed! Ticket: {result.order}")
    print(f"  Volume : {lots:.2f} lots")
    print(f"  Entry  : {price:.{digits}f}")
    print(f"  SL     : {sl_price:.{digits}f}")
    print(f"  TP     : {tp_price:.{digits}f}")

    return result


def check_and_close_existing(symbol=SYMBOL):
    position = get_open_position(symbol)
    if position is None:
        print("No open position.")
        return True

    pos_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
    print(f"Open position: {pos_type} {position.volume:.2f} lots @ {position.price_open:.5f} "
          f"SL={position.sl:.5f} TP={position.tp:.5f}")
    print(f"P&L: ${position.profit:.2f}")

    return close_position(position)


def should_trade_now():
    now = datetime.datetime.utcnow()
    h4_hours = [0, 4, 8, 12, 16, 20]
    minutes_since_bar = now.minute + (now.second / 60.0)
    return now.hour in h4_hours and minutes_since_bar < 15


def run_trading_cycle(force_new=False):
    print("\n" + "=" * 70)
    print(f"  TRADING CYCLE - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    signal_info = build_signal()
    signal = signal_info["signal"]

    if signal == "HOLD":
        print("\nNo-trade zone. Checking for existing positions...")
        position = get_open_position()
        if position is not None:
            pos_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
            print(f"Existing {pos_type} position: {position.volume:.2f} lots P&L=${position.profit:.2f}")
        return None

    existing = get_open_position()
    if existing is not None:
        pos_type = "BUY" if existing.type == mt5.ORDER_TYPE_BUY else "SELL"

        new_signal = signal_info["signal"]
        same_direction = (pos_type == "BUY" and new_signal == "BUY") or \
                         (pos_type == "SELL" and new_signal == "SELL")

        if same_direction and not force_new:
            print(f"\nAlready in {pos_type} position matching signal {new_signal}. Holding.")
            print(f"P&L: ${existing.profit:.2f}")
            return None

        print(f"\nClosing {pos_type} position (new signal: {new_signal})...")
        if not close_position(existing):
            print("Failed to close existing position!")
            return None
        time.sleep(1)

    result = place_order(signal_info)
    if result:
        print(f"\nTrade placed successfully. Ticket: {result.order}")
    return result


def run_bot_h4():
    print("Starting Forex Model Trading Bot (H4)")
    print("=" * 50)

    print("\n[1/2] Training model on fresh data...")
    from trainer import run_training
    model, scaler = run_training()
    print("Model trained and saved.")

    print("\n[2/2] Connecting to MT5...")
    if not connect_mt5():
        print("Cannot connect to MT5. Make sure MT5 is running and logged in.")
        return

    try:
        while True:
            now = datetime.datetime.utcnow()
            next_h4_hour = None
            for h in [0, 4, 8, 12, 16, 20]:
                if h > now.hour or (h == now.hour and now.minute < 5):
                    next_h4_hour = h
                    break
            if next_h4_hour is None:
                next_h4_hour = 0
                wait_hours = 24 - now.hour + next_h4_hour
                wait_seconds = wait_hours * 3600 - now.minute * 60 - now.second
            else:
                wait_seconds = (next_h4_hour - now.hour) * 3600 - now.minute * 60 - now.second

            if wait_seconds < 300:
                print(f"\n--- H4 bar close at {next_h4_hour:02d}:00 UTC ---")
                print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                run_trading_cycle()
                time.sleep(300)
            else:
                print(f"Next H4 bar in {wait_seconds // 60} minutes. Sleeping...")
                time.sleep(min(wait_seconds, 300))
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        disconnect_mt5()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python trader.py connect                     - Test MT5 connection")
        print("  python trader.py signal                       - Get current signal")
        print("  python trader.py trade                        - Execute one trade")
        print("  python trader.py close                        - Close current position")
        print("  python trader.py bot                          - Run H4 trading bot")
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "connect":
        if connect_mt5():
            account = mt5.account_info()
            symbol_info = get_symbol_info()
            ask, bid = get_current_price()
            print(f"\nEURUSD Ask: {ask:.5f}  Bid: {bid:.5f}")
            position = get_open_position()
            if position:
                pos_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                print(f"Open position: {pos_type} {position.volume:.2f} lots P&L=${position.profit:.2f}")
            else:
                print("No open positions.")
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
        check_and_close_existing()
        disconnect_mt5()

    elif cmd == "bot":
        run_bot_h4()

    else:
        print(f"Unknown command: {cmd}")