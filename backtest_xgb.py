import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from config import (
    DATA_CONFIG, BACKTEST_CONFIG, RISK_CONFIG,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS,
)
from data_loader import load_dataset, load_raw_prices, load_labels, build_dataset, compute_atr, compute_adx

REGIME_MODEL_PATH = "models/regime_model.json"
REGIME_SCALER_PATH = "models/regime_scaler.pkl"
ENSEMBLE_MODEL_DIR = "models/ensemble"
LONG_MODEL_PATH = "models/long_model.json"
SHORT_MODEL_PATH = "models/short_model.json"


def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss < 1e-10:
        return float("inf")
    return gross_profit / gross_loss


def max_dd_from_equity(equity_arr):
    running_max = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - running_max) / running_max
    return dd.min()


def run_backtest():
    feats_df = build_dataset(force_download=False)
    prices_df = load_raw_prices()
    labels_df = load_labels()

    common_idx = feats_df.index.intersection(labels_df.index)
    feats_df = feats_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]
    price_common = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[price_common]
    labels_df = labels_df.loc[price_common]
    prices_df = prices_df.loc[price_common]

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)

    scaled = scaler.transform(feats_df.values)
    no_trade_buy_above = DATA_CONFIG["no_trade_buy_above"]
    no_trade_sell_below = DATA_CONFIG["no_trade_sell_below"]

    use_ensemble = os.path.isdir(ENSEMBLE_MODEL_DIR) and len([f for f in os.listdir(ENSEMBLE_MODEL_DIR) if f.endswith(".json")]) >= 2

    if use_ensemble:
        ensemble_models = []
        for fname in sorted(os.listdir(ENSEMBLE_MODEL_DIR)):
            if fname.endswith(".json"):
                m = xgb.XGBClassifier()
                m.load_model(os.path.join(ENSEMBLE_MODEL_DIR, fname))
                ensemble_models.append(m)
        all_probs_list = []
        for m in ensemble_models:
            all_probs_list.append(m.predict_proba(scaled)[:, 1])
        all_probs = np.mean(all_probs_list, axis=0)
        print(f"Using ensemble of {len(ensemble_models)} models")
    else:
        model = xgb.XGBClassifier()
        model_path = MODEL_SAVE_PATH.replace(".pth", ".json")
        model.load_model(model_path)
        all_probs = model.predict_proba(scaled)[:, 1]

    regime_model = None
    regime_scaler = None
    if os.path.exists(REGIME_MODEL_PATH) and os.path.exists(REGIME_SCALER_PATH):
        regime_model = xgb.XGBClassifier()
        regime_model.load_model(REGIME_MODEL_PATH)
        with open(REGIME_SCALER_PATH, "rb") as f:
            regime_scaler = pickle.load(f)
        print("Regime model loaded")

    use_split_models = os.path.exists(LONG_MODEL_PATH) and os.path.exists(SHORT_MODEL_PATH)
    long_model_inst = None
    short_model_inst = None
    if use_split_models:
        long_model_inst = xgb.XGBClassifier()
        long_model_inst.load_model(LONG_MODEL_PATH)
        short_model_inst = xgb.XGBClassifier()
        short_model_inst.load_model(SHORT_MODEL_PATH)
        print("Split LONG/SHORT models loaded")

    tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
    no_trade_sell_below = DATA_CONFIG["no_trade_sell_below"]
    tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
    min_sl_pips = DATA_CONFIG.get("min_sl_pips", 20.0)
    risk_per_trade = RISK_CONFIG["risk_per_trade"]
    initial_equity = RISK_CONFIG["initial_equity"]
    pip_value = RISK_CONFIG["pip_value_per_lot"]
    contract_size = RISK_CONFIG["contract_size"]
    lot_step = RISK_CONFIG.get("lot_step", 0.01)
    min_lots = RISK_CONFIG.get("min_lots", 0.01)
    max_leverage = RISK_CONFIG["max_leverage"]
    spread_pips = RISK_CONFIG.get("spread_pips", 1.0)
    commission_per_lot = RISK_CONFIG.get("commission_per_lot", 3.5)
    compound = RISK_CONFIG.get("compound", False)
    session_filter = DATA_CONFIG.get("session_filter", False)
    session_start = DATA_CONFIG.get("session_start_utc", 7)
    session_end = DATA_CONFIG.get("session_end_utc", 20)
    min_adx = DATA_CONFIG.get("min_adx", 0.20)
    trend_filter = DATA_CONFIG.get("trend_filter", False)
    entry_on_open = DATA_CONFIG.get("entry_on_open", False)

    open_prices = prices_df["open"].values if "open" in prices_df.columns else close

    print("\nComputing feature importance (XGBoost)...")
    if use_ensemble:
        importances = np.mean([m.feature_importances_ for m in ensemble_models], axis=0)
    else:
        importances = model.feature_importances_
    print("\nFeature Importance (XGBoost gain):")
    sorted_pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])
    for fname, imp in sorted_pairs:
        bar = "#" * max(0, int(imp * 200))
        print(f"  {fname:20s} {imp:.4f} {bar}")

    if not use_ensemble:
        all_probs = model.predict_proba(scaled)[:, 1]

    regime_probs = None
    if regime_model is not None:
        regime_scaled = regime_scaler.transform(feats_df.values)
        regime_probs = regime_model.predict_proba(regime_scaled)[:, 1]

    close = prices_df["close"].values
    high = prices_df["high"].values
    low = prices_df["low"].values

    atr_series = compute_atr(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])
    atr_mean = atr_series.rolling(60, min_periods=1).mean()
    adx_series = compute_adx(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])

    close_series = pd.Series(close)
    sma50 = close_series.rolling(50, min_periods=1).mean()
    sma200 = close_series.rolling(200, min_periods=1).mean()

    n_adx = min(500, len(scaled))
    X_base = scaled[:n_adx]
    y_base = ((labels_df["label"].values[:n_adx] + 1) / 2.0)
    y_base = np.where(y_base == 0.5, 0, y_base).astype(int)
    mask_valid = y_base != -1
    base_probs = all_probs[:n_adx]
    base_preds = (base_probs[mask_valid] > 0.5).astype(int)
    base_acc = np.mean(base_preds == y_base[mask_valid])

    importance = np.zeros(len(FEATURE_COLUMNS))
    predict_fn = (lambda x: np.mean([m.predict_proba(x)[:, 1] for m in ensemble_models], axis=0)) if use_ensemble else (lambda x: model.predict_proba(x)[:, 1])
    for feat_idx in range(len(FEATURE_COLUMNS)):
        drops = []
        for _ in range(3):
            X_perm = X_base.copy()
            col = X_perm[:, feat_idx].flatten()
            np.random.shuffle(col)
            X_perm[:, feat_idx] = col.reshape(-1)
            perm_probs = predict_fn(X_perm)
            perm_preds = (perm_probs[mask_valid] > 0.5).astype(int)
            perm_acc = np.mean(perm_preds == y_base[mask_valid])
            drops.append(perm_acc)
        importance[feat_idx] = base_acc - np.mean(drops)

    print("\nPermutation Importance:")
    sorted_imp = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: -x[1])
    for fname, imp in sorted_imp:
        marker = "+" if imp > 0 else "-"
        bar = "#" * max(0, int(abs(imp) * 1000))
        print(f"  {fname:20s} {marker}{abs(imp):.4f} {bar}")

    mask = (feats_df.index >= BACKTEST_CONFIG["start_date"]) & \
           (feats_df.index < BACKTEST_CONFIG["end_date"])
    bt_indices = np.where(mask)[0]
    bt_indices = bt_indices[bt_indices < len(prices_df) - 1]

    if len(bt_indices) == 0:
        print("No backtest data available.")
        return

    equity = initial_equity
    equity_curve = [initial_equity]
    all_dates, all_equity, all_probs_tracked, all_signals = [], [], [], []
    all_pnl, all_lots, all_pips, actual_returns_pct = [], [], [], []
    trade_pips, trade_pnl_dollars, trade_lots_list = [], [], []
    long_count = short_count = wins = losses = 0
    long_wins = long_losses = short_wins = short_losses = 0
    total_sl_pips = total_tp_pips = 0.0

    for i, idx in enumerate(bt_indices):
        prob = all_probs[idx]
        if entry_on_open and idx + 1 < len(open_prices):
            current_price = open_prices[idx + 1]
            next_price = close[idx + 1]
        else:
            current_price = close[idx]
            next_price = close[idx + 1]
        price_move_pips = (next_price - current_price) * 10000

        atr_val = atr_series.iloc[idx] if idx < len(atr_series) else np.nan
        atr_mean_val = atr_mean.iloc[idx] if idx < len(atr_mean) else np.nan

        idx_date = feats_df.index[idx]
        signal = 0.0
        lots = 0.0
        pips_result = 0.0
        pnl = 0.0

        in_session = True
        if session_filter:
            hour = idx_date.hour
            if not (session_start <= hour < session_end):
                in_session = False

        adx_val = adx_series.iloc[idx] if idx < len(adx_series) else 0.5
        adx_ok = adx_val >= min_adx

        trend_up = True
        trend_down = True
        if trend_filter:
            sma50_val = sma50.iloc[idx] if idx < len(sma50) else None
            sma200_val = sma200.iloc[idx] if idx < len(sma200) else None
            if sma50_val is not None and sma200_val is not None and not np.isnan(sma50_val) and not np.isnan(sma200_val):
                trend_up = sma50_val > sma200_val
                trend_down = sma50_val < sma200_val

        if not np.isnan(atr_val) and atr_val > 0 and in_session and adx_ok:
            long_prob = prob
            short_prob = 1.0 - prob

            if use_split_models and long_model_inst is not None and short_model_inst is not None and idx < len(scaled):
                long_prob = long_model_inst.predict_proba(scaled[idx].reshape(1, -1))[0, 1]
                short_prob = short_model_inst.predict_proba(scaled[idx].reshape(1, -1))[0, 1]

            if long_prob > no_trade_buy_above:
                signal = 1.0
            elif short_prob > (1.0 - no_trade_sell_below):
                signal = -1.0

            if trend_filter and signal == 1.0 and not trend_up:
                signal = 0.0
            if trend_filter and signal == -1.0 and not trend_down:
                signal = 0.0

            if signal != 0.0:
                atr_pips = atr_val * 10000
                sl_pips = max(atr_pips, min_sl_pips)

                if not np.isnan(atr_mean_val) and atr_mean_val > 0:
                    vol_ratio = atr_val / atr_mean_val
                    dyn_tp_sl = tp_sl_ratio / max(vol_ratio, 0.5)
                    dyn_tp_sl = min(dyn_tp_sl, 4.0)
                else:
                    dyn_tp_sl = tp_sl_ratio

                tp_pips = sl_pips * dyn_tp_sl

                risk_base = initial_equity if not compound else equity
                risk_dollars = risk_base * risk_per_trade
                lots = risk_dollars / (sl_pips * pip_value)
                lots = round(lots / lot_step) * lot_step

                max_lots_val = (equity * max_leverage) / contract_size
                max_lots_val = round(max_lots_val / lot_step) * lot_step
                lots = min(lots, max_lots_val)
                lots = max(min_lots, lots)

                if signal > 0:
                    if price_move_pips >= tp_pips:
                        pips_result = tp_pips
                    elif price_move_pips <= -sl_pips:
                        pips_result = -sl_pips
                    else:
                        pips_result = price_move_pips
                else:
                    if price_move_pips <= -tp_pips:
                        pips_result = tp_pips
                    elif price_move_pips >= sl_pips:
                        pips_result = -sl_pips
                    else:
                        pips_result = -price_move_pips

                pnl = lots * pip_value * pips_result
                pnl -= lots * commission_per_lot
                pnl -= lots * pip_value * spread_pips
                equity += pnl

                total_sl_pips += sl_pips
                total_tp_pips += tp_pips
                trade_pips.append(pips_result)
                trade_pnl_dollars.append(pnl)
                trade_lots_list.append(lots)

                if signal > 0:
                    long_count += 1
                else:
                    short_count += 1

                if pnl > 0:
                    wins += 1
                    if signal > 0:
                        long_wins += 1
                    else:
                        short_wins += 1
                elif pnl < 0:
                    losses += 1
                    if signal > 0:
                        long_losses += 1
                    else:
                        short_losses += 1

        all_dates.append(idx_date)
        all_equity.append(equity)
        all_probs_tracked.append(prob)
        all_signals.append(signal)
        all_pnl.append(pnl)
        all_lots.append(lots)
        all_pips.append(pips_result)
        actual_ret = np.log(next_price / current_price) if current_price > 0 else 0.0
        actual_returns_pct.append(actual_ret)

    all_equity = np.array(all_equity)
    all_probs_tracked = np.array(all_probs_tracked)
    all_signals = np.array(all_signals)
    all_pnl = np.array(all_pnl)
    trade_pips = np.array(trade_pips)
    trade_pnl_dollars = np.array(trade_pnl_dollars)
    trade_lots_arr = np.array(trade_lots_list) if trade_lots_list else np.array([0])
    actual_returns_pct = np.array(actual_returns_pct)
    dates = pd.to_datetime(all_dates)

    n_trades = wins + losses
    n_total = len(all_signals)
    trade_freq = n_trades / n_total * 100 if n_total > 0 else 0

    final_equity = all_equity[-1]
    net_profit = final_equity - initial_equity
    total_pips = trade_pips.sum() if len(trade_pips) > 0 else 0
    avg_lots = trade_lots_arr.mean() if len(trade_lots_arr) > 0 else 0
    max_lots_used = trade_lots_arr.max() if len(trade_lots_arr) > 0 else 0
    avg_sl_pips = total_sl_pips / n_trades if n_trades > 0 else 0
    avg_tp_pips = total_tp_pips / n_trades if n_trades > 0 else 0
    md_pct = max_dd_from_equity(all_equity)
    dd_arr = all_equity - np.maximum.accumulate(all_equity)
    md_dollars_exact = dd_arr[np.argmin(dd_arr)]
    pf_dollars = profit_factor(trade_pnl_dollars) if len(trade_pnl_dollars) > 0 else 0.0

    daily_returns = np.diff(all_equity) / all_equity[:-1]
    bars_per_day = 1
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252 * bars_per_day)

    actual_dir = (actual_returns_pct > 0).astype(float)
    pred_dir = (all_probs_tracked > 0.5).astype(float)
    da_raw = np.mean(pred_dir == actual_dir) * 100
    traded_mask = all_signals != 0
    da_traded = np.mean(pred_dir[traded_mask] == actual_dir[traded_mask]) * 100 if traded_mask.any() else 0.0

    n_long = int((all_signals > 0).sum())
    n_short = int((all_signals < 0).sum())
    prec_long = np.mean(actual_dir[all_signals > 0] == 1) * 100 if n_long > 0 else 0.0
    prec_short = np.mean(actual_dir[all_signals < 0] == 0) * 100 if n_short > 0 else 0.0

    long_wr = long_wins / (long_wins + long_losses) * 100 if (long_wins + long_losses) > 0 else 0.0
    short_wr = short_wins / (short_wins + short_losses) * 100 if (short_wins + short_losses) > 0 else 0.0

    winning_pips = trade_pips[trade_pips > 0] if len(trade_pips) > 0 else np.array([])
    losing_pips = trade_pips[trade_pips < 0] if len(trade_pips) > 0 else np.array([])
    winning_dollars = trade_pnl_dollars[trade_pnl_dollars > 0] if len(trade_pnl_dollars) > 0 else np.array([])
    losing_dollars = trade_pnl_dollars[trade_pnl_dollars < 0] if len(trade_pnl_dollars) > 0 else np.array([])
    avg_win_pips = np.mean(winning_pips) if len(winning_pips) > 0 else 0.0
    avg_loss_pips = abs(np.mean(losing_pips)) if len(losing_pips) > 0 else 0.0
    avg_win_dollars = np.mean(winning_dollars) if len(winning_dollars) > 0 else 0.0
    avg_loss_dollars = abs(np.mean(losing_dollars)) if len(losing_dollars) > 0 else 0.0
    wl_ratio_pips = avg_win_pips / avg_loss_pips if avg_loss_pips > 0 else float("inf")

    wr = wins / n_trades * 100 if n_trades > 0 else 0.0
    cum_hold = np.cumprod(1 + actual_returns_pct) - 1
    hold_final = initial_equity * (1 + cum_hold[-1])

    tf_str = "XGBoost H4"

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS ({BACKTEST_CONFIG['start_date']} - {BACKTEST_CONFIG['end_date']})")
    print(f"  Timeframe: {tf_str} | Bars: {n_total}")
    print(f"{'='*70}")
    print(f"  TP/SL Ratio  : {tp_sl_ratio}x (dynamic vol-scaled)")
    print(f"  No-Trade Zone: BUY >{no_trade_buy_above}, SELL <{no_trade_sell_below}")
    print(f"  Risk/Trade   : {risk_per_trade*100:.1f}% | Max Leverage: {max_leverage:.0f}:1")
    print(f"  Spread       : {spread_pips:.1f} pips | Commission: ${commission_per_lot:.1f}/lot")
    print(f"  Compound     : {'Yes' if compound else 'No'}")
    sf_str = f"{session_start}-{session_end} UTC" if session_filter else "OFF"
    tf_str2 = f" | Trend: ON" if trend_filter else ""
    print(f"  Session      : {sf_str} | Min ADX: {min_adx:.2f}{tf_str2}")
    print(f"")
    print(f"  --- DIRECTIONAL ACCURACY ---")
    print(f"  DA (raw)       : {da_raw:.1f}%")
    print(f"  DA (traded)    : {da_traded:.1f}%")
    print(f"  LONG Precision : {prec_long:.1f}% ({long_count} trades)")
    print(f"  SHORT Precision: {prec_short:.1f}% ({short_count} trades)")
    print(f"")
    print(f"  --- FINANCIAL SUMMARY ---")
    print(f"  Deposit        : ${initial_equity:,.0f}")
    print(f"  Net Profit     : ${net_profit:,.0f} ({(final_equity/initial_equity - 1)*100:.1f}%)")
    print(f"  Total Balance  : ${final_equity:,.0f}")
    print(f"  Total Pips     : {total_pips:,.1f}")
    print(f"")
    print(f"  --- POSITION SIZING ---")
    print(f"  Avg Lots       : {avg_lots:.2f} ({avg_lots*contract_size:,.0f} units)")
    print(f"  Max Lots       : {max_lots_used:.2f} ({max_lots_used*contract_size:,.0f} units)")
    print(f"  Avg SL/TP      : {avg_sl_pips:.1f} / {avg_tp_pips:.1f} pips")
    print(f"  Avg Win        : ${avg_win_dollars:,.2f} ({avg_win_pips:.1f} pips)")
    print(f"  Avg Loss       : ${avg_loss_dollars:,.2f} ({avg_loss_pips:.1f} pips)")
    print(f"  W/L Ratio      : {wl_ratio_pips:.2f}x (pips)")
    print(f"")
    print(f"  --- PERFORMANCE ---")
    print(f"  Long WR        : {long_wr:.1f}% ({long_wins}/{long_wins+long_losses})")
    print(f"  SHORT WR       : {short_wr:.1f}% ({short_wins}/{short_wins+short_losses})")
    print(f"  Profit Factor  : {pf_dollars:.2f}")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Max Drawdown   : {abs(md_pct)*100:.1f}% (${abs(md_dollars_exact):,.0f})")
    print(f"  Win Rate       : {wr:.1f}% ({wins}/{n_trades})")
    print(f"  Trades         : {n_trades} / {n_total} bars ({trade_freq:.1f}%)")
    print(f"  LONG / SHORT   : {n_long} / {n_short}")
    print(f"  Buy & Hold     : {cum_hold[-1]*100:.1f}% (${hold_final:,.0f})")
    print(f"{'='*70}")

    bt_dir = "backtest"
    os.makedirs(bt_dir, exist_ok=True)

    results = pd.DataFrame({
        "date": dates,
        "equity": all_equity,
        "dir_prob": all_probs_tracked,
        "signal": all_signals,
        "lots": all_lots,
        "pips": all_pips,
        "pnl": all_pnl,
        "actual_return": actual_returns_pct,
        "cum_hold": cum_hold,
    })
    results.to_csv(os.path.join(bt_dir, "backtest_results_xgb.csv"), index=False)
    print(f"Results saved to {bt_dir}/backtest_results_xgb.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, all_equity, label="XGBoost Strategy", color="green", linewidth=2.0)
    ax1.plot(dates, initial_equity * (1 + cum_hold), label="Buy & Hold", color="gray", linewidth=1.0, linestyle="--")
    ax1.axhline(initial_equity, color="black", linewidth=0.5, alpha=0.5)
    ax1.set_title(f"Equity Curve (XGBoost H4) | Net: ${net_profit:,.0f} ({(final_equity/initial_equity-1)*100:.1f}%) | PF={pf_dollars:.2f} | Sharpe={sharpe:.2f}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = fig.add_subplot(gs[1, 0])
    correct = (pred_dir == actual_dir).astype(int)
    cum_da = np.cumsum(correct) / np.arange(1, len(correct) + 1) * 100
    ax2.plot(dates, cum_da, color="green", linewidth=1.5)
    ax2.axhline(50, color="gray", linestyle="--", linewidth=1.0)
    ax2.axhline(55, color="orange", linestyle=":", linewidth=0.8, label="55%")
    ax2.axhline(60, color="green", linestyle=":", linewidth=0.8, label="60%")
    ax2.set_title("Cumulative Directional Accuracy", fontsize=12, fontweight="bold")
    ax2.set_ylabel("DA %")
    ax2.set_ylim(40, 80)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = fig.add_subplot(gs[1, 1])
    up_mask = actual_returns_pct > 0
    ax3.hist(all_probs_tracked[up_mask], bins=50, alpha=0.6, label="Actual UP", color="green", density=True)
    ax3.hist(all_probs_tracked[~up_mask], bins=50, alpha=0.6, label="Actual DOWN", color="red", density=True)
    ax3.axvline(no_trade_buy_above, color="green", linestyle="-", linewidth=1.5, label=f"BUY >{no_trade_buy_above}")
    ax3.axvline(no_trade_sell_below, color="red", linestyle="-", linewidth=1.5, label=f"SELL <{no_trade_sell_below}")
    ax3.axvspan(no_trade_sell_below, no_trade_buy_above, alpha=0.15, color="gray", label="No-Trade")
    ax3.set_title(f"P(UP) Distribution (XGBoost)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("P(UP)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    if importance is not None and len(importance) == len(FEATURE_COLUMNS):
        sorted_idx = np.argsort(importance)
        sorted_features = [FEATURE_COLUMNS[i] for i in sorted_idx]
        sorted_imp = importance[sorted_idx] * 100
        colors_fi = ["green" if v > 0.1 else ("orange" if v > 0 else "red") for v in sorted_imp]
        ax4.barh(sorted_features, sorted_imp, color=colors_fi)
        ax4.axvline(0, color="black", linewidth=0.5)
        ax4.set_title("Permutation Importance", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Importance (%)")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    running_max = np.maximum.accumulate(all_equity)
    drawdown_pct = (all_equity - running_max) / running_max * 100
    ax5.fill_between(dates, drawdown_pct, 0, color="red", alpha=0.4)
    ax5.set_title(f"Drawdown (Max: {abs(md_pct)*100:.1f}% / ${abs(md_dollars_exact):,.0f})", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Drawdown %")
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    ax6 = fig.add_subplot(gs[3, 0])
    traded_mask = all_signals != 0
    if np.any(traded_mask) and len(trade_pips) > 0:
        trade_pips_full = np.array(all_pips)
        trade_dates = dates[traded_mask]
        trade_pips_plot = trade_pips_full[traded_mask]
        colors_pips = ["green" if p > 0 else "red" for p in trade_pips_plot]
        ax6.bar(trade_dates, trade_pips_plot, color=colors_pips, alpha=0.7, width=0.8)
        ax6.axhline(0, color="black", linewidth=0.5)
        ax6.axhline(avg_win_pips, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label=f"Avg Win {avg_win_pips:.1f}p")
        ax6.axhline(-avg_loss_pips, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label=f"Avg Loss {avg_loss_pips:.1f}p")
        ax6.set_title(f"Pips per Trade (Total: {total_pips:,.0f})", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Pips")
        ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    ax7 = fig.add_subplot(gs[3, 1])
    cats = ["LONG\nPrec", "SHORT\nPrec", "DA\nTraded", "Total\nWR", "LONG\nWR", "SHORT\nWR", "W/L\nRatio"]
    vals = [prec_long, prec_short, da_traded, wr, long_wr, short_wr, wl_ratio_pips]
    colors_bar = ["green", "red", "steelblue", "purple", "limegreen", "salmon", "orange"]
    bars = ax7.bar(cats, vals, color=colors_bar, alpha=0.7, edgecolor="black")
    for bar, val in zip(bars, vals):
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax7.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax7.set_title("Strategy Metrics (XGBoost H4)", fontsize=12, fontweight="bold")
    ax7.set_ylabel("%")
    ax7.set_ylim(0, max(max(vals) if vals else 10, 10) + 10)
    ax7.grid(True, alpha=0.3, axis="y")

    chart_path = os.path.join(bt_dir, "backtest_chart_xgb.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {chart_path}")

    return {
        "net_profit": net_profit, "final_equity": final_equity,
        "pf": pf_dollars, "sharpe": sharpe,
        "max_dd_pct": abs(md_pct), "max_dd_dollars": abs(md_dollars_exact),
        "win_rate": wr, "wl_ratio": wl_ratio_pips,
        "n_trades": n_trades, "da_traded": da_traded,
        "total_pips": total_pips, "avg_lots": avg_lots,
        "max_lots": max_lots_used,
    }


if __name__ == "__main__":
    run_backtest()