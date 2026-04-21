import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import (
    DATA_CONFIG, BACKTEST_CONFIG, RISK_CONFIG,
    SCALER_SAVE_PATH,
    FEATURE_COLUMNS,
)
from data_loader import load_raw_prices, compute_atr, compute_adx, build_dataset
from ensemble import predict_direction_proba_all

VOL_MODEL_PATH = "models/vol_model.json"
VOL_SCALER_PATH = "models/vol_scaler.pkl"


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


def run_full_backtest():
    feats_df = build_dataset(force_download=False)
    prices_df = load_raw_prices()

    common_idx = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)

    dir_probs_ens, dir_probs_detail = predict_direction_proba_all(feats_df.values)
    dir_probs = dir_probs_ens

    vol_model = xgb.XGBRegressor()
    vol_model.load_model(VOL_MODEL_PATH)
    with open(VOL_SCALER_PATH, "rb") as f:
        vol_scaler = pickle.load(f)
    vol_scaled = vol_scaler.transform(feats_df.values)
    vol_preds = vol_model.predict(vol_scaled)
    vol_percentile_arr = pd.Series(vol_preds).rolling(252, min_periods=30).quantile(0.20).values
    atr_pred_median = vol_percentile_arr

    close = prices_df["close"].values
    high = prices_df["high"].values
    low = prices_df["low"].values

    atr_series = compute_atr(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])
    atr_mean = atr_series.rolling(60, min_periods=1).mean()
    adx_series = compute_adx(pd.Series(high), pd.Series(low), pd.Series(close), DATA_CONFIG["atr_period"])

    no_trade_buy_above = DATA_CONFIG["no_trade_buy_above"]
    no_trade_sell_below = DATA_CONFIG["no_trade_sell_below"]
    tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
    min_sl_pips = DATA_CONFIG.get("min_sl_pips", 40.0)
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
    min_adx = DATA_CONFIG.get("min_adx", 0.20)

    mask = (feats_df.index >= BACKTEST_CONFIG["start_date"]) & \
           (feats_df.index < BACKTEST_CONFIG["end_date"])
    bt_indices = np.where(mask)[0]
    bt_indices = bt_indices[bt_indices < len(prices_df) - 1]

    equity = initial_equity
    equity_curve = [initial_equity]
    dates_list = []
    all_dir_p = []
    all_signals = np.zeros(len(bt_indices))
    all_pnl = np.zeros(len(bt_indices))
    all_lots_list = []
    all_pips_list = []

    trade_pips = []
    trade_pnl_dollars = []
    trade_durations = []
    long_count = short_count = wins = losses = 0
    long_wins = long_losses = short_wins = short_losses = 0
    total_sl_pips = total_tp_pips = 0.0
    avg_sl_pips = 0.0
    avg_tp_pips = 0.0
    max_lots_used = 0.0

    n_filtered_vol = 0

    for i, idx in enumerate(bt_indices):
        prob = dir_probs[idx]
        vol_pred = vol_preds[idx]
        vol_median_val = atr_pred_median[idx] if idx < len(atr_pred_median) else vol_pred
        vol_high = vol_pred > vol_median_val

        current_price = close[idx]
        next_price = close[idx + 1]
        price_move_pips = (next_price - current_price) * 10000

        atr_val = atr_series.iloc[idx] if idx < len(atr_series) else np.nan
        atr_mean_val = atr_mean.iloc[idx] if idx < len(atr_mean) else np.nan
        adx_val = adx_series.iloc[idx] if idx < len(adx_series) else 0.5
        adx_ok = adx_val >= min_adx

        signal = 0.0
        lots = 0.0
        pips_result = 0.0
        pnl = 0.0

        if not np.isnan(atr_val) and atr_val > 0 and adx_ok:
            if prob > no_trade_buy_above:
                signal = 1.0
            elif prob < no_trade_sell_below:
                signal = -1.0

        if signal != 0 and not vol_high:
            signal = 0.0
            n_filtered_vol += 1

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
            if lots > max_lots_used:
                max_lots_used = lots

            use_trailing = DATA_CONFIG.get("trailing_stop", False)
            breakeven_pips = DATA_CONFIG.get("breakeven_pips", 30)
            trail_pips = DATA_CONFIG.get("trail_pips", 25)
            max_hold = DATA_CONFIG.get("max_hold_bars", 10)

            if use_trailing:
                entry_price = current_price
                cur_sl = entry_price - sl_pips / 10000 if signal > 0 else entry_price + sl_pips / 10000
                cur_tp = entry_price + tp_pips / 10000 if signal > 0 else entry_price - tp_pips / 10000
                highest = entry_price if signal > 0 else entry_price
                lowest = entry_price if signal > 0 else entry_price
                be_triggered = False
                trade_closed = False

                for hold_bar in range(1, max_hold + 1):
                    if idx + hold_bar >= len(close):
                        last_price = close[-1]
                        if signal > 0:
                            pips_result = (last_price - entry_price) * 10000
                        else:
                            pips_result = (entry_price - last_price) * 10000
                        trade_closed = True
                        break

                    bar_high = high[idx + hold_bar]
                    bar_low = low[idx + hold_bar]
                    bar_close = close[idx + hold_bar]

                    if signal > 0:
                        highest = max(highest, bar_high)
                        if bar_high >= cur_tp:
                            pips_result = tp_pips
                            trade_closed = True
                            break
                        if bar_low <= cur_sl:
                            pips_result = (cur_sl - entry_price) * 10000
                            trade_closed = True
                            break
                        unrealized = (bar_high - entry_price) * 10000
                        if not be_triggered and unrealized >= breakeven_pips:
                            cur_sl = entry_price
                            be_triggered = True
                        if be_triggered:
                            new_sl = highest - trail_pips / 10000
                            if new_sl > cur_sl:
                                cur_sl = new_sl
                    else:
                        lowest = min(lowest, bar_low)
                        if bar_low <= cur_tp:
                            pips_result = tp_pips
                            trade_closed = True
                            break
                        if bar_high >= cur_sl:
                            pips_result = (entry_price - cur_sl) * 10000
                            trade_closed = True
                            break
                        unrealized = (entry_price - bar_low) * 10000
                        if not be_triggered and unrealized >= breakeven_pips:
                            cur_sl = entry_price
                            be_triggered = True
                        if be_triggered:
                            new_sl = lowest + trail_pips / 10000
                            if new_sl < cur_sl:
                                cur_sl = new_sl

                if not trade_closed:
                    last_price = close[idx + max_hold] if idx + max_hold < len(close) else close[-1]
                    if signal > 0:
                        pips_result = (last_price - entry_price) * 10000
                    else:
                        pips_result = (entry_price - last_price) * 10000
                    if be_triggered:
                        if signal > 0:
                            trail_sl_pips = (highest - trail_pips / 10000 - entry_price) * 10000
                        else:
                            trail_sl_pips = (entry_price - (lowest + trail_pips / 10000)) * 10000
                        if trail_sl_pips > 0:
                            pips_result = max(pips_result, trail_sl_pips)
            else:
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

        all_dir_p.append(prob)
        all_signals[i] = signal
        all_pnl[i] = pnl
        all_lots_list.append(lots)
        all_pips_list.append(pips_result)
        dates_list.append(feats_df.index[idx])
        equity_curve.append(equity)

    n_trades = wins + losses
    final_equity = equity
    net_profit = final_equity - initial_equity
    md_pct = max_dd_from_equity(np.array(equity_curve))
    md_dollars = md_pct * initial_equity
    pf_dollars = profit_factor(np.array(trade_pnl_dollars)) if len(trade_pnl_dollars) > 0 else 0.0

    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

    wr = wins / n_trades * 100 if n_trades > 0 else 0.0
    long_wr = long_wins / (long_wins + long_losses) * 100 if (long_wins + long_losses) > 0 else 0.0
    short_wr = short_wins / (short_wins + short_losses) * 100 if (short_wins + short_losses) > 0 else 0.0

    winning_pips = np.array([p for p in trade_pips if p > 0]) if len(trade_pips) > 0 else np.array([])
    losing_pips = np.array([p for p in trade_pips if p < 0]) if len(trade_pips) > 0 else np.array([])
    avg_win_pips = np.mean(winning_pips) if len(winning_pips) > 0 else 0.0
    avg_loss_pips = abs(np.mean(losing_pips)) if len(losing_pips) > 0 else 0.0
    wl_ratio = avg_win_pips / avg_loss_pips if avg_loss_pips > 0 else float("inf")

    winning_dollars = np.array([p for p in trade_pnl_dollars if p > 0]) if len(trade_pnl_dollars) > 0 else np.array([])
    losing_dollars = np.array([p for p in trade_pnl_dollars if p < 0]) if len(trade_pnl_dollars) > 0 else np.array([])
    avg_win_dollars = np.mean(winning_dollars) if len(winning_dollars) > 0 else 0.0
    avg_loss_dollars = abs(np.mean(losing_dollars)) if len(losing_dollars) > 0 else 0.0

    n_long = int((all_signals > 0).sum())
    n_short = int((all_signals < 0).sum())
    n_total = len(bt_indices)

    actual_dir = np.sign(np.diff(close[bt_indices[0]:bt_indices[-1]+2]))
    pred_dir = (np.array(all_dir_p) > 0.5).astype(float) if len(all_dir_p) > 1 else np.array([])
    da_raw = np.mean(pred_dir == actual_dir[:len(pred_dir)]) * 100 if len(pred_dir) > 0 else 0

    traded_mask = all_signals != 0
    da_traded = np.mean(pred_dir[traded_mask[:len(pred_dir)]] == actual_dir[:len(pred_dir)][traded_mask[:len(pred_dir)]]) * 100 if traded_mask[:len(pred_dir)].sum() > 0 else 0

    prec_long = long_wins / (long_wins + long_losses) * 100 if (long_wins + long_losses) > 0 else 0
    prec_short = short_wins / (short_wins + short_losses) * 100 if (short_wins + short_losses) > 0 else 0

    cum_hold = np.cumprod(1 + np.diff(close[bt_indices[0]:bt_indices[-1]+2]) / close[bt_indices[0]:bt_indices[-1]+1]) - 1
    hold_final = initial_equity * (1 + cum_hold[-1])

    total_pips = sum(trade_pips) if len(trade_pips) > 0 else 0
    avg_lots = np.mean([l for l in all_lots_list if l > 0]) if any(l > 0 for l in all_lots_list) else 0

    dates = pd.to_datetime(dates_list)
    all_equity = np.array(equity_curve[1:])

    print(f"\n{'='*70}")
    print(" /$$$$$$$$                                       /$$$$$$  /$$$$$$")
    print("| $$_____/                                      /$$__  $$|_  $$_/")
    print("| $$     /$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$| $$  \\ $$  | $$  ")
    print("| $$$$$ /$$__  $$ /$$__  $$ /$$__  $$|  $$ /$$/| $$$$$$$$  | $$  ")
    print("| $$__/| $$  \\ $$| $$  \\__/| $$$$$$$$ \\  $$$$/ | $$__  $$  | $$  ")
    print("| $$   | $$  | $$| $$      | $$_____/  >$$  $$ | $$  | $$  | $$  ")
    print("| $$   |  $$$$$$/| $$      |  $$$$$$$ /$$/\\  $$| $$  | $$ /$$$$$$")
    print("|__/    \\______/ |__/       \\_______/|__/  \\__/|__/  |__/|______/")
    print(f"  BACKTEST {BACKTEST_CONFIG['start_date']} - {BACKTEST_CONFIG['end_date']} | Strategy: XGBoost + Vol + Trail | Bars: {n_total}")
    print(f"{'='*70}")
    print(f"  TP/SL Ratio  : {tp_sl_ratio}x (dynamic vol-scaled)")
    print(f"  No-Trade Zone: BUY >{no_trade_buy_above}, SELL <{no_trade_sell_below}")
    print(f"  Vol Filter   : ATR_pred > 20th pct ({n_filtered_vol} filtered)")
    print(f"  Risk/Trade   : {risk_per_trade*100:.1f}% | Max Leverage: {max_leverage:.0f}:1")
    print(f"  Spread       : {spread_pips:.1f} pips | Commission: ${commission_per_lot:.1f}/lot")
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
    if n_trades > 0:
        print(f"  Avg SL/TP      : {total_sl_pips/n_trades:.1f} / {total_tp_pips/n_trades:.1f} pips")
    print(f"  Avg Win        : ${avg_win_dollars:,.2f} ({avg_win_pips:.1f} pips)")
    print(f"  Avg Loss       : ${avg_loss_dollars:,.2f} ({avg_loss_pips:.1f} pips)")
    print(f"  W/L Ratio      : {wl_ratio:.2f}x (pips)")
    print(f"")
    print(f"  --- PERFORMANCE ---")
    print(f"  Long WR        : {long_wr:.1f}% ({long_wins}/{long_wins+long_losses})")
    print(f"  SHORT WR       : {short_wr:.1f}% ({short_wins}/{short_wins+short_losses})")
    print(f"  Profit Factor  : {pf_dollars:.2f}")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Max Drawdown   : {abs(md_pct)*100:.1f}% (${abs(md_dollars):,.0f})")
    print(f"  Win Rate       : {wr:.1f}% ({wins}/{n_trades})")
    print(f"  Trades         : {n_trades} / {n_total} bars ({n_trades/n_total*100:.1f}%)")
    print(f"  LONG / SHORT   : {n_long} / {n_short}")
    print(f"  Buy & Hold     : {cum_hold[-1]*100:.1f}% (${hold_final:,.0f})")
    print(f"{'='*70}")

    bt_dir = "backtest"
    os.makedirs(bt_dir, exist_ok=True)

    results = pd.DataFrame({
        "date": dates,
        "equity": all_equity,
        "dir_prob": all_dir_p,
        "signal": all_signals,
        "lots": all_lots_list,
        "pips": all_pips_list,
        "pnl": all_pnl,
    })
    results.to_csv(os.path.join(bt_dir, "backtest_results_vol.csv"), index=False)

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, all_equity, label="Vol Filter Strategy", color="green", linewidth=2.0)
    ax1.plot(dates, initial_equity * (1 + cum_hold), label="Buy & Hold", color="gray", linewidth=1.0, linestyle="--")
    ax1.axhline(initial_equity, color="black", linewidth=0.5, alpha=0.5)
    ax1.set_title(f"Equity Curve (Vol Filter) | Net: ${net_profit:,.0f} ({(final_equity/initial_equity-1)*100:.1f}%) | PF={pf_dollars:.2f} | Sharpe={sharpe:.2f}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = fig.add_subplot(gs[1, 0])
    correct = (pred_dir == actual_dir[:len(pred_dir)]).astype(int)
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
    up_mask = np.diff(close[bt_indices[0]:bt_indices[-1]+2]) > 0
    ax3.hist(np.array(all_dir_p)[up_mask[:len(all_dir_p)]], bins=50, alpha=0.6, label="Actual UP", color="green", density=True)
    ax3.hist(np.array(all_dir_p)[~up_mask[:len(all_dir_p)]], bins=50, alpha=0.6, label="Actual DOWN", color="red", density=True)
    ax3.axvline(no_trade_buy_above, color="green", linestyle="-", linewidth=1.5, label=f"BUY >{no_trade_buy_above}")
    ax3.axvline(no_trade_sell_below, color="red", linestyle="-", linewidth=1.5, label=f"SELL <{no_trade_sell_below}")
    ax3.axvspan(no_trade_sell_below, no_trade_buy_above, alpha=0.15, color="gray", label="No-Trade")
    ax3.set_title("P(UP) Distribution (Vol Filter)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("P(UP)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    rolling_medians = np.array([atr_pred_median[idx] if idx < len(atr_pred_median) else np.median(vol_preds[:max(idx,1)]) for idx in bt_indices])
    vol_filter_colors = ["green" if vol_preds[bt_indices[i]] > rolling_medians[i] else "red" for i in range(len(rolling_medians))]
    ax4.scatter(np.array(all_dir_p), vol_preds[bt_indices[:len(all_dir_p)]],
                c=vol_filter_colors[:len(all_dir_p)], alpha=0.3, s=10)
    global_median = np.median(vol_preds)
    ax4.axhline(global_median, color="yellow", linestyle="--", linewidth=1, label=f"Vol Median={global_median:.5f}")
    ax4.axvline(no_trade_buy_above, color="green", linestyle=":", alpha=0.5)
    ax4.axvline(no_trade_sell_below, color="red", linestyle=":", alpha=0.5)
    ax4.set_title("Vol Filter: P(UP) vs Predicted ATR", fontsize=12, fontweight="bold")
    ax4.set_xlabel("P(UP)")
    ax4.set_ylabel("Predicted ATR")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    running_max = np.maximum.accumulate(all_equity)
    drawdown_pct = (all_equity - running_max) / running_max * 100
    ax5.fill_between(dates, drawdown_pct, 0, color="red", alpha=0.4)
    ax5.set_title(f"Drawdown (Max: {abs(md_pct)*100:.1f}% / ${abs(md_dollars):,.0f})", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Drawdown %")
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    ax6 = fig.add_subplot(gs[3, 0])
    if len(trade_pips) > 0:
        trade_pips_arr = np.array(trade_pips)
        trade_pnl_arr = np.array(trade_pnl_dollars)
        colors_pips = ["green" if p > 0 else "red" for p in trade_pnl_arr]
        trade_dates_pd = dates[traded_mask[:len(dates)]]
        if len(trade_dates_pd) > 0:
            ax6.bar(trade_dates_pd[:len(trade_pnl_arr)], trade_pnl_arr, color=colors_pips, alpha=0.7, width=0.8)
        ax6.axhline(0, color="black", linewidth=0.5)
        ax6.axhline(avg_win_dollars, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label=f"Avg Win ${avg_win_dollars:.0f}")
        ax6.axhline(-avg_loss_dollars, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label=f"Avg Loss ${avg_loss_dollars:.0f}")
        ax6.set_title(f"Trade P&L ($) | W/L Ratio: {wl_ratio:.2f}x", fontsize=12, fontweight="bold")
        ax6.set_ylabel("P&L ($)")
        ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    ax7 = fig.add_subplot(gs[3, 1])
    cats = ["LONG\nPrec", "SHORT\nPrec", "DA\nTraded", "Total\nWR", "LONG\nWR", "SHORT\nWR", "W/L\nRatio"]
    vals = [prec_long, prec_short, da_traded, wr, long_wr, short_wr, wl_ratio]
    colors_bar = ["green", "red", "steelblue", "purple", "limegreen", "salmon", "orange"]
    bars = ax7.bar(cats, vals, color=colors_bar, alpha=0.7, edgecolor="black")
    for bar, val in zip(bars, vals):
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax7.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax7.set_title("Strategy Metrics (Vol Filter)", fontsize=12, fontweight="bold")
    ax7.set_ylabel("%")
    ax7.set_ylim(0, max(max(vals) if vals else 10, 10) + 10)
    ax7.grid(True, alpha=0.3, axis="y")

    chart_path = os.path.join(bt_dir, "backtest_chart_vol.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {chart_path}")

    return {"net_profit": net_profit, "pf": pf_dollars, "sharpe": sharpe,
            "max_dd": abs(md_pct), "win_rate": wr, "n_trades": n_trades}


if __name__ == "__main__":
    run_full_backtest()