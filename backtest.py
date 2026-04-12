import os
import pickle

import numpy as np
import pandas as pd
import torch

from config import (
    DATA_CONFIG, BACKTEST_CONFIG, DEVICE, RISK_CONFIG,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, FEATURE_WEIGHTS,
)
from data_loader import load_dataset, load_raw_prices, load_labels, build_dataset
from model import ForexClassifier


def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss < 1e-10:
        return float("inf")
    return gross_profit / gross_loss


def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(curve)
    dd = (curve - running_max) / running_max
    return dd.min()


def run_backtest():
    feats_df = load_dataset()
    if feats_df.empty:
        feats_df = build_dataset()

    prices_df = load_raw_prices()
    labels_df = load_labels()

    common_idx = feats_df.index.intersection(labels_df.index)
    feats_df = feats_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]
    price_common = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[price_common]
    labels_df = labels_df.loc[price_common]
    prices_df = prices_df.loc[price_common]

    eurusd = prices_df["EUR_USD"].values

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    scaled = scaler.transform(feats_df.values)
    lookback = DATA_CONFIG["lookback"]
    no_trade_low = DATA_CONFIG["no_trade_low"]
    no_trade_high = DATA_CONFIG["no_trade_high"]
    tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
    risk_pct = RISK_CONFIG["risk_per_trade"]
    max_pos_frac = RISK_CONFIG["max_position_fraction"]
    initial_equity = RISK_CONFIG["initial_equity"]

    # Feature importance
    print("\nComputing feature importance...")
    model.eval()
    n_imp = min(500, len(scaled))
    X_base_seq = np.array([scaled[i:i+lookback] for i in range(n_imp)])
    X_base = torch.tensor(X_base_seq, dtype=torch.float32).to(DEVICE)
    y_base_np = ((labels_df["label"].values[:n_imp] + 1) / 2.0)

    with torch.no_grad():
        base_probs = model.predict_proba(X_base).detach().cpu().numpy()
        base_preds = (base_probs > 0.5).astype(float)
        base_acc = np.mean(base_preds == y_base_np)

    importance = np.zeros(len(FEATURE_COLUMNS))
    for feat_idx in range(len(FEATURE_COLUMNS)):
        drops = []
        for _ in range(5):
            X_perm = X_base_seq.copy()
            col = X_perm[:, :, feat_idx].flatten()
            np.random.shuffle(col)
            X_perm[:, :, feat_idx] = col.reshape(n_imp, -1)
            X_perm_t = torch.tensor(X_perm, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                perm_probs = model.predict_proba(X_perm_t).detach().cpu().numpy()
                perm_preds = (perm_probs > 0.5).astype(float)
                perm_acc = np.mean(perm_preds == y_base_np)
                drops.append(perm_acc)
        importance[feat_idx] = base_acc - np.mean(drops)

    print("\nFeature Importance (accuracy drop when permuted):")
    sorted_pairs = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: -x[1])
    for fname, imp in sorted_pairs:
        bar = "#" * max(0, int(abs(imp) * 1000))
        marker = "+" if imp > 0 else "-"
        print(f"  {fname:20s} {marker}{abs(imp):.4f} {bar}")

    mask = (feats_df.index >= BACKTEST_CONFIG["start_date"]) & \
           (feats_df.index < BACKTEST_CONFIG["end_date"])
    bt_indices = np.where(mask)[0]
    bt_indices = bt_indices[bt_indices >= lookback]
    bt_indices = bt_indices[bt_indices < len(eurusd) - 1]

    if len(bt_indices) == 0:
        print("No backtest data available.")
        return

    atr_series = pd.Series(eurusd).diff().abs().rolling(DATA_CONFIG["atr_period"]).mean()
    atr_mean = atr_series.rolling(60).mean()
    atr_threshold = atr_series.rolling(60).quantile(DATA_CONFIG["atr_filter_quantile"])

    dates_list = []
    actual_returns = []
    dir_probs = []
    signals = []
    pos_fractions = []
    returns_tpsl = []

    with torch.no_grad():
        for idx in bt_indices:
            seq = scaled[idx - lookback:idx]
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prob = model.predict_proba(X).detach().cpu().item()

            current_price = eurusd[idx - 1]
            next_price = eurusd[idx]
            actual_ret = np.log(next_price / current_price) if current_price > 0 else 0.0

            idx_date = feats_df.index[idx]
            atr_val = atr_series.iloc[idx] if idx < len(atr_series) else np.nan
            atr_thr = atr_threshold.iloc[idx] if idx < len(atr_threshold) else np.nan
            atr_mean_val = atr_mean.iloc[idx] if idx < len(atr_mean) else np.nan

            atr_ok = not np.isnan(atr_val)

            signal = 0.0
            pos_frac = 0.0
            tm_ret = 0.0

            if atr_ok:
                if prob > no_trade_high:
                    signal = 1.0
                elif prob < no_trade_low:
                    signal = -1.0

                if signal != 0.0:
                    sl_pct = atr_val / current_price if atr_val > 0 and current_price > 0 else 0.01
                    if not np.isnan(atr_mean_val) and atr_mean_val > 0:
                        vol_ratio = atr_val / atr_mean_val
                        dyn_tp_sl = tp_sl_ratio / max(vol_ratio, 0.5)
                        dyn_tp_sl = min(dyn_tp_sl, 4.0)
                    else:
                        dyn_tp_sl = tp_sl_ratio

                    pos_frac = min(risk_pct / sl_pct, max_pos_frac) if sl_pct > 0 else 0.0
                    tp_pct = sl_pct * dyn_tp_sl

                    if signal > 0:
                        if actual_ret >= tp_pct:
                            tm_ret = tp_pct
                        elif actual_ret <= -sl_pct:
                            tm_ret = -sl_pct
                        else:
                            tm_ret = actual_ret
                    else:
                        if actual_ret <= -tp_pct:
                            tm_ret = tp_pct
                        elif actual_ret >= sl_pct:
                            tm_ret = -sl_pct
                        else:
                            tm_ret = -actual_ret

            dates_list.append(idx_date)
            actual_returns.append(actual_ret)
            dir_probs.append(prob)
            signals.append(signal)
            pos_fractions.append(pos_frac)
            returns_tpsl.append(tm_ret * pos_frac)

    actual_returns = np.array(actual_returns)
    dir_probs = np.array(dir_probs)
    signals = np.array(signals)
    pos_fractions = np.array(pos_fractions)
    returns_tpsl = np.array(returns_tpsl)
    dates = pd.to_datetime(dates_list)

    traded_mask = signals != 0
    n_trades = int(traded_mask.sum())
    trade_freq = traded_mask.sum() / len(traded_mask) * 100

    cum_tpsl = np.cumprod(1 + returns_tpsl) - 1
    cum_hold = np.cumprod(1 + actual_returns) - 1

    final_equity = initial_equity * (1 + cum_tpsl[-1])
    net_profit = final_equity - initial_equity
    md = max_drawdown(returns_tpsl)
    pf = profit_factor(returns_tpsl[traded_mask]) if traded_mask.any() else 0.0
    sharpe = np.mean(returns_tpsl) / (np.std(returns_tpsl) + 1e-8) * np.sqrt(252)

    actual_dir = (actual_returns > 0).astype(float)
    pred_dir = (dir_probs > 0.5).astype(float)
    da_raw = np.mean(pred_dir == actual_dir) * 100
    da_traded = np.mean(pred_dir[traded_mask] == actual_dir[traded_mask]) * 100 if traded_mask.any() else 0.0

    n_long = int((signals > 0).sum())
    n_short = int((signals < 0).sum())
    prec_long = np.mean(actual_dir[signals > 0] == 1) * 100 if n_long > 0 else 0.0
    prec_short = np.mean(actual_dir[signals < 0] == 0) * 100 if n_short > 0 else 0.0

    wins = returns_tpsl[traded_mask][returns_tpsl[traded_mask] > 0]
    losses = returns_tpsl[traded_mask][returns_tpsl[traded_mask] < 0]
    wr = len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0.0
    avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) * 100 if len(losses) > 0 else 0.0
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS ({BACKTEST_CONFIG['start_date']} - {BACKTEST_CONFIG['end_date']})")
    print(f"{'='*70}")
    print(f"  TP/SL Ratio  : {tp_sl_ratio}x (dynamic vol-scaled)")
    print(f"  No-Trade Zone: [{no_trade_low}, {no_trade_high}]")
    print(f"  Risk/Trade   : {risk_pct*100:.1f}%, Max Position: {max_pos_frac*100:.0f}%")
    print(f"  Samples      : {len(actual_returns)}")
    print(f"")
    print(f"  --- DIRECTIONAL ACCURACY ---")
    print(f"  DA (raw)       : {da_raw:.1f}%")
    print(f"  DA (traded)    : {da_traded:.1f}%")
    print(f"  LONG Precision : {prec_long:.1f}% ({n_long} trades)")
    print(f"  SHORT Precision: {prec_short:.1f}% ({n_short} trades)")
    print(f"")
    print(f"  --- FINANCIAL SUMMARY ---")
    print(f"  Deposit        : ${initial_equity:,.0f}")
    print(f"  Profit         : ${net_profit:,.0f} ({cum_tpsl[-1]*100:.1f}%)")
    print(f"  Total Balance  : ${final_equity:,.0f}")
    print(f"")
    print(f"  --- PERFORMANCE ---")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Max Drawdown   : {abs(md)*100:.1f}% (${abs(md)*initial_equity:,.0f})")
    print(f"  Win Rate       : {wr:.1f}%")
    print(f"  Avg Win / Loss : {avg_win:.3f}% / {avg_loss:.3f}% (W/L={wl_ratio:.2f}x)")
    print(f"  Trades         : {n_trades} / {len(traded_mask)} days ({trade_freq:.1f}%)")
    print(f"  Buy & Hold     : {cum_hold[-1]*100:.1f}% (${initial_equity*(1+cum_hold[-1]):,.0f})")
    print(f"{'='*70}")

    # Save CSV
    bt_dir = "backtest"
    os.makedirs(bt_dir, exist_ok=True)

    results = pd.DataFrame({
        "date": dates,
        "actual_return": actual_returns,
        "dir_prob": dir_probs,
        "signal": signals,
        "pos_fraction": pos_fractions,
        "return_tpsl": returns_tpsl,
        "cum_tpsl": cum_tpsl,
        "cum_hold": cum_hold,
        "equity": initial_equity * (1 + cum_tpsl),
    })
    csv_path = os.path.join(bt_dir, "backtest_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # --- CHARTS ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, initial_equity * (1 + cum_tpsl), label="TP/SL Strategy", color="green", linewidth=2.0)
    ax1.plot(dates, initial_equity * (1 + cum_hold), label="Buy & Hold", color="gray", linewidth=1.0, linestyle="--")
    ax1.axhline(initial_equity, color="black", linewidth=0.5, alpha=0.5)
    ax1.set_title(f"Equity Curve | Net: ${net_profit:,.0f} ({cum_tpsl[-1]*100:.1f}%) | PF={pf:.2f} | Sharpe={sharpe:.2f}",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 2. Cumulative DA
    ax2 = fig.add_subplot(gs[1, 0])
    correct = (pred_dir == actual_dir).astype(int)
    cum_da = np.cumsum(correct) / np.arange(1, len(correct) + 1) * 100
    ax2.plot(dates, cum_da, color="green", linewidth=1.5)
    ax2.axhline(50, color="gray", linestyle="--", linewidth=1.0)
    ax2.axhline(55, color="orange", linestyle=":", linewidth=0.8, label="55%")
    ax2.set_title("Cumulative Directional Accuracy", fontsize=12, fontweight="bold")
    ax2.set_ylabel("DA %")
    ax2.set_ylim(40, 65)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. Probability Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    up_mask = actual_returns > 0
    ax3.hist(dir_probs[up_mask], bins=50, alpha=0.6, label="Actual UP", color="green", density=True)
    ax3.hist(dir_probs[~up_mask], bins=50, alpha=0.6, label="Actual DOWN", color="red", density=True)
    ax3.axvline(no_trade_high, color="green", linestyle="-", linewidth=1.5, label=f"BUY >{no_trade_high}")
    ax3.axvline(no_trade_low, color="red", linestyle="-", linewidth=1.5, label=f"SELL <{no_trade_low}")
    ax3.axvspan(no_trade_low, no_trade_high, alpha=0.15, color="gray", label="No-Trade")
    ax3.set_title(f"P(UP) Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("P(UP)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Feature Importance
    ax4 = fig.add_subplot(gs[2, 0])
    if importance is not None and len(importance) == len(FEATURE_COLUMNS):
        sorted_idx = np.argsort(importance)
        sorted_features = [FEATURE_COLUMNS[i] for i in sorted_idx]
        sorted_imp = importance[sorted_idx] * 100
        colors_fi = ["green" if v > 0.1 else ("orange" if v > 0 else "red") for v in sorted_imp]
        ax4.barh(sorted_features, sorted_imp, color=colors_fi)
        ax4.axvline(0, color="black", linewidth=0.5)
        ax4.set_title("Feature Importance", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Importance (%)")
    ax4.grid(True, alpha=0.3)

    # 5. Drawdown
    ax5 = fig.add_subplot(gs[2, 1])
    equity = initial_equity * (1 + cum_tpsl)
    running_max = np.maximum.accumulate(equity)
    drawdown_pct = (equity - running_max) / running_max * 100
    ax5.fill_between(dates, drawdown_pct, 0, color="red", alpha=0.4)
    ax5.set_title(f"Drawdown (Max: {abs(md)*100:.1f}%)", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Drawdown %")
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # 6. Rolling Trade Stats
    ax6 = fig.add_subplot(gs[3, 0])
    window = 30
    rolling_wr = pd.Series(wins.tolist() + [0]*(len(returns_tpsl)-len(wins)),
                            index=dates).rolling(window).mean().values * 100 if len(wins) > 0 else np.zeros(len(dates))
    if len(wins) > 0:
        trade_rets = pd.Series(returns_tpsl[traded_mask], index=dates[traded_mask])
        rolling_wr = trade_rets.rolling(window).apply(lambda x: (x > 0).mean() * 100).values
        ax6.plot(trade_rets.index, rolling_wr, color="green", linewidth=1.0)
        ax6.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax6.set_title(f"Rolling Win Rate ({window}-day)", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Win Rate %")
    ax6.grid(True, alpha=0.3)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    # 7. Per-Direction + Summary
    ax7 = fig.add_subplot(gs[3, 1])
    cats = ["LONG\nPrec", "SHORT\nPrec", "DA\nTraded", "WR%", "W/L\nRatio"]
    vals = [prec_long, prec_short, da_traded, wr, wl_ratio]
    colors_bar = ["green", "red", "steelblue", "purple", "orange"]
    bars = ax7.bar(cats, vals, color=colors_bar, alpha=0.7, edgecolor="black")
    for bar, val in zip(bars, vals):
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax7.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax7.set_title("Strategy Metrics", fontsize=12, fontweight="bold")
    ax7.set_ylabel("%")
    ax7.set_ylim(0, max(vals) + 10)
    ax7.grid(True, alpha=0.3, axis="y")

    chart_path = os.path.join(bt_dir, "backtest_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {chart_path}")

    return {
        "net_profit": net_profit,
        "final_equity": final_equity,
        "pf": pf,
        "sharpe": sharpe,
        "max_dd": abs(md),
        "win_rate": wr,
        "wl_ratio": wl_ratio,
        "n_trades": n_trades,
        "da_traded": da_traded,
    }


if __name__ == "__main__":
    run_backtest()