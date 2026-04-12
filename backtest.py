import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_CONFIG, BACKTEST_CONFIG, DEVICE,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, FEATURE_WEIGHTS,
)
from data_loader import load_dataset, load_raw_prices, load_labels, build_dataset
from model import ForexClassifier


def compute_atr_filter(prices_df, lookback_days=60):
    eurusd = prices_df["EUR_USD"]
    daily_range = eurusd.diff().abs()
    atr = daily_range.rolling(DATA_CONFIG["atr_period"]).mean()
    atr_threshold = atr.rolling(lookback_days).quantile(DATA_CONFIG["atr_filter_quantile"])
    return atr, atr_threshold


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
    threshold = DATA_CONFIG["trade_threshold"]

    # Feature importance
    print("\nComputing feature importance...")
    model.eval()
    n_imp = min(500, len(scaled))

    X_base_seq = np.array([scaled[i:i+lookback] for i in range(n_imp)])
    X_base = torch.tensor(X_base_seq, dtype=torch.float32).to(DEVICE)
    y_base_np = ((labels_df["label"].values[:n_imp] + 1) / 2.0)

    with torch.no_grad():
        base_probs = model.predict_proba(X_base).cpu().numpy()
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
                perm_probs = model.predict_proba(X_perm_t).cpu().numpy()
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

    mask = (feats_df.index >= BACKTEST_CONFIG["start_date"]) & (feats_df.index < BACKTEST_CONFIG["end_date"])
    bt_indices = np.where(mask)[0]
    bt_indices = bt_indices[bt_indices >= lookback]
    bt_indices = bt_indices[bt_indices < len(eurusd) - 1]

    if len(bt_indices) == 0:
        print("No backtest data available.")
        return

    atr_series, atr_threshold_series = compute_atr_filter(prices_df)

    dates_list = []
    actual_returns = []
    dir_probs = []
    clf_signals_basic = []
    clf_signals_filtered = []
    atr_active_list = []

    with torch.no_grad():
        for idx in bt_indices:
            seq = scaled[idx - lookback:idx]
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prob = model.predict_proba(X).cpu().item()

            current_price = eurusd[idx - 1]
            next_price = eurusd[idx]
            actual_ret = np.log(next_price / current_price) if current_price > 0 else 0.0

            idx_date = feats_df.index[idx]
            atr_val = atr_series.loc[idx_date] if idx_date in atr_series.index else np.nan
            atr_thr = atr_threshold_series.loc[idx_date] if idx_date in atr_threshold_series.index else np.nan

            atr_ok = not np.isnan(atr_val) and not np.isnan(atr_thr) and atr_val > atr_thr

            signal_basic = 1.0 if prob > 0.5 else -1.0
            signal_filtered = 0.0
            if atr_ok:
                if prob > threshold:
                    signal_filtered = 1.0
                elif prob < (1 - threshold):
                    signal_filtered = -1.0

            dates_list.append(idx_date)
            actual_returns.append(actual_ret)
            dir_probs.append(prob)
            clf_signals_basic.append(signal_basic)
            clf_signals_filtered.append(signal_filtered)
            atr_active_list.append(atr_ok)

    actual_returns = np.array(actual_returns)
    dir_probs = np.array(dir_probs)
    clf_signals_basic = np.array(clf_signals_basic)
    clf_signals_filtered = np.array(clf_signals_filtered)
    atr_active = np.array(atr_active_list)
    dates = pd.to_datetime(dates_list)

    actual_direction = (actual_returns > 0).astype(float)
    pred_direction = (dir_probs > 0.5).astype(float)

    da_basic = np.mean(pred_direction == actual_direction) * 100
    da_flipped = np.mean(pred_direction != actual_direction) * 100

    clf_signals_basic = np.where(dir_probs > 0.5, 1.0, -1.0)

    clf_signals_filtered = np.zeros(len(dir_probs))
    for i in range(len(dir_probs)):
        p = dir_probs[i]
        if atr_active[i]:
            if p > threshold:
                clf_signals_filtered[i] = 1.0
            elif p < (1 - threshold):
                clf_signals_filtered[i] = -1.0

    da_effective = np.mean(pred_direction == actual_direction) * 100

    if atr_active.sum() > 0:
        da_filtered = np.mean(pred_direction[atr_active] == actual_direction[atr_active]) * 100
    else:
        da_filtered = 0.0

    strategy_basic = clf_signals_basic * actual_returns
    strategy_atr = clf_signals_filtered * actual_returns
    cum_basic = np.cumprod(1 + strategy_basic) - 1
    cum_atr = np.cumprod(1 + strategy_atr) - 1
    cum_hold = np.cumprod(1 + actual_returns) - 1

    def max_drawdown(returns):
        curve = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(curve)
        dd = (curve - running_max) / running_max
        return dd.min()

    md_basic = max_drawdown(strategy_basic)
    md_atr = max_drawdown(strategy_atr) if (clf_signals_filtered != 0).any() else 0.0
    md_hold = max_drawdown(actual_returns)

    sharpe_basic = np.mean(strategy_basic) / (np.std(strategy_basic) + 1e-8) * np.sqrt(252)
    sharpe_atr = np.mean(strategy_atr) / (np.std(strategy_atr) + 1e-8) * np.sqrt(252) if (clf_signals_filtered != 0).any() else 0.0

    n_atr_trades = (clf_signals_filtered != 0).sum()
    pct_active = atr_active.sum() / len(atr_active) * 100

    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS ({BACKTEST_CONFIG['start_date']} - {BACKTEST_CONFIG['end_date']})")
    print(f"{'='*70}")
    print(f"Samples                    : {len(actual_returns)}")
    print(f"")
    print(f"DA Raw (>0.5)             : {da_basic:.2f}%")
    print(f"DA Effective (auto-flip)  : {da_effective:.2f}%")
    print(f"DA ATR-filtered          : {da_filtered:.2f}%")
    print(f"ATR active days           : {pct_active:.1f}%")
    print(f"")
    print(f"Strategy Basic Return       : {cum_basic[-1]*100:.2f}%")
    print(f"Strategy ATR Return         : {cum_atr[-1]*100:.2f}%")
    print(f"Buy & Hold Return           : {cum_hold[-1]*100:.2f}%")
    print(f"")
    print(f"Max Drawdown (Basic)        : {md_basic*100:.2f}%")
    print(f"Max Drawdown (ATR)          : {md_atr*100:.2f}%")
    print(f"Max Drawdown (Hold)         : {md_hold*100:.2f}%")
    print(f"")
    print(f"Sharpe (Basic)              : {sharpe_basic:.2f}")
    print(f"Sharpe (ATR)                : {sharpe_atr:.2f}")
    print(f"ATR trades                   : {n_atr_trades}")
    print(f"")
    print(f"Prob range                  : [{np.min(dir_probs):.3f}, {np.max(dir_probs):.3f}]")
    print(f"Prob std                    : {np.std(dir_probs):.4f}")

    results = pd.DataFrame({
        "date": dates,
        "actual_return": actual_returns,
        "dir_prob": dir_probs,
        "clf_signal_basic": clf_signals_basic,
        "clf_signal_atr": clf_signals_filtered,
        "atr_active": atr_active.astype(float),
        "strategy_return_basic": strategy_basic,
        "strategy_return_atr": strategy_atr,
        "cum_basic": cum_basic,
        "cum_atr": cum_atr,
        "cum_hold": cum_hold,
    })
    results.to_csv("backtest_results.csv", index=False)
    print("Results saved to backtest_results.csv")

    plot_backtest(dates, actual_returns, dir_probs, clf_signals_basic, clf_signals_filtered,
                  cum_basic, cum_atr, cum_hold, atr_active, importance,
                  da_basic=da_basic, da_effective=da_effective, da_filtered=da_filtered,
                  pct_active=pct_active, md_basic=md_basic, md_atr=md_atr, md_hold=md_hold,
                  sharpe_basic=sharpe_basic, sharpe_atr=sharpe_atr, n_atr_trades=n_atr_trades,
                  cum_basic_last=cum_basic[-1], cum_atr_last=cum_atr[-1], cum_hold_last=cum_hold[-1])


def plot_backtest(dates, actual_returns, dir_probs, clf_signals_basic, clf_signals_filtered,
                  cum_basic, cum_atr, cum_hold, atr_active, importance,
                  da_basic=0, da_effective=0, da_filtered=0, pct_active=0,
                  md_basic=0, md_atr=0, md_hold=0,
                  sharpe_basic=0, sharpe_atr=0, n_atr_trades=0,
                  cum_basic_last=0, cum_atr_last=0, cum_hold_last=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig = plt.figure(figsize=(24, 30))
    gs = fig.add_gridspec(5, 2, hspace=0.45, wspace=0.3)

    flip_note = ""
    actual_direction = (actual_returns > 0).astype(float)
    pred_direction = (dir_probs > 0.5).astype(float)

    # 1. Equity Curves
    ax = fig.add_subplot(gs[0, :])
    ax.plot(dates, cum_basic * 100, label="Basic Strategy", color="blue", linewidth=1.0, alpha=0.7)
    ax.plot(dates, cum_atr * 100, label="ATR Filter Strategy", color="green", linewidth=2.0)
    ax.plot(dates, cum_hold * 100, label="Buy & Hold", color="gray", linewidth=1.0, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Equity Curves{flip_note}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 2. Cumulative DA
    ax = fig.add_subplot(gs[1, 0])
    correct = (pred_direction == actual_direction).astype(int)
    cumulative_da = np.cumsum(correct) / np.arange(1, len(correct) + 1) * 100
    ax.plot(dates, cumulative_da, color="green", linewidth=1.5, label="Classifier DA")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.0, label="50%")
    ax.axhline(52, color="orange", linestyle=":", linewidth=0.8, label="52%")
    ax.axhline(55, color="red", linestyle=":", linewidth=0.8, label="55%")
    ax.set_ylim(40, 65)
    ax.set_title("Cumulative Directional Accuracy")
    ax.set_ylabel("DA %")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 3. Probability Distribution
    ax = fig.add_subplot(gs[1, 1])
    up_mask = actual_returns > 0
    ax.hist(dir_probs[up_mask], bins=50, alpha=0.6, label="Actual UP", color="green", density=True)
    ax.hist(dir_probs[~up_mask], bins=50, alpha=0.6, label="Actual DOWN", color="red", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="0.5")
    ax.axvline(DATA_CONFIG["trade_threshold"], color="orange", linestyle=":", linewidth=1.0,
               label=f"Threshold={DATA_CONFIG['trade_threshold']}")
    ax.axvline(1 - DATA_CONFIG["trade_threshold"], color="orange", linestyle=":", linewidth=1.0)
    ax.set_title(f"Probability Distribution (T={DATA_CONFIG['temperature']})")
    ax.set_xlabel("P(UP)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Rolling DA
    ax = fig.add_subplot(gs[2, 0])
    window = 30
    rolling_da = pd.Series(correct).rolling(window).mean() * 100
    ax.plot(dates, rolling_da, color="green", linewidth=1.0)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(52, color="orange", linestyle=":", linewidth=0.8, label="52%")
    ax.axhline(55, color="red", linestyle=":", linewidth=0.8, label="55%")
    ax.set_title(f"Rolling DA ({window}-day)")
    ax.set_ylabel("DA %")
    ax.set_ylim(30, 70)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 5. DA vs Confidence
    ax = fig.add_subplot(gs[2, 1])
    thresholds = np.arange(0.5, 0.7, 0.01)
    da_by_conf = []
    pct_by_conf = []
    for t in thresholds:
        mask = (dir_probs > t) | (dir_probs < (1 - t))
        if mask.sum() > 0:
            da_t = np.mean(pred_direction[mask] == actual_direction[mask]) * 100
            da_by_conf.append(da_t)
            pct_by_conf.append(mask.sum() / len(mask) * 100)
        else:
            da_by_conf.append(0)
            pct_by_conf.append(0)

    ax2 = ax.twinx()
    ax.bar(thresholds, da_by_conf, width=0.008, color="steelblue", alpha=0.7, label="DA %")
    ax2.plot(thresholds, pct_by_conf, "ro-", linewidth=1.5, label="% of trades")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("DA %")
    ax2.set_ylabel("% of Trades")
    ax.set_title("DA vs Confidence Threshold")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 6. Feature Importance
    ax = fig.add_subplot(gs[3, 0])
    if importance is not None and len(importance) == len(FEATURE_COLUMNS):
        sorted_idx = np.argsort(importance)
        sorted_features = [FEATURE_COLUMNS[i] for i in sorted_idx]
        sorted_imp = importance[sorted_idx] * 100
        colors = ["green" if v > 0.1 else ("orange" if v > 0 else "red") for v in sorted_imp]
        ax.barh(sorted_features, sorted_imp, color=colors)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title("Feature Importance\n(accuracy % drop when permuted)")
        ax.set_xlabel("Importance (%)")
    else:
        ax.text(0.5, 0.5, "Not available", ha="center", va="center")
    ax.grid(True, alpha=0.3)

    # 7. Prob Range over Time
    ax = fig.add_subplot(gs[3, 1])
    rolling_std = pd.Series(dir_probs).rolling(30).std()
    rolling_mean = pd.Series(dir_probs).rolling(30).mean()
    ax.plot(dates, rolling_mean, color="blue", linewidth=1.0, label="Mean P(UP)")
    ax.fill_between(dates, rolling_mean - rolling_std, rolling_mean + rolling_std,
                    color="blue", alpha=0.2, label="1 STD")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Rolling Probability Spread (30-day)")
    ax.set_ylabel("P(UP)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    from sklearn.metrics import classification_report, confusion_matrix

    actual_direction = (actual_returns > 0).astype(float)
    pred_direction = (dir_probs > 0.5).astype(float)
    y_true = actual_direction.astype(int)
    y_pred = pred_direction.astype(int)

    clf_report = classification_report(y_true, y_pred, target_names=["DOWN", "UP"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # 8. Statistics Table
    ax_stats = fig.add_subplot(gs[4, 0])
    ax_stats.axis("off")
    stats_data = [
        ["Metric", "Value"],
        ["DA Basic", f"{da_basic:.2f}%"],
        ["DA Effective", f"{da_effective:.2f}%"],
        ["DA ATR-filtered", f"{da_filtered:.2f}%"],
        ["ATR Active Days", f"{pct_active:.1f}%"],
        ["ATR Trades", f"{n_atr_trades}"],
        ["Return (Basic)", f"{cum_basic_last*100:.2f}%"],
        ["Return (ATR)", f"{cum_atr_last*100:.2f}%"],
        ["Return (Hold)", f"{cum_hold_last*100:.2f}%"],
        ["Max DD (Basic)", f"{md_basic*100:.2f}%"],
        ["Max DD (ATR)", f"{md_atr*100:.2f}%"],
        ["Max DD (Hold)", f"{md_hold*100:.2f}%"],
        ["Sharpe (Basic)", f"{sharpe_basic:.2f}"],
        ["Sharpe (ATR)", f"{sharpe_atr:.2f}"],
    ]
    tbl = ax_stats.table(cellText=stats_data[1:], colLabels=stats_data[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    for key, cell in tbl.get_celld().items():
        if key[1] == 0:
            cell.set_text_props(fontweight="bold")
    ax_stats.set_title("Backtest Statistics", fontsize=13, fontweight="bold")

    # 9. Classification Report + Confusion Matrix
    ax_cm = fig.add_subplot(gs[4, 1])
    ax_cm.axis("off")

    ax_cm.text(0.02, 0.98, "Classification Report", fontsize=11, fontweight="bold",
               transform=ax_cm.transAxes, va="top")
    cr_lines = classification_report(y_true, y_pred, target_names=["DOWN", "UP"]).split("\n")
    cr_text = "\n".join(cr_lines)
    ax_cm.text(0.02, 0.90, cr_text, fontsize=8.5, fontfamily="monospace",
               transform=ax_cm.transAxes, va="top")

    ax_cm.text(0.60, 0.98, "Confusion Matrix", fontsize=11, fontweight="bold",
               transform=ax_cm.transAxes, va="top")

    cm_ax = fig.add_axes([0.68, 0.04, 0.25, 0.12])
    cm_ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, aspect="auto")
    cm_ax.set_xticks([0, 1])
    cm_ax.set_yticks([0, 1])
    cm_ax.set_xticklabels(["DOWN", "UP"])
    cm_ax.set_yticklabels(["DOWN", "UP"])
    cm_ax.set_xlabel("Predicted", fontsize=9)
    cm_ax.set_ylabel("Actual", fontsize=9)
    for i in range(2):
        for j in range(2):
            cm_ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)

    plt.savefig("backtest_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved to backtest_chart.png")


if __name__ == "__main__":
    run_backtest()