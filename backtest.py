import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_CONFIG, BACKTEST_CONFIG, DEVICE, RISK_CONFIG,
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


def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss < 1e-10:
        return float("inf")
    return gross_profit / gross_loss


def max_drawdown(returns):
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
    threshold = DATA_CONFIG["trade_threshold"]
    threshold_down = DATA_CONFIG["trade_threshold_down"]
    threshold_up = DATA_CONFIG["trade_threshold_up"]
    risk_pct = RISK_CONFIG["risk_per_trade"]
    max_lot = RISK_CONFIG["max_lot_multiplier"]
    min_lot = RISK_CONFIG["min_lot_multiplier"]
    initial_equity = RISK_CONFIG["initial_equity"]

    
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
    clf_signals_atr = []
    clf_signals_asymm = []
    lot_multipliers = []
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

            
            signal_atr = 0.0
            if atr_ok:
                if prob > threshold:
                    signal_atr = 1.0
                elif prob < (1 - threshold):
                    signal_atr = -1.0

            
            signal_asymm = 0.0
            if atr_ok:
                if prob > threshold_up:
                    signal_asymm = 1.0
                elif prob < (1 - threshold_down):
                    signal_asymm = -1.0

            
            sl_pct = atr_val / current_price if (not np.isnan(atr_val) and atr_val > 0 and current_price > 0) else 0.01
            lot_mult = risk_pct / sl_pct if sl_pct > 0 else 1.0
            lot_mult = float(np.clip(lot_mult, min_lot, max_lot))

            dates_list.append(idx_date)
            actual_returns.append(actual_ret)
            dir_probs.append(prob)
            clf_signals_basic.append(signal_basic)
            clf_signals_atr.append(signal_atr)
            clf_signals_asymm.append(signal_asymm)
            lot_multipliers.append(lot_mult)
            atr_active_list.append(atr_ok)

    actual_returns = np.array(actual_returns)
    dir_probs = np.array(dir_probs)
    clf_signals_basic = np.array(clf_signals_basic)
    clf_signals_atr = np.array(clf_signals_atr)
    clf_signals_asymm = np.array(clf_signals_asymm)
    lot_multipliers = np.array(lot_multipliers)
    atr_active = np.array(atr_active_list)
    dates = pd.to_datetime(dates_list)

    actual_direction = (actual_returns > 0).astype(float)
    pred_direction = (dir_probs > 0.5).astype(float)

    
    da_basic = np.mean(pred_direction == actual_direction) * 100

    
    traded_mask = clf_signals_asymm != 0
    da_asymm = np.mean(pred_direction[traded_mask] == actual_direction[traded_mask]) * 100 if traded_mask.sum() > 0 else 0.0

    
    up_trades = clf_signals_asymm == 1.0
    down_trades = clf_signals_asymm == -1.0
    n_up = up_trades.sum()
    n_down = down_trades.sum()

    precision_up = np.mean(actual_direction[up_trades] == 1) * 100 if n_up > 0 else 0.0
    precision_down = np.mean(actual_direction[down_trades] == 0) * 100 if n_down > 0 else 0.0

    
    if atr_active.sum() > 0:
        da_atr = np.mean(pred_direction[atr_active] == actual_direction[atr_active]) * 100
    else:
        da_atr = 0.0

    
    strategy_basic = clf_signals_basic * actual_returns
    strategy_atr = clf_signals_atr * actual_returns
    strategy_asymm = clf_signals_asymm * actual_returns
    strategy_dynamic = clf_signals_asymm * lot_multipliers * actual_returns

    cum_basic = np.cumprod(1 + strategy_basic) - 1
    cum_atr = np.cumprod(1 + strategy_atr) - 1
    cum_asymm = np.cumprod(1 + strategy_asymm) - 1
    cum_dynamic = np.cumprod(1 + strategy_dynamic) - 1
    cum_hold = np.cumprod(1 + actual_returns) - 1

    
    md_basic = max_drawdown(strategy_basic)
    md_atr = max_drawdown(strategy_atr) if (clf_signals_atr != 0).any() else 0.0
    md_asymm = max_drawdown(strategy_asymm) if traded_mask.any() else 0.0
    md_dynamic = max_drawdown(strategy_dynamic) if traded_mask.any() else 0.0
    md_hold = max_drawdown(actual_returns)

    sharpe_basic = np.mean(strategy_basic) / (np.std(strategy_basic) + 1e-8) * np.sqrt(252)
    sharpe_atr = np.mean(strategy_atr) / (np.std(strategy_atr) + 1e-8) * np.sqrt(252) if (clf_signals_atr != 0).any() else 0.0
    sharpe_asymm = np.mean(strategy_asymm) / (np.std(strategy_asymm) + 1e-8) * np.sqrt(252) if traded_mask.any() else 0.0
    sharpe_dynamic = np.mean(strategy_dynamic) / (np.std(strategy_dynamic) + 1e-8) * np.sqrt(252) if traded_mask.any() else 0.0

    pf_basic = profit_factor(strategy_basic)
    pf_atr = profit_factor(strategy_atr) if (clf_signals_atr != 0).any() else 0.0
    pf_asymm = profit_factor(strategy_asymm) if traded_mask.any() else 0.0
    pf_dynamic = profit_factor(strategy_dynamic) if traded_mask.any() else 0.0

    n_atr_trades = int((clf_signals_atr != 0).sum())
    n_asymm_trades = int(traded_mask.sum())
    n_up_trades = int(n_up)
    n_down_trades = int(n_down)
    pct_active = atr_active.sum() / len(atr_active) * 100

    
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS ({BACKTEST_CONFIG['start_date']} - {BACKTEST_CONFIG['end_date']})")
    print(f"{'='*70}")
    print(f"Samples                    : {len(actual_returns)}")
    print(f"")
    print(f"--- Directional Accuracy ---")
    print(f"DA Raw (>0.5)             : {da_basic:.2f}%")
    print(f"DA ATR-filtered (sym)     : {da_atr:.2f}%")
    print(f"DA Asymmetric              : {da_asymm:.2f}%")
    print(f"")
    print(f"--- Per-Direction (Asymmetric) ---")
    print(f"UP Precision               : {precision_up:.2f}%  ({n_up_trades} trades)")
    print(f"DOWN Precision             : {precision_down:.2f}%  ({n_down_trades} trades)")
    print(f"")
    print(f"--- Returns ---")
    print(f"Basic Return               : {cum_basic[-1]*100:.2f}%")
    print(f"ATR Return                 : {cum_atr[-1]*100:.2f}%")
    print(f"Asymmetric Return          : {cum_asymm[-1]*100:.2f}%")
    print(f"Dynamic Return             : {cum_dynamic[-1]*100:.2f}%")
    print(f"Buy & Hold Return          : {cum_hold[-1]*100:.2f}%")
    print(f"")
    print(f"--- Max Drawdown ---")
    print(f"Max DD (Basic)             : {md_basic*100:.2f}%")
    print(f"Max DD (ATR)               : {md_atr*100:.2f}%")
    print(f"Max DD (Asymmetric)        : {md_asymm*100:.2f}%")
    print(f"Max DD (Dynamic)           : {md_dynamic*100:.2f}%")
    print(f"Max DD (Hold)              : {md_hold*100:.2f}%")
    print(f"")
    print(f"--- Sharpe Ratio ---")
    print(f"Sharpe (Basic)             : {sharpe_basic:.2f}")
    print(f"Sharpe (ATR)               : {sharpe_atr:.2f}")
    print(f"Sharpe (Asymmetric)        : {sharpe_asymm:.2f}")
    print(f"Sharpe (Dynamic)           : {sharpe_dynamic:.2f}")
    print(f"")
    print(f"--- Profit Factor ---")
    print(f"PF (Basic)                 : {pf_basic:.2f}")
    print(f"PF (ATR)                   : {pf_atr:.2f}")
    print(f"PF (Asymmetric)            : {pf_asymm:.2f}")
    print(f"PF (Dynamic)               : {pf_dynamic:.2f}")
    print(f"")
    print(f"--- Trade Counts ---")
    print(f"ATR trades (symmetric)     : {n_atr_trades}")
    print(f"Asymmetric trades          : {n_asymm_trades}")
    print(f"  UP trades                : {n_up_trades}")
    print(f"  DOWN trades              : {n_down_trades}")
    print(f"ATR active days            : {pct_active:.1f}%")
    print(f"")
    print(f"--- Lot Multiplier ---")
    print(f"Mean lot mult (ATR active) : {lot_multipliers[atr_active].mean():.3f}" if atr_active.sum() > 0 else "  N/A")
    print(f"Min lot mult                : {lot_multipliers.min():.3f}")
    print(f"Max lot mult                : {lot_multipliers.max():.3f}")
    print(f"Std lot mult                : {lot_multipliers.std():.3f}")
    print(f"")
    print(f"--- Signal Thresholds ---")
    print(f"UP threshold               : {threshold_up}")
    print(f"DOWN threshold             : {threshold_down}")
    print(f"ATR filter quantile        : {DATA_CONFIG['atr_filter_quantile']}")
    print(f"")
    print(f"Prob range                  : [{np.min(dir_probs):.3f}, {np.max(dir_probs):.3f}]")
    print(f"Prob std                    : {np.std(dir_probs):.4f}")

    
    results = pd.DataFrame({
        "date": dates,
        "actual_return": actual_returns,
        "dir_prob": dir_probs,
        "signal_basic": clf_signals_basic,
        "signal_atr": clf_signals_atr,
        "signal_asymm": clf_signals_asymm,
        "lot_multiplier": lot_multipliers,
        "atr_active": atr_active.astype(float),
        "strategy_basic": strategy_basic,
        "strategy_atr": strategy_atr,
        "strategy_asymm": strategy_asymm,
        "strategy_dynamic": strategy_dynamic,
        "cum_basic": cum_basic,
        "cum_atr": cum_atr,
        "cum_asymm": cum_asymm,
        "cum_dynamic": cum_dynamic,
        "cum_hold": cum_hold,
    })
    results.to_csv("backtest_results.csv", index=False)
    print("Results saved to backtest_results.csv")

    plot_backtest(
        dates, actual_returns, dir_probs,
        clf_signals_basic, clf_signals_atr, clf_signals_asymm,
        cum_basic, cum_atr, cum_asymm, cum_dynamic, cum_hold,
        atr_active, lot_multipliers, importance,
        da_basic=da_basic, da_asymm=da_asymm, da_atr=da_atr,
        precision_up=precision_up, precision_down=precision_down,
        n_up_trades=n_up_trades, n_down_trades=n_down_trades,
        pct_active=pct_active, n_atr_trades=n_atr_trades, n_asymm_trades=n_asymm_trades,
        md_basic=md_basic, md_atr=md_atr, md_asymm=md_asymm,
        md_dynamic=md_dynamic, md_hold=md_hold,
        sharpe_basic=sharpe_basic, sharpe_atr=sharpe_atr,
        sharpe_asymm=sharpe_asymm, sharpe_dynamic=sharpe_dynamic,
        pf_basic=pf_basic, pf_atr=pf_atr, pf_asymm=pf_asymm, pf_dynamic=pf_dynamic,
        cum_basic_last=cum_basic[-1], cum_atr_last=cum_atr[-1],
        cum_asymm_last=cum_asymm[-1], cum_dynamic_last=cum_dynamic[-1],
        cum_hold_last=cum_hold[-1],
    )


def plot_backtest(
    dates, actual_returns, dir_probs,
    clf_signals_basic, clf_signals_atr, clf_signals_asymm,
    cum_basic, cum_atr, cum_asymm, cum_dynamic, cum_hold,
    atr_active, lot_multipliers, importance,
    da_basic=0, da_asymm=0, da_atr=0,
    precision_up=0, precision_down=0,
    n_up_trades=0, n_down_trades=0,
    pct_active=0, n_atr_trades=0, n_asymm_trades=0,
    md_basic=0, md_atr=0, md_asymm=0,
    md_dynamic=0, md_hold=0,
    sharpe_basic=0, sharpe_atr=0,
    sharpe_asymm=0, sharpe_dynamic=0,
    pf_basic=0, pf_atr=0, pf_asymm=0, pf_dynamic=0,
    cum_basic_last=0, cum_atr_last=0,
    cum_asymm_last=0, cum_dynamic_last=0,
    cum_hold_last=0,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from sklearn.metrics import classification_report, confusion_matrix

    threshold_up = DATA_CONFIG["trade_threshold_up"]
    threshold_down = DATA_CONFIG["trade_threshold_down"]

    fig = plt.figure(figsize=(24, 36))
    gs = fig.add_gridspec(6, 2, hspace=0.45, wspace=0.3)

    actual_direction = (actual_returns > 0).astype(float)
    pred_direction = (dir_probs > 0.5).astype(float)

    
    ax = fig.add_subplot(gs[0, :])
    ax.plot(dates, cum_basic * 100, label="Basic (0.5)", color="gray", linewidth=0.8, alpha=0.5)
    ax.plot(dates, cum_atr * 100, label="ATR Symmetric", color="blue", linewidth=1.0, alpha=0.7)
    ax.plot(dates, cum_asymm * 100, label=f"Asymmetric (UP>{threshold_up}, DN<{1-threshold_down})", color="orange", linewidth=1.5)
    ax.plot(dates, cum_dynamic * 100, label="Dynamic Sizing", color="green", linewidth=2.0)
    ax.plot(dates, cum_hold * 100, label="Buy & Hold", color="black", linewidth=0.8, alpha=0.4, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Equity Curves", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    
    ax = fig.add_subplot(gs[1, 0])
    correct = (pred_direction == actual_direction).astype(int)
    cumulative_da = np.cumsum(correct) / np.arange(1, len(correct) + 1) * 100
    ax.plot(dates, cumulative_da, color="green", linewidth=1.5, label="Classifier DA")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.0, label="50%")
    ax.axhline(55, color="orange", linestyle=":", linewidth=0.8, label="55%")
    ax.axhline(60, color="red", linestyle=":", linewidth=0.8, label="60%")
    ax.set_ylim(40, 65)
    ax.set_title("Cumulative Directional Accuracy")
    ax.set_ylabel("DA %")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    
    ax = fig.add_subplot(gs[1, 1])
    up_mask = actual_returns > 0
    ax.hist(dir_probs[up_mask], bins=50, alpha=0.6, label="Actual UP", color="green", density=True)
    ax.hist(dir_probs[~up_mask], bins=50, alpha=0.6, label="Actual DOWN", color="red", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="0.5")
    ax.axvline(threshold_up, color="green", linestyle="-", linewidth=1.5,
               label=f"UP threshold={threshold_up}")
    ax.axvline(1 - threshold_down, color="red", linestyle="-", linewidth=1.5,
               label=f"DOWN threshold={1-threshold_down:.3f}")
    ax.axvline(threshold_down, color="orange", linestyle=":", linewidth=0.8)
    ax.set_title(f"Probability Distribution (Asymmetric T: UP>{threshold_up}, DN<{1-threshold_down:.3f})")
    ax.set_xlabel("P(UP)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    
    ax = fig.add_subplot(gs[2, 0])
    window = 30
    rolling_da = pd.Series(correct).rolling(window).mean() * 100
    ax.plot(dates, rolling_da, color="green", linewidth=1.0)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(55, color="orange", linestyle=":", linewidth=0.8, label="55%")
    ax.axhline(60, color="red", linestyle=":", linewidth=0.8, label="60%")
    ax.set_title(f"Rolling DA ({window}-day)")
    ax.set_ylabel("DA %")
    ax.set_ylim(30, 70)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    
    ax = fig.add_subplot(gs[2, 1])
    categories = ["UP\nPrecision", "DOWN\nPrecision", "Overall\nAsymm DA"]
    values = [precision_up, precision_down, da_asymm]
    colors_bar = ["green", "red", "steelblue"]
    bars = ax.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50%")
    ax.axhline(60, color="red", linestyle="--", linewidth=0.8, label="60%")
    ax.set_title("Per-Direction Precision (Asymmetric Strategy)")
    ax.set_ylabel("Precision %")
    ax.set_ylim(0, 80)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    
    ax = fig.add_subplot(gs[3, 0])
    if importance is not None and len(importance) == len(FEATURE_COLUMNS):
        sorted_idx = np.argsort(importance)
        sorted_features = [FEATURE_COLUMNS[i] for i in sorted_idx]
        sorted_imp = importance[sorted_idx] * 100
        colors_fi = ["green" if v > 0.1 else ("orange" if v > 0 else "red") for v in sorted_imp]
        ax.barh(sorted_features, sorted_imp, color=colors_fi)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title("Feature Importance\n(accuracy % drop when permuted)")
        ax.set_xlabel("Importance (%)")
    else:
        ax.text(0.5, 0.5, "Not available", ha="center", va="center")
    ax.grid(True, alpha=0.3)

    
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

    
    ax_stats = fig.add_subplot(gs[4, 0])
    ax_stats.axis("off")
    stats_data = [
        ["Metric", "Basic", "ATR Sym", "Asymm", "Dynamic"],
        ["DA (%)", f"{da_basic:.2f}", f"{da_atr:.2f}", f"{da_asymm:.2f}", f"{da_asymm:.2f}"],
        ["UP Prec (%)", "-", "-", f"{precision_up:.2f}", f"{precision_up:.2f}"],
        ["DN Prec (%)", "-", "-", f"{precision_down:.2f}", f"{precision_down:.2f}"],
        ["Return (%)", f"{cum_basic_last*100:.2f}", f"{cum_atr_last*100:.2f}",
         f"{cum_asymm_last*100:.2f}", f"{cum_dynamic_last*100:.2f}"],
        ["Max DD (%)", f"{md_basic*100:.2f}", f"{md_atr*100:.2f}",
         f"{md_asymm*100:.2f}", f"{md_dynamic*100:.2f}"],
        ["Sharpe", f"{sharpe_basic:.2f}", f"{sharpe_atr:.2f}",
         f"{sharpe_asymm:.2f}", f"{sharpe_dynamic:.2f}"],
        ["PF", f"{pf_basic:.2f}", f"{pf_atr:.2f}", f"{pf_asymm:.2f}", f"{pf_dynamic:.2f}"],
        ["Trades", f"{len(actual_returns)}", f"{n_atr_trades}", f"{n_asymm_trades}", f"{n_asymm_trades}"],
        ["Hold Return", f"{cum_hold_last*100:.2f}%", "", "", ""],
    ]
    tbl = ax_stats.table(cellText=stats_data[1:], colLabels=stats_data[0],
                         loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    for key, cell in tbl.get_celld().items():
        if key[1] == 0:
            cell.set_text_props(fontweight="bold")
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
    ax_stats.set_title("Backtest Statistics", fontsize=13, fontweight="bold")

    
    ax_cm = fig.add_subplot(gs[4, 1])
    ax_cm.axis("off")

    y_true = actual_direction.astype(int)
    y_pred = pred_direction.astype(int)

    ax_cm.text(0.02, 0.98, "Classification Report", fontsize=11, fontweight="bold",
               transform=ax_cm.transAxes, va="top")
    cr_text = classification_report(y_true, y_pred, target_names=["DOWN", "UP"])
    ax_cm.text(0.02, 0.88, cr_text, fontsize=7.5, fontfamily="monospace",
               transform=ax_cm.transAxes, va="top")

    ax_cm.text(0.55, 0.98, "Confusion Matrix", fontsize=11, fontweight="bold",
               transform=ax_cm.transAxes, va="top")

    cm = confusion_matrix(y_true, y_pred)
    cm_ax = fig.add_axes([0.66, 0.17, 0.27, 0.12])
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

    
    ax = fig.add_subplot(gs[5, 0])
    ax.plot(dates, cum_asymm * 100, label="Asymmetric (Fixed Lot)", color="orange", linewidth=1.5)
    ax.plot(dates, cum_dynamic * 100, label="Dynamic Sizing", color="green", linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Fixed Lot vs Dynamic Sizing", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    
    ax_dd = ax.twinx()
    dd_asymm_curve = np.minimum(0, (np.cumprod(1 + clf_signals_asymm * actual_returns) /
                                     np.maximum.accumulate(np.cumprod(1 + clf_signals_asymm * actual_returns)) - 1) * 100)
    dd_dynamic_curve = np.minimum(0, (np.cumprod(1 + clf_signals_asymm * lot_multipliers * actual_returns) /
                                       np.maximum.accumulate(np.cumprod(1 + clf_signals_asymm * lot_multipliers * actual_returns)) - 1) * 100)
    ax_dd.fill_between(dates, dd_asymm_curve, 0, color="orange", alpha=0.15, label="_nolegend_")
    ax_dd.fill_between(dates, dd_dynamic_curve, 0, color="green", alpha=0.15, label="_nolegend_")
    ax_dd.set_ylabel("Drawdown (%)", fontsize=8, color="gray")
    ax_dd.tick_params(axis="y", labelsize=7, colors="gray")

    
    ax = fig.add_subplot(gs[5, 1])
    active_lot = lot_multipliers[atr_active]
    if len(active_lot) > 0:
        ax.hist(active_lot, bins=50, color="steelblue", edgecolor="black", alpha=0.7, density=True)
        ax.axvline(active_lot.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean={active_lot.mean():.2f}")
        ax.axvline(np.median(active_lot), color="orange", linestyle="--", linewidth=1.5,
                   label=f"Median={np.median(active_lot):.2f}")
        ax.set_title(f"Lot Multiplier Distribution (ATR active days, n={len(active_lot)})")
        ax.set_xlabel("Lot Multiplier")
        ax.set_ylabel("Density")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No ATR active days", ha="center", va="center")
    ax.grid(True, alpha=0.3)

    plt.savefig("backtest_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved to backtest_chart.png")


if __name__ == "__main__":
    run_backtest()