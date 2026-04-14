import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from hmmlearn.hmm import GaussianHMM

from config import (
    TRAIN_CONFIG, DATA_CONFIG, BACKTEST_CONFIG,
    SCALER_SAVE_PATH, FEATURE_COLUMNS,
)
from data_loader import build_dataset, load_raw_prices, compute_atr


VOL_MODEL_PATH = "models/vol_model.json"
VOL_SCALER_PATH = "models/vol_scaler.pkl"
HMM_MODEL_PATH = "models/hmm_regime.pkl"
MEANREV_MODEL_PATH = "models/meanrev_model.json"
MEANREV_SCALER_PATH = "models/meanrev_scaler.pkl"


def run_volatility_training():
    print("=" * 65)
    print("TASK 1: Volatility Prediction (ATR Regression)")
    print("=" * 65)
    feats_df = build_dataset(force_download=False)
    prices_df = load_raw_prices()

    common_idx = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    close = prices_df["close"]
    high = prices_df["high"]
    low = prices_df["low"]
    atr_series = compute_atr(high, low, close, DATA_CONFIG["atr_period"])
    next_atr = atr_series.shift(-1)

    valid = next_atr.notna() & feats_df.notna().all(axis=1)
    feats_df_v = feats_df.loc[valid].copy()
    next_atr_v = next_atr.loc[valid].copy()

    bt_start = pd.Timestamp(BACKTEST_CONFIG["start_date"]) if isinstance(BACKTEST_CONFIG["start_date"], str) else BACKTEST_CONFIG["start_date"]
    train_mask = feats_df_v.index < bt_start
    X_train_full = feats_df_v.loc[train_mask].values
    y_train_full = next_atr_v.loc[train_mask].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_full)

    os.makedirs(os.path.dirname(VOL_SCALER_PATH), exist_ok=True)
    with open(VOL_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    wf_months = TRAIN_CONFIG["walk_forward_months"]
    val_months = TRAIN_CONFIG["val_months"]
    step_months = TRAIN_CONFIG["step_months"]
    dates = feats_df_v.loc[train_mask].index

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "min_child_weight": 20,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
        "n_jobs": -1,
    }

    start_date = dates[0]
    end_date = dates[-1]
    current_val_start = start_date + pd.DateOffset(months=wf_months)
    fold_results = []
    best_r2 = -999
    best_model = None

    while current_val_start + pd.DateOffset(months=val_months) <= end_date + pd.DateOffset(days=1):
        val_end = current_val_start + pd.DateOffset(months=val_months)
        tr_m = dates < current_val_start
        va_m = (dates >= current_val_start) & (dates < val_end)
        if tr_m.sum() < 500 or va_m.sum() < 50:
            current_val_start += pd.DateOffset(months=step_months)
            continue

        X_tr, y_tr = X_scaled[tr_m], y_train_full[tr_m]
        X_va, y_va = X_scaled[va_m], y_train_full[va_m]

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_va)
        r2 = r2_score(y_va, pred)
        mae = mean_absolute_error(y_va, pred)
        corr = np.corrcoef(y_va, pred)[0, 1]

        vol_dir_correct = np.mean((np.diff(pred) > 0) == (np.diff(y_va) > 0)) * 100

        fold_results.append({"r2": r2, "mae": mae, "corr": corr, "vol_dir": vol_dir_correct})
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
        if len(fold_results) % 10 == 0:
            print(f"  Fold {len(fold_results)}: R2={r2:.3f} Corr={corr:.3f} VolDir={vol_dir_correct:.1f}%")
        current_val_start += pd.DateOffset(months=step_months)

    avg_r2 = np.mean([r["r2"] for r in fold_results])
    avg_corr = np.mean([r["corr"] for r in fold_results])
    avg_vd = np.mean([r["vol_dir"] for r in fold_results])
    print(f"\n  VOL MODEL: Avg R2={avg_r2:.3f} | Avg Corr={avg_corr:.3f} | Avg VolDir={avg_vd:.1f}% | Best R2={best_r2:.3f}")

    best_model.save_model(VOL_MODEL_PATH)
    print(f"  Saved to {VOL_MODEL_PATH}")

    importances = best_model.feature_importances_
    print("\n  Top features:")
    for fname, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])[:8]:
        print(f"    {fname:20s} {imp:.4f}")

    return best_model, scaler


def run_hmm_regime():
    print("\n" + "=" * 65)
    print("TASK 2: HMM Regime Detection")
    print("=" * 65)
    prices_df = load_raw_prices()
    close = prices_df["close"].values
    high = prices_df["high"].values
    low = prices_df["low"].values

    log_ret = np.diff(np.log(close))
    atr_raw = compute_atr(pd.Series(high), pd.Series(low), pd.Series(close), 14).values
    atr_ret = np.diff(atr_raw) / (atr_raw[:-1] + 1e-10)
    vol_20 = np.array([np.std(log_ret[max(0, i-20):i]) if i >= 20 else np.nan for i in range(1, len(log_ret)+1)])

    min_len = min(len(log_ret), len(atr_ret), len(vol_20), len(prices_df) - 2)
    log_ret_s = log_ret[:min_len]
    atr_ret_s = atr_ret[:min_len]
    vol_20_s = vol_20[:min_len]
    dates_s = prices_df.index[2:2+min_len]

    valid = ~np.isnan(vol_20_s)
    log_ret_c = log_ret_s[valid]
    atr_ret_c = atr_ret_s[valid]
    vol_20_c = vol_20_s[valid]
    dates_c = dates_s[valid]

    obs = np.column_stack([log_ret_c, atr_ret_c])

    bt_start = pd.Timestamp(BACKTEST_CONFIG["start_date"]) if isinstance(BACKTEST_CONFIG["start_date"], str) else BACKTEST_CONFIG["start_date"]
    train_mask_np = dates_c < bt_start
    obs_train = obs[train_mask_np]

    best_hmm = None
    best_bic = np.inf
    for n_states in [2, 3]:
        try:
            hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
            hmm.fit(obs_train)
            ll = hmm.score(obs_train)
            n_params = n_states * (2 + 2 * 3) - 1
            bic = -2 * ll + n_params * np.log(len(obs_train))
            print(f"  HMM n_states={n_states}: LL={ll:.0f} BIC={bic:.0f}")
            if bic < best_bic:
                best_bic = bic
                best_hmm = hmm
        except Exception as e:
            print(f"  HMM n_states={n_states} failed: {e}")

    if best_hmm is None:
        print("  HMM training failed!")
        return None

    os.makedirs(os.path.dirname(HMM_MODEL_PATH), exist_ok=True)
    with open(HMM_MODEL_PATH, "wb") as f:
        pickle.dump(best_hmm, f)

    states_train = best_hmm.predict(obs_train)
    regime_info = []
    for s in range(best_hmm.n_components):
        mask = states_train == s
        mean_ret = log_ret_c[train_mask_np][:min(len(states_train), train_mask_np.sum())][mask].mean() * 252 * 100
        std_ret = log_ret_c[train_mask_np][:min(len(states_train), train_mask_np.sum())][mask].std() * np.sqrt(252) * 100
        pct = mask.sum() / len(mask) * 100
        regime_info.append({"state": s, "mean_ret": mean_ret, "vol": std_ret, "pct": pct})
        print(f"  State {s}: annualized_ret={mean_ret:.2f}% annualized_vol={std_ret:.1f}% pct={pct:.1f}%")

    trending_state = max(regime_info, key=lambda x: abs(x["mean_ret"]) + x["vol"])
    ranging_state = min(regime_info, key=lambda x: abs(x["mean_ret"]) + x["vol"])
    print(f"  Trending state: {trending_state['state']} (|ret|={abs(trending_state['mean_ret']):.2f}%, vol={trending_state['vol']:.1f}%)")
    print(f"  Ranging state: {ranging_state['state']} (|ret|={abs(ranging_state['mean_ret']):.2f}%, vol={ranging_state['vol']:.1f}%)")

    oos_mask = dates_c >= bt_start
    if oos_mask.sum() > 10:
        obs_bt = obs[oos_mask]
        states_bt = best_hmm.predict(obs_bt)
        trending_pct = np.mean(states_bt == trending_state["state"]) * 100
        print(f"  OOS: Trending {trending_pct:.1f}% of time")

        bt_log_ret = log_ret_c[oos_mask]
        dir_correct = 0
        total = 0
        for i in range(min(len(states_bt), len(bt_log_ret))):
            if states_bt[i] == trending_state["state"]:
                total += 1
                if abs(bt_log_ret[i]) > np.median(np.abs(bt_log_ret)):
                    dir_correct += 1
        if total > 0:
            print(f"  Big moves when trending: {dir_correct}/{total} ({dir_correct/total*100:.1f}%)")
    else:
        print(f"  No OOS data (all data used for training)")
        dir_correct = 0
        total = 0

    print(f"  Big moves when trending: {dir_correct}/{total} ({dir_correct/(total+1e-10)*100:.1f}%)")
    print(f"  Saved to {HMM_MODEL_PATH}")

    return best_hmm, trending_state["state"], ranging_state["state"]


def run_meanrev_training():
    print("\n" + "=" * 65)
    print("TASK 3: Mean-Reversion Prediction")
    print("=" * 65)
    feats_df = build_dataset(force_download=False)
    prices_df = load_raw_prices()

    common_idx = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    close = prices_df["close"].values
    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    dist = (close - sma20) / (sma20 + 1e-10)
    next_close = np.roll(close, -1)
    next_sma = np.roll(sma20, -1)
    next_dist = (next_close - next_sma) / (next_sma + 1e-10)
    reverted = (np.abs(next_dist) < np.abs(dist)).astype(int)

    valid = feats_df.notna().all(axis=1)
    valid_idx = valid[valid].index
    feats_df_mr = feats_df.loc[valid_idx].copy()

    bt_start = pd.Timestamp(BACKTEST_CONFIG["start_date"]) if isinstance(BACKTEST_CONFIG["start_date"], str) else BACKTEST_CONFIG["start_date"]
    train_mask = feats_df_mr.index < bt_start

    X_all = feats_df_mr.values
    y_all = reverted[:len(feats_df_mr)]

    min_len = min(len(X_all), len(y_all))
    X_all = X_all[:min_len]
    y_all = y_all[:min_len]
    train_mask = train_mask[:min_len]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    os.makedirs(os.path.dirname(MEANREV_SCALER_PATH), exist_ok=True)
    with open(MEANREV_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    y_train = y_all[train_mask]
    pos_ratio = y_train.mean()
    print(f"  Mean-reversion label: {pos_ratio*100:.1f}% revert")

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "min_child_weight": 20,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "n_jobs": -1,
    }

    dates = feats_df_mr.index[:min_len]
    X_train_full = X_scaled

    wf_months = TRAIN_CONFIG["walk_forward_months"]
    val_months = TRAIN_CONFIG["val_months"]
    step_months = TRAIN_CONFIG["step_months"]
    start_date = dates[0] if isinstance(dates, pd.DatetimeIndex) else pd.Timestamp(dates[0])
    end_date = dates[-1] if isinstance(dates, pd.DatetimeIndex) else pd.Timestamp(dates[-1])

    train_dates = dates[train_mask]
    current_val_start = train_dates[0] + pd.DateOffset(months=wf_months)
    fold_results = []
    best_da = 0
    best_model = None

    while current_val_start + pd.DateOffset(months=val_months) <= end_date + pd.DateOffset(days=1):
        val_end = current_val_start + pd.DateOffset(months=val_months)
        tr_m = train_dates < current_val_start
        va_m = (train_dates >= current_val_start) & (train_dates < val_end)
        if tr_m.sum() < 500 or va_m.sum() < 50:
            current_val_start += pd.DateOffset(months=step_months)
            continue

        X_tr = X_train_full[train_mask][tr_m]
        y_tr = y_all[train_mask][tr_m]
        X_va = X_train_full[train_mask][va_m]
        y_va = y_all[train_mask][va_m]

        pos_tr = y_tr.sum()
        neg_tr = len(y_tr) - pos_tr
        sw = np.where(y_tr == 1, neg_tr / (pos_tr + 1e-10), 1.0)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        pred = (model.predict_proba(X_va)[:, 1] > 0.5).astype(int)
        da = np.mean(pred == y_va) * 100
        fold_results.append(da)
        if da > best_da:
            best_da = da
            best_model = model
        current_val_start += pd.DateOffset(months=step_months)

    avg_da = np.mean(fold_results)
    print(f"  MeanRev Model: Avg DA={avg_da:.1f}%, Best={best_da:.1f}%")
    best_model.save_model(MEANREV_MODEL_PATH)
    print(f"  Saved to {MEANREV_MODEL_PATH}")
    return best_model, scaler


if __name__ == "__main__":
    vol_result = run_volatility_training()
    hmm_result = run_hmm_regime()
    mr_result = run_meanrev_training()

    print("\n" + "=" * 65)
    print("ALL TASKS COMPLETE")
    print(f"  Volatility: {VOL_MODEL_PATH}")
    print(f"  HMM Regime: {HMM_MODEL_PATH}")
    print(f"  Mean-Rev: {MEANREV_MODEL_PATH}")
    print("=" * 65)