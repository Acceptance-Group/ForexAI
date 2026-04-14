import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

from config import (
    TRAIN_CONFIG, DATA_CONFIG, BACKTEST_CONFIG,
    SCALER_SAVE_PATH, FEATURE_COLUMNS,
)
from data_loader import build_dataset, load_raw_prices, compute_atr


REGIME_MODEL_PATH = "models/regime_model.json"
REGIME_SCALER_PATH = "models/regime_scaler.pkl"


def compute_regime_labels(prices_series: pd.Series, atr_series: pd.Series):
    next_ret = prices_series.pct_change().shift(-1)
    next_abs_ret = next_ret.abs()
    threshold = 0.5 * atr_series.values / prices_series.values
    threshold = pd.Series(threshold, index=prices_series.index)
    trending = (next_abs_ret > threshold).astype(int)
    results = pd.DataFrame({
        "regime_label": trending,
        "next_abs_ret": next_abs_ret,
    }, index=prices_series.index)
    results = results.replace([np.inf, -np.inf], np.nan).dropna()
    return results


def run_regime_training():
    feats_df = build_dataset(force_download=True)
    prices_df = load_raw_prices()

    common_idx = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[common_idx]
    prices_df = prices_df.loc[common_idx]

    close = prices_df["close"]
    high = prices_df["high"]
    low = prices_df["low"]
    atr_series = compute_atr(high, low, close, DATA_CONFIG["atr_period"])

    regime_df = compute_regime_labels(close, atr_series)
    common_idx2 = feats_df.index.intersection(regime_df.index)
    feats_df = feats_df.loc[common_idx2]
    regime_df = regime_df.loc[common_idx2]
    prices_df = prices_df.loc[prices_df.index.isin(common_idx2)]

    bt_start = BACKTEST_CONFIG["start_date"]
    if isinstance(bt_start, str):
        bt_start = pd.Timestamp(bt_start)
    train_mask = feats_df.index < bt_start
    feats_train = feats_df.loc[train_mask]
    regime_train = regime_df.loc[train_mask]

    y = regime_train["regime_label"].values
    pos_ratio = y.mean()
    print(f"Regime training data: {len(y)} samples")
    print(f"  TRENDING: {pos_ratio*100:.1f}% | RANGING: {(1-pos_ratio)*100:.1f}%")

    scaler = StandardScaler()
    X = scaler.fit_transform(feats_train.values)

    os.makedirs(os.path.dirname(REGIME_SCALER_PATH), exist_ok=True)
    with open(REGIME_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    dates = feats_train.index
    wf_months = TRAIN_CONFIG["walk_forward_months"]
    val_months = TRAIN_CONFIG["val_months"]
    step_months = TRAIN_CONFIG["step_months"]

    start_date = dates[0]
    end_date = dates[-1]
    first_val_start = start_date + pd.DateOffset(months=wf_months)
    current_val_start = first_val_start
    fold = 0
    fold_results = []
    best_f1 = 0.0
    best_model = None

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

    print(f"\n{'='*65}")
    print("REGIME MODEL - Walk-Forward Validation")
    print(f"{'='*65}")

    while current_val_start + pd.DateOffset(months=val_months) <= end_date + pd.DateOffset(days=1):
        current_val_end = current_val_start + pd.DateOffset(months=val_months)
        train_mask_d = dates < current_val_start
        val_mask_d = (dates >= current_val_start) & (dates < current_val_end)

        if train_mask_d.sum() < 500 or val_mask_d.sum() < 50:
            current_val_start += pd.DateOffset(months=step_months)
            continue

        X_tr = X[train_mask_d]
        y_tr = y[train_mask_d]
        X_va = X[val_mask_d]
        y_va = y[val_mask_d]

        pos_tr = y_tr.sum()
        neg_tr = len(y_tr) - pos_tr
        scale_pos = neg_tr / (pos_tr + 1e-10)
        sw_tr = np.where(y_tr == 1, scale_pos, 1.0)

        fold += 1
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)

        y_pred_proba = model.predict_proba(X_va)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        acc = np.mean(y_pred == y_va) * 100
        f1 = f1_score(y_va, y_pred, zero_division=0)
        trending_pct = (y_pred_proba > 0.5).mean() * 100

        print(f"  Fold {fold}: Acc={acc:.1f}% | F1={f1:.3f} | P(trending)={trending_pct:.0f}%")
        fold_results.append({"fold": fold, "acc": acc, "f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

        current_val_start += pd.DateOffset(months=step_months)

    avg_acc = np.mean([r["acc"] for r in fold_results])
    avg_f1 = np.mean([r["f1"] for r in fold_results])
    print(f"\n  Average: Acc={avg_acc:.1f}% | F1={avg_f1:.3f} | Best F1={best_f1:.3f}")

    if best_model is not None:
        best_model.save_model(REGIME_MODEL_PATH)
        print(f"Regime model saved to {REGIME_MODEL_PATH}")

        importances = best_model.feature_importances_
        print("\nRegime Model Feature Importance:")
        sorted_pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])
        for fname, imp in sorted_pairs:
            bar = "#" * max(0, int(imp * 200))
            print(f"  {fname:20s} {imp:.4f} {bar}")

    return best_model, scaler


if __name__ == "__main__":
    run_regime_training()