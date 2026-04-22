import os
import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from config import (
    TRAIN_CONFIG, DATA_CONFIG, BACKTEST_CONFIG,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, FEATURE_WEIGHTS,
)
from data_loader import load_dataset, load_raw_prices, load_labels, build_dataset

ENSEMBLE_DIR = "models"
XGB_MODEL_PATH = os.path.join(ENSEMBLE_DIR, "dir_xgb.json")
LGBM_MODEL_PATH = os.path.join(ENSEMBLE_DIR, "dir_lgbm.txt")
CB_MODEL_PATH = os.path.join(ENSEMBLE_DIR, "dir_cb.cbm")

CACHE_FILES = [
    DATA_CONFIG.get("parquet_path", "data/eurusd_d1_features.parquet"),
    "data/eurusd_d1_features_labels.parquet",
    DATA_CONFIG.get("raw_parquet_path", "data/eurusd_d1_raw.parquet"),
]


def _clear_cache():
    for f in CACHE_FILES:
        if f and os.path.exists(f):
            os.remove(f)
            print(f"  Cache cleared: {f}")


def run_training():
    _clear_cache()
    feats_df = build_dataset(force_download=True)
    prices_df = load_raw_prices()
    labels_df = load_labels()

    common_idx = feats_df.index.intersection(labels_df.index)
    feats_df = feats_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]
    price_common = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[price_common]
    labels_df = labels_df.loc[price_common]
    prices_df = prices_df.loc[price_common]

    bt_start = BACKTEST_CONFIG["start_date"]
    if isinstance(bt_start, str):
        bt_start = pd.Timestamp(bt_start)
    use_all_data = os.environ.get("USE_ALL_DATA", "0") == "1"
    if not use_all_data:
        train_mask = feats_df.index < bt_start
        feats_df = feats_df.loc[train_mask]
        labels_df = labels_df.loc[train_mask]
        prices_df = prices_df.loc[prices_df.index.isin(feats_df.index)]

    data_desc = "all data" if use_all_data else bt_start.strftime("%Y-%m-%d")
    print(f"Dataset: {len(feats_df)} samples (before {data_desc}), {len(FEATURE_COLUMNS)} features")
    print(f"Label distribution: UP={np.mean(labels_df['label'] == 1)*100:.1f}%, "
          f"DOWN={np.mean(labels_df['label'] == -1)*100:.1f}%, "
          f"FLAT={np.mean(labels_df['label'] == 0)*100:.1f}%")

    scaler = StandardScaler()
    scaler.fit(feats_df.values)
    scaled = scaler.transform(feats_df.values)

    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    y_binary = ((labels_df["label"].values + 1) / 2.0).astype(int)
    y_binary[y_binary == 0.5] = 0
    valid_mask = y_binary != -1
    X = scaled[valid_mask]
    y = y_binary[valid_mask]
    dates = feats_df.index[valid_mask]

    pos_ratio = y.sum() / len(y)
    neg_ratio = 1 - pos_ratio
    print(f"After removing FLAT: {len(y)} samples, UP={pos_ratio*100:.1f}%")

    wf_months = TRAIN_CONFIG["walk_forward_months"]
    val_months = TRAIN_CONFIG["val_months"]
    step_months = TRAIN_CONFIG["step_months"]

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

    lgbm_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "objective": "binary",
        "verbosity": -1,
        "n_jobs": -1,
    }

    cb_params = {
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bylevel": 0.8,
        "l2_leaf_reg": 0.1,
        "min_data_in_leaf": 20,
        "loss_function": "Logloss",
        "verbose": 0,
        "thread_count": -1,
    }

    fold_results = []
    best_ens_da = 0.0
    best_xgb = None
    best_lgbm = None
    best_cb = None

    start_date = dates[0]
    end_date = dates[-1]
    first_val_start = start_date + pd.DateOffset(months=wf_months)
    current_val_start = first_val_start
    fold = 0

    while current_val_start + pd.DateOffset(months=val_months) <= end_date + pd.DateOffset(days=1):
        current_val_end = current_val_start + pd.DateOffset(months=val_months)

        train_mask_dates = dates < current_val_start
        val_mask_dates = (dates >= current_val_start) & (dates < current_val_end)

        if train_mask_dates.sum() < 500 or val_mask_dates.sum() < 50:
            current_val_start += pd.DateOffset(months=step_months)
            continue

        X_train = X[train_mask_dates]
        y_train = y[train_mask_dates]
        X_val = X[val_mask_dates]
        y_val = y[val_mask_dates]

        pos_tr = y_train.sum()
        neg_tr = len(y_train) - pos_tr
        scale_pos = neg_tr / (pos_tr + 1e-10)
        sw_train = np.where(y_train == 1, scale_pos, 1.0)

        fold += 1
        print(f"\n{'='*65}")
        print(f"Fold {fold}: Train {len(X_train)} | Val {len(X_val)}")
        print(f"  Train: {dates[train_mask_dates][0].strftime('%Y-%m-%d')} - {dates[train_mask_dates][-1].strftime('%Y-%m-%d')}")
        print(f"  Val:   {dates[val_mask_dates][0].strftime('%Y-%m-%d')} - {dates[val_mask_dates][-1].strftime('%Y-%m-%d')}")

        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_train, y_train, sample_weight=sw_train, eval_set=[(X_val, y_val)], verbose=False)
        p_xgb = m_xgb.predict_proba(X_val)[:, 1]

        m_lgbm = lgb.LGBMClassifier(**lgbm_params)
        m_lgbm.fit(X_train, y_train, sample_weight=sw_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
        p_lgbm = m_lgbm.predict_proba(X_val)[:, 1]

        m_cb = CatBoostClassifier(**cb_params)
        m_cb.fit(X_train, y_train, sample_weight=sw_train, eval_set=(X_val, y_val), verbose=0)
        p_cb = m_cb.predict_proba(X_val)[:, 1]

        p_ens = (p_xgb + p_lgbm + p_cb) / 3.0

        for name, p in [("XGB", p_xgb), ("LGBM", p_lgbm), ("CB", p_cb)]:
            da = np.mean((p > 0.5).astype(int) == y_val) * 100
            print(f"  {name}: DA={da:.1f}% mean_P={p.mean():.3f} std_P={p.std():.4f}")

        ens_preds = (p_ens > 0.5).astype(int)
        ens_da = np.mean(ens_preds == y_val) * 100

        high_conf = (p_ens > 0.6) | (p_ens < 0.4)
        if high_conf.sum() > 0:
            da_hc = np.mean(ens_preds[high_conf] == y_val[high_conf]) * 100
        else:
            da_hc = 0

        print(f"  ENSEMBLE: DA={ens_da:.1f}% | DA_hi={da_hc:.1f}% | mean_P={p_ens.mean():.3f}")

        fold_results.append({"fold": fold, "ens_da": ens_da})
        if ens_da > best_ens_da:
            best_ens_da = ens_da
            best_xgb = m_xgb
            best_lgbm = m_lgbm
            best_cb = m_cb

        current_val_start += pd.DateOffset(months=step_months)

    print(f"\n{'='*65}")
    print(f"Walk-Forward Ensemble Results ({fold} folds):")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: Ens_DA={fr['ens_da']:.1f}%")
    avg_da = np.mean([fr["ens_da"] for fr in fold_results])
    print(f"  Average Ensemble DA: {avg_da:.1f}%")
    print(f"  Best Ensemble DA: {best_ens_da:.1f}%")

    if best_xgb is not None:
        os.makedirs(ENSEMBLE_DIR, exist_ok=True)

        best_xgb.get_booster().save_model(XGB_MODEL_PATH)
        best_lgbm.booster_.save_model(LGBM_MODEL_PATH)
        best_cb.save_model(CB_MODEL_PATH)

        best_xgb.get_booster().save_model(MODEL_SAVE_PATH.replace(".pth", ".json"))

        print(f"\nModels saved:")
        print(f"  XGB:  {XGB_MODEL_PATH}")
        print(f"  LGBM: {LGBM_MODEL_PATH}")
        print(f"  CB:   {CB_MODEL_PATH}")

        print(f"\nFeature Importance (XGBoost gain):")
        importances = best_xgb.feature_importances_
        for fname, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])[:15]:
            bar = "#" * max(0, int(imp * 200))
            print(f"  {fname:20s} {imp:.4f} {bar}")

    return best_xgb, best_lgbm, best_cb, scaler


if __name__ == "__main__":
    run_training()
