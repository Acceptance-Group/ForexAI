import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config import (
    TRAIN_CONFIG, DATA_CONFIG, BACKTEST_CONFIG,
    SCALER_SAVE_PATH, FEATURE_COLUMNS,
)
from data_loader import build_dataset, load_raw_prices, load_labels


LONG_MODEL_PATH = "models/long_model.json"
SHORT_MODEL_PATH = "models/short_model.json"


def run_split_training():
    feats_df = build_dataset(force_download=True)
    prices_df = load_raw_prices()
    labels_df = load_labels()

    common_idx = feats_df.index.intersection(labels_df.index)
    feats_df = feats_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]
    price_common = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[price_common]
    labels_df = labels_df.loc[price_common]
    prices_df = prices_df.loc[prices_df.index.isin(feats_df.index)]

    bt_start = BACKTEST_CONFIG["start_date"]
    if isinstance(bt_start, str):
        bt_start = pd.Timestamp(bt_start)
    train_mask = feats_df.index < bt_start
    feats_df = feats_df.loc[train_mask]
    labels_df = labels_df.loc[train_mask]

    print(f"Train-only dataset: {len(feats_df)} samples, {len(FEATURE_COLUMNS)} features")

    scaler = StandardScaler()
    scaler.fit(feats_df.values)
    scaled = scaler.transform(feats_df.values)

    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    label_vals = labels_df["label"].values
    valid_mask = label_vals != 0

    X_orig = scaled[valid_mask]
    y_binary = ((label_vals[valid_mask] + 1) / 2.0).astype(int)
    dates = feats_df.index[valid_mask]

    y_long = y_binary.copy()
    y_short = (1 - y_binary).copy()

    pos_long = y_long.sum()
    neg_long = len(y_long) - pos_long
    pos_short = y_short.sum()
    neg_short = len(y_short) - pos_short

    print(f"LONG: {pos_long} UP vs {neg_long} DOWN ({pos_long/len(y_long)*100:.1f}% UP)")
    print(f"SHORT: {pos_short} DOWN vs {neg_short} UP ({pos_short/len(y_short)*100:.1f}% DOWN)")

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

    def train_model(y_target, model_name, model_path):
        pos_ratio = y_target.sum() / len(y_target)
        sw = np.where(y_target == 1, 1.0 / (pos_ratio + 1e-10), 1.0 / (1 - pos_ratio + 1e-10))

        wf_months = TRAIN_CONFIG["walk_forward_months"]
        val_months = TRAIN_CONFIG["val_months"]
        step_months = TRAIN_CONFIG["step_months"]

        start_date = dates[0]
        end_date = dates[-1]
        current_val_start = start_date + pd.DateOffset(months=wf_months)

        fold_results = []
        best_da = 0.0
        best_model = None

        while current_val_start + pd.DateOffset(months=val_months) <= end_date + pd.DateOffset(days=1):
            val_end = current_val_start + pd.DateOffset(months=val_months)
            tr_mask = dates < current_val_start
            va_mask = (dates >= current_val_start) & (dates < val_end)

            if tr_mask.sum() < 500 or va_mask.sum() < 50:
                current_val_start += pd.DateOffset(months=step_months)
                continue

            X_tr, y_tr = X_orig[tr_mask], y_target[tr_mask]
            X_va, y_va = X_orig[va_mask], y_target[va_mask]
            sw_tr = sw[tr_mask]

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_tr, y_tr, sample_weight=sw_tr, verbose=False)

            val_probs = model.predict_proba(X_va)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            da = np.mean(val_preds == y_va) * 100

            fold_results.append(da)
            if da > best_da:
                best_da = da
                best_model = model

            current_val_start += pd.DateOffset(months=step_months)

        avg_da = np.mean(fold_results)
        print(f"\n{model_name} - Avg DA: {avg_da:.1f}%, Best: {best_da:.1f}%")
        best_model.save_model(model_path)
        return best_model, avg_da

    print("\n" + "="*65)
    print("Training LONG model (P(UP))...")
    print("="*65)
    long_model, long_da = train_model(y_long, "LONG", LONG_MODEL_PATH)

    print("\n" + "="*65)
    print("Training SHORT model (P(DOWN))...")
    print("="*65)
    short_model, short_da = train_model(y_short, "SHORT", SHORT_MODEL_PATH)

    print(f"\n{'='*65}")
    print(f"RESULTS: LONG DA={long_da:.1f}% | SHORT DA={short_da:.1f}%")
    print(f"{'='*65}")

    return long_model, short_model


if __name__ == "__main__":
    run_split_training()