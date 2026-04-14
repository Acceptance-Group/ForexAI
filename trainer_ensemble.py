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


N_MODELS = 5
ENSEMBLE_MODEL_DIR = "models/ensemble"


def run_ensemble_training():
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
    train_mask = feats_df.index < bt_start
    feats_df = feats_df.loc[train_mask]
    labels_df = labels_df.loc[train_mask]
    prices_df = prices_df.loc[prices_df.index.isin(feats_df.index)]

    print(f"Train dataset: {len(feats_df)} samples, {len(FEATURE_COLUMNS)} features")

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

    wf_months = TRAIN_CONFIG["walk_forward_months"]
    val_months = TRAIN_CONFIG["val_months"]
    step_months = TRAIN_CONFIG["step_months"]

    start_date = dates[0]
    end_date = dates[-1]
    first_val_start = start_date + pd.DateOffset(months=wf_months)
    current_val_start = first_val_start

    folds_data = []
    while current_val_start + pd.DateOffset(months=val_months) <= end_date + pd.DateOffset(days=1):
        current_val_end = current_val_start + pd.DateOffset(months=val_months)
        train_mask_d = dates < current_val_start
        val_mask_d = (dates >= current_val_start) & (dates < current_val_end)
        if train_mask_d.sum() < 500 or val_mask_d.sum() < 50:
            current_val_start += pd.DateOffset(months=step_months)
            continue
        folds_data.append({
            "train_mask": train_mask_d,
            "val_mask": val_mask_d,
            "val_start": current_val_start,
            "val_end": current_val_end,
        })
        current_val_start += pd.DateOffset(months=step_months)

    print(f"\nTotal walk-forward folds: {len(folds_data)}")
    print(f"Training ensemble of {N_MODELS} models...\n")

    os.makedirs(ENSEMBLE_MODEL_DIR, exist_ok=True)

    all_model_results = []

    for model_idx in range(N_MODELS):
        seed = 42 + model_idx * 7
        print(f"{'='*65}")
        print(f"MODEL {model_idx + 1}/{N_MODELS} (seed={seed})")
        print(f"{'='*65}")

        xgb_params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.01,
            "reg_lambda": 0.1,
            "min_child_weight": 20,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "n_jobs": -1,
            "random_state": seed,
        }

        best_da = 0.0
        best_model = None

        for fold_idx, fold in enumerate(folds_data):
            X_train = X[fold["train_mask"]]
            y_train = y[fold["train_mask"]]
            X_val = X[fold["val_mask"]]
            y_val = y[fold["val_mask"]]

            pos_tr = y_train.sum()
            neg_tr = len(y_train) - pos_tr
            scale_pos = neg_tr / (pos_tr + 1e-10)
            sw_train = np.where(y_train == 1, scale_pos, 1.0)

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, y_train, sample_weight=sw_train, verbose=False)

            val_proba = model.predict_proba(X_val)[:, 1]
            val_preds = (val_proba > 0.5).astype(int)
            da = np.mean(val_preds == y_val) * 100

            if da > best_da:
                best_da = da
                best_model = model

        model_path = os.path.join(ENSEMBLE_MODEL_DIR, f"model_{model_idx}.json")
        best_model.save_model(model_path)
        all_model_results.append({"seed": seed, "best_da": best_da})
        print(f"  Best fold DA: {best_da:.1f}%")

    print(f"\n{'='*65}")
    print(f"ENSEMBLE SUMMARY ({N_MODELS} models)")
    print(f"{'='*65}")
    for r in all_model_results:
        print(f"  Seed {r['seed']}: Best DA={r['best_da']:.1f}%")
    avg_da = np.mean([r["best_da"] for r in all_model_results])
    print(f"  Average Best DA: {avg_da:.1f}%")

    return all_model_results


if __name__ == "__main__":
    run_ensemble_training()