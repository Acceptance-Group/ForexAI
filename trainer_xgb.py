import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from config import (
    TRAIN_CONFIG, DATA_CONFIG, BACKTEST_CONFIG,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, FEATURE_WEIGHTS,
)
from data_loader import load_dataset, load_raw_prices, load_labels, build_dataset, fetch_cross_symbols


def run_training():
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
    print(f"Feature weights will be applied as sample weights")

    sample_weights = np.where(y == 1, 1.0 / (pos_ratio + 1e-10), 1.0 / (neg_ratio + 1e-10))

    wf_months = TRAIN_CONFIG["walk_forward_months"]
    val_months = TRAIN_CONFIG["val_months"]
    step_months = TRAIN_CONFIG["step_months"]

    fold_results = []
    best_da_overall = 0.0
    best_model = None

    start_date = dates[0]
    end_date = dates[-1]
    first_val_start = start_date + pd.DateOffset(months=wf_months)
    current_val_start = first_val_start
    fold = 0

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

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        val_preds = (val_proba > 0.5).astype(int)
        da = np.mean(val_preds == y_val) * 100

        high_conf = (val_proba > 0.6) | (val_proba < 0.4)
        if high_conf.sum() > 0:
            da_hc = np.mean(val_preds[high_conf] == y_val[high_conf]) * 100
            pct_hc = high_conf.sum() / len(y_val) * 100
        else:
            da_hc = 0
            pct_hc = 0

        prob_std = np.std(val_proba)
        prob_mean = np.mean(val_proba)

        print(f"  Val_DA={da:.1f}% | DA_hi={da_hc:.1f}% ({pct_hc:.0f}%) | Prob: mean={prob_mean:.3f} std={prob_std:.4f}")

        fold_results.append({"fold": fold, "best_da": da})
        if da > best_da_overall:
            best_da_overall = da
            best_model = model

        current_val_start += pd.DateOffset(months=step_months)

    print(f"\n{'='*65}")
    print(f"Walk-Forward Results ({fold} folds):")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: DA={fr['best_da']:.1f}%")
    avg_da = np.mean([fr["best_da"] for fr in fold_results])
    print(f"  Average DA: {avg_da:.1f}%")
    print(f"  Best DA: {best_da_overall:.1f}%")

    if best_model is not None:
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        best_model.get_booster().save_model(MODEL_SAVE_PATH.replace(".pth", ".json"))
        print(f"\nBest XGBoost model saved (DA={best_da_overall:.1f}%)")

        importances = best_model.feature_importances_
        print(f"\nFeature Importance (XGBoost gain):")
        sorted_pairs = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])
        for fname, imp in sorted_pairs:
            bar = "#" * max(0, int(imp * 200))
            marker = "+" if imp > 0 else "-"
            print(f"  {fname:20s} {marker}{abs(imp):.4f} {bar}")

    return best_model, scaler


if __name__ == "__main__":
    run_training()