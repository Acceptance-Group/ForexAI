import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

from config import (
    DATA_CONFIG, TRAIN_CONFIG, DEVICE,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS,
)
from data_loader import load_dataset, load_raw_prices, load_labels


def build_tabular_features(features_np, features_df, labels_df, lookback):
    rows = []
    valid_indices = []
    label_indices = labels_df.index.intersection(features_df.index)

    for idx_str in label_indices:
        idx = features_df.index.get_loc(idx_str)
        if idx < lookback:
            continue
        seq = features_np[idx - lookback:idx]

        row = {}
        for feat_idx, feat_name in enumerate(FEATURE_COLUMNS):
            vals = seq[:, feat_idx]
            row[f"{feat_name}_mean"] = np.mean(vals)
            row[f"{feat_name}_std"] = np.std(vals)
            row[f"{feat_name}_last"] = vals[-1]
            row[f"{feat_name}_trend"] = (vals[-1] - vals[0]) / (np.std(vals) + 1e-10)
            row[f"{feat_name}_min"] = np.min(vals)
            row[f"{feat_name}_max"] = np.max(vals)

        rows.append(row)
        valid_indices.append(idx_str)

    df_tabular = pd.DataFrame(rows, index=valid_indices)
    labels = labels_df.loc[valid_indices, "label"]
    label_vals = (labels.values + 1) / 2.0
    return df_tabular, label_vals, valid_indices


def run_lgb_training():
    feats_df = load_dataset()
    prices_df = load_raw_prices()
    labels_df = load_labels()

    common_idx = feats_df.index.intersection(labels_df.index)
    feats_df = feats_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    price_common = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[price_common]
    labels_df = labels_df.loc[price_common]
    prices_df = prices_df.loc[price_common]

    print(f"Dataset: {len(feats_df)} samples, {len(FEATURE_COLUMNS)} features")

    lookback = DATA_CONFIG["lookback"]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats_df.values)

    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    X_tabular, y_tabular, valid_indices = build_tabular_features(
        scaled, feats_df, labels_df, lookback
    )

    X = X_tabular.values.astype(np.float32)
    y = y_tabular.astype(np.float32)

    print(f"Tabular features: {X.shape[1]} columns, {X.shape[0]} samples")
    print(f"Label distribution: UP={np.mean(y)*100:.1f}%")

    feature_names = list(X_tabular.columns)

    tscv = TimeSeriesSplit(n_splits=TRAIN_CONFIG["n_splits"])
    purge_gap = TRAIN_CONFIG["purge_gap"]

    best_score = 0.0
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        purge = max(0, min(val_idx) - purge_gap)
        train_idx_purged = train_idx[train_idx < purge]
        if len(train_idx_purged) < 100:
            train_idx_purged = train_idx

        X_train, y_train = X[train_idx_purged], y[train_idx_purged]
        X_val, y_val = X[val_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names)

        params = {
            "objective": "binary",
            "metric": "binary_error",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": 42,
            "is_unbalance": False,
        }

        callbacks = [
            lgb.log_evaluation(period=0),
            lgb.early_stopping(stopping_rounds=50),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        val_probs = model.predict(X_val)
        val_preds = (val_probs > 0.5).astype(float)
        da = np.mean(val_preds == y_val) * 100

        high_conf = (val_probs > 0.6) | (val_probs < 0.4)
        if high_conf.sum() > 0:
            da_high = np.mean(val_preds[high_conf] == y_val[high_conf]) * 100
            pct_high = high_conf.sum() / len(high_conf) * 100
        else:
            da_high = 0.0
            pct_high = 0.0

        print(f"\nFold {fold + 1}/{TRAIN_CONFIG['n_splits']}")
        print(f"  Train: {len(train_idx_purged)}, Val: {len(val_idx)}")
        print(f"  Val DA: {da:.1f}%")
        print(f"  DA High-conf: {da_high:.1f}% ({pct_high:.0f}%)")
        print(f"  Prob range: [{val_probs.min():.3f}, {val_probs.max():.3f}]")
        print(f"  Prob mean: {val_probs.mean():.3f}, std: {val_probs.std():.4f}")

        if da > best_score:
            best_score = da
            best_model = model

    print(f"\nBest Val DA: {best_score:.1f}%")

    importance = best_model.feature_importance(importance_type="gain")
    feature_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    print("\nFeature Importance (gain):")
    for fname, imp in feature_imp[:20]:
        bar = "#" * max(0, int(imp / max(importance) * 40))
        print(f"  {fname:30s} {imp:8.1f} {bar}")

    model_path = MODEL_SAVE_PATH.replace(".pth", "_lgb.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    scaler_path = SCALER_SAVE_PATH.replace(".pkl", "_lgb.pkl")
    tabular_info = {
        "feature_names": feature_names,
        "lookback": lookback,
    }
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "tabular_info": tabular_info}, f)

    print(f"LightGBM model saved to {model_path}")
    return best_model


if __name__ == "__main__":
    run_lgb_training()