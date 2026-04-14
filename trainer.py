import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import (
    TRAIN_CONFIG, DATA_CONFIG, DEVICE, BACKTEST_CONFIG,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, FEATURE_WEIGHTS,
)
from data_loader import load_dataset, load_raw_prices, load_labels, build_dataset
from model import ForexClassifier, FocalLoss


def create_sequences(features_np, features_df, labels_df, lookback, label_sharpening=0.0):
    X, y_label = [], []
    valid_indices = labels_df.index.intersection(features_df.index)

    for idx_str in valid_indices:
        idx = features_df.index.get_loc(idx_str)
        if idx < lookback:
            continue
        seq = features_np[idx - lookback:idx]
        label_row = labels_df.loc[idx_str]
        label_val = (label_row["label"] + 1) / 2.0

        if label_sharpening > 0:
            if label_val > 0.5:
                label_val = min(1.0, 0.5 + label_sharpening + (label_val - 0.5))
            elif label_val < 0.5:
                label_val = max(0.0, 0.5 - label_sharpening - (0.5 - label_val))

        X.append(seq)
        y_label.append(label_val)

    return np.array(X), np.array(y_label)


def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_label_batch in loader:
            X_batch = X_batch.to(DEVICE)
            probs = model.predict_proba(X_batch).cpu().numpy()
            labels = y_label_batch.numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    preds = (all_probs > 0.5).astype(float)
    da = np.mean(preds == all_labels) * 100

    high_conf = (all_probs > 0.6) | (all_probs < 0.4)
    if high_conf.sum() > 0:
        da_high = np.mean(preds[high_conf] == all_labels[high_conf]) * 100
        pct_high = high_conf.sum() / len(high_conf) * 100
    else:
        da_high = 0.0
        pct_high = 0.0

    prob_std = np.std(all_probs)
    prob_mean = np.mean(all_probs)

    return da, da_high, pct_high, prob_std, prob_mean


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
    train_mask = feats_df.index < bt_start
    feats_df = feats_df.loc[train_mask]
    labels_df = labels_df.loc[train_mask]
    prices_df = prices_df.loc[prices_df.index.isin(feats_df.index)]

    print(f"Train-only dataset: {len(feats_df)} samples (before {bt_start.strftime('%Y-%m-%d')}), {len(FEATURE_COLUMNS)} features")
    print(f"Label distribution: UP={np.mean(labels_df['label'] == 1)*100:.1f}%, "
          f"DOWN={np.mean(labels_df['label'] == -1)*100:.1f}%, "
          f"FLAT={np.mean(labels_df['label'] == 0)*100:.1f}%")

    lookback = DATA_CONFIG["lookback"]
    purge_gap = TRAIN_CONFIG["purge_gap"]
    label_sharpening = TRAIN_CONFIG["label_sharpening"]

    scaler = StandardScaler()
    scaler.fit(feats_df.values)
    scaled = scaler.transform(feats_df.values)

    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    X_all, y_label_all = create_sequences(scaled, feats_df, labels_df, lookback, label_sharpening)

    print(f"Sequences: {X_all.shape}, Labels: {np.mean(y_label_all)*100:.1f}% UP")

    dates = feats_df.index[lookback:lookback + len(X_all)]

    wf_months = TRAIN_CONFIG["walk_forward_months"]
    val_months = TRAIN_CONFIG["val_months"]
    step_months = TRAIN_CONFIG["step_months"]

    fold_results = []
    best_da_overall = 0.0

    start_date = dates[0]
    end_date = dates[-1]

    first_val_start = start_date + pd.DateOffset(months=wf_months)
    fold = 0
    current_val_start = first_val_start

    while current_val_start + pd.DateOffset(months=val_months) <= end_date + pd.DateOffset(days=1):
        current_val_end = current_val_start + pd.DateOffset(months=val_months)

        train_mask = dates < current_val_start
        val_mask = (dates >= current_val_start) & (dates < current_val_end)

        if train_mask.sum() < 500 or val_mask.sum() < 50:
            current_val_start += pd.DateOffset(months=step_months)
            continue

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        purge_start = val_indices[0] - purge_gap
        train_indices = train_indices[train_indices < purge_start]

        if len(train_indices) < 200 or len(val_indices) < 30:
            current_val_start += pd.DateOffset(months=step_months)
            continue

        X_train = torch.tensor(X_all[train_indices], dtype=torch.float32)
        y_train = torch.tensor(y_label_all[train_indices], dtype=torch.float32)
        X_val = torch.tensor(X_all[val_indices], dtype=torch.float32)
        y_val = torch.tensor(y_label_all[val_indices], dtype=torch.float32)

        fold += 1
        print(f"\n{'='*65}")
        print(f"Fold {fold}: Train {len(train_indices)} | Val {len(val_indices)}")
        print(f"  Train: {dates[train_indices[0]].strftime('%Y-%m-%d')} - {dates[train_indices[-1]].strftime('%Y-%m-%d')}")
        print(f"  Val:   {dates[val_indices[0]].strftime('%Y-%m-%d')} - {dates[val_indices[-1]].strftime('%Y-%m-%d')}")

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val),
            batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
        )

        model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
        criterion = FocalLoss(
            alpha=TRAIN_CONFIG["focal_alpha"],
            gamma=TRAIN_CONFIG["focal_gamma"]
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            weight_decay=TRAIN_CONFIG["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAIN_CONFIG["epochs"], eta_min=1e-6
        )

        best_da_fold = 0.0
        patience_counter = 0
        warmup_epochs = 5

        for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
            if epoch <= warmup_epochs:
                warmup_factor = epoch / warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = TRAIN_CONFIG["learning_rate"] * warmup_factor

            model.train()
            total_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            val_da, val_da_high, pct_high, prob_std, prob_mean = evaluate(model, val_loader)

            if epoch % 20 == 0:
                lr_current = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:3d} | Loss={total_loss/n_batches:.4f} | "
                      f"Val_DA={val_da:.1f}% | DA_hi={val_da_high:.1f}% ({pct_high:.0f}%) | "
                      f"Prob: mean={prob_mean:.3f} std={prob_std:.4f} | LR={lr_current:.2e}")

            if val_da > best_da_fold:
                best_da_fold = val_da
                patience_counter = 0
                if val_da > best_da_overall:
                    best_da_overall = val_da
                    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= TRAIN_CONFIG["patience"]:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        fold_results.append({"fold": fold, "best_da": best_da_fold})
        print(f"  Fold {fold} Best DA: {best_da_fold:.1f}%")

        current_val_start += pd.DateOffset(months=step_months)

    print(f"\n{'='*65}")
    print(f"Walk-Forward Results ({fold} folds):")
    for fr in fold_results:
        print(f"  Fold {fr['fold']}: DA={fr['best_da']:.1f}%")
    avg_da = np.mean([fr["best_da"] for fr in fold_results])
    print(f"  Average DA: {avg_da:.1f}%")
    print(f"  Best DA: {best_da_overall:.1f}%")

    print(f"\nLoading best model (Best DA={best_da_overall:.1f}%)...")
    model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        X_check = torch.tensor(X_all[-500:], dtype=torch.float32).to(DEVICE)
        probs = model.predict_proba(X_check).detach().cpu().numpy()
    y_check = y_label_all[-500:]
    preds = (probs > 0.5).astype(float)
    da = np.mean(preds == y_check) * 100

    if da < 35.0:
        print(f"\n*** SIGNAL INVERSION (DA={da:.1f}%) ***")
        with torch.no_grad():
            for name, param in model.classifier.named_parameters():
                if 'weight' in name:
                    param.data = -param.data
                elif 'bias' in name:
                    param.data = -param.data
        probs_after = model.predict_proba(X_check).detach().cpu().numpy()
        preds_after = (probs_after > 0.5).astype(float)
        da_after = np.mean(preds_after == y_check) * 100
        print(f"  DA after correction: {da_after:.1f}%")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        print(f"Signal direction OK (DA={da:.1f}%)")

    print("\nComputing feature importance...")
    n_imp = min(500, len(X_all))
    X_base = torch.tensor(X_all[:n_imp], dtype=torch.float32).to(DEVICE)
    y_base_np = y_label_all[:n_imp]

    with torch.no_grad():
        base_probs = model.predict_proba(X_base).detach().cpu().numpy()
        base_preds = (base_probs > 0.5).astype(float)
        base_acc = np.mean(base_preds == y_base_np)

    importance = np.zeros(len(FEATURE_COLUMNS))
    for feat_idx in range(len(FEATURE_COLUMNS)):
        drops = []
        for _ in range(3):
            X_perm = X_all[:n_imp].copy()
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

    print("\nFeature Importance:")
    sorted_pairs = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: -x[1])
    for fname, imp in sorted_pairs:
        bar = "#" * max(0, int(abs(imp) * 1000))
        marker = "+" if imp > 0 else "-"
        print(f"  {fname:20s} {marker}{abs(imp):.4f} {bar}")

    print(f"\nTraining complete. Best Val DA={best_da_overall:.1f}%")
    return model, scaler


if __name__ == "__main__":
    run_training()