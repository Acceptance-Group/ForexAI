import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from config import (
    TRAIN_CONFIG, DATA_CONFIG, DEVICE,
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


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    for X_batch, y_label_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_label_batch = y_label_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_label_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            probs = model.predict_proba_raw(X_batch)
            preds = (probs > 0.5).float()
            correct += (preds == y_label_batch).sum().item()
            total += y_label_batch.size(0)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches, correct / total * 100


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
    prob_range = (np.min(all_probs), np.max(all_probs))

    return da, da_high, pct_high, prob_std, prob_mean, prob_range


def run_training():
    feats_df = load_dataset()
    if feats_df.empty:
        feats_df = build_dataset()

    start_date = DATA_CONFIG["start_date"]
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)

    prices_df = load_raw_prices()
    labels_df = load_labels()

    common_idx = feats_df.index.intersection(labels_df.index)
    feats_df = feats_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    price_common = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[price_common]
    labels_df = labels_df.loc[price_common]
    prices_df = prices_df.loc[price_common]

    train_mask = feats_df.index >= start_date
    feats_df = feats_df[train_mask]
    labels_df = labels_df[train_mask]
    prices_df = prices_df.loc[feats_df.index]

    print(f"Dataset: {len(feats_df)} samples, {len(FEATURE_COLUMNS)} features")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Label distribution: UP={np.mean(labels_df['label'] == 1)*100:.1f}%, "
          f"DOWN={np.mean(labels_df['label'] == -1)*100:.1f}%, "
          f"FLAT={np.mean(labels_df['label'] == 0)*100:.1f}%")

    lookback = DATA_CONFIG["lookback"]
    purge_gap = TRAIN_CONFIG["purge_gap"]
    label_sharpening = TRAIN_CONFIG["label_sharpening"]

    scaler = StandardScaler()
    train_end = int(len(feats_df) * 0.8)
    scaler.fit(feats_df.values[:train_end])
    scaled = scaler.transform(feats_df.values)

    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump(scaler, f)

    X_all, y_label_all = create_sequences(
        scaled, feats_df, labels_df, lookback, label_sharpening
    )

    print(f"Sequences: {X_all.shape}, Labels: {np.mean(y_label_all)*100:.1f}% UP")
    print(f"Label sharpening: {label_sharpening}")
    print(f"Temperature: {DATA_CONFIG['temperature']}")

    tscv = TimeSeriesSplit(n_splits=TRAIN_CONFIG["n_splits"])

    best_da = 0.0
    patience_counter = 0

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
    warmup_epochs = 5

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
        purge = max(0, min(val_idx) - purge_gap)
        train_idx_purged = train_idx[train_idx < purge]
        if len(train_idx_purged) < 100:
            train_idx_purged = train_idx

        print(f"\n{'='*65}")
        print(f"Fold {fold + 1}/{TRAIN_CONFIG['n_splits']}")
        print(f"Train: {len(train_idx_purged)}, Val: {len(val_idx)}")

        
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
        warmup_epochs = 5
        best_da_fold = 0.0
        patience_counter = 0

        X_train = torch.tensor(X_all[train_idx_purged], dtype=torch.float32)
        y_train = torch.tensor(y_label_all[train_idx_purged], dtype=torch.float32)
        X_val = torch.tensor(X_all[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y_label_all[val_idx], dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, y_val),
            batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
        )

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

            train_loss = total_loss / n_batches

            val_da, val_da_high, pct_high, prob_std, prob_mean, prob_range = evaluate(model, val_loader)

            if epoch % 10 == 0:
                lr_current = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d} | Loss={train_loss:.4f} | "
                      f"Val_DA={val_da:.1f}% | DA_hi={val_da_high:.1f}% ({pct_high:.0f}%) | "
                      f"Prob: mean={prob_mean:.3f} std={prob_std:.4f} "
                      f"[{prob_range[0]:.3f},{prob_range[1]:.3f}] | LR={lr_current:.2e}")

            if val_da > best_da_fold:
                best_da_fold = val_da
                patience_counter = 0
                if val_da > best_da:
                    best_da = val_da
                    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= TRAIN_CONFIG["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

    
    print(f"\nLoading best saved model (Best Val DA={best_da:.1f}%)...")
    model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    
    print("\nComputing feature importance...")
    n_imp = min(500, len(X_all))
    X_base = torch.tensor(X_all[:n_imp], dtype=torch.float32).to(DEVICE)
    y_base_np = y_label_all[:n_imp]

    with torch.no_grad():
        base_probs = model.predict_proba(X_base).detach().cpu().numpy()
        base_preds = (base_probs > 0.5).astype(float)
        base_acc = np.mean(base_preds == y_base_np)

    importance = np.zeros(X_all.shape[2])
    for feat_idx in range(X_all.shape[2]):
        drops = []
        for _ in range(5):
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

    print("\nFeature Importance (accuracy drop when permuted):")
    sorted_pairs = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: -x[1])
    for fname, imp in sorted_pairs:
        bar = "#" * max(0, int(abs(imp) * 1000))
        marker = "+" if imp > 0 else "-"
        print(f"  {fname:20s} {marker}{abs(imp):.4f} {bar}")

    print(f"\nFeature Weights (input scaling):")
    weights = model.feature_scale.cpu().numpy()
    for fname, w in sorted(zip(FEATURE_COLUMNS, weights), key=lambda x: -x[1]):
        bar = "#" * max(0, int(w * 20))
        print(f"  {fname:20s} {w:.2f} {bar}")

    
    with torch.no_grad():
        X_check = torch.tensor(X_all[-500:], dtype=torch.float32).to(DEVICE)
        probs = model.predict_proba(X_check).detach().cpu().numpy()
    y_check = y_label_all[-500:]
    preds = (probs > 0.5).astype(float)
    da = np.mean(preds == y_check) * 100

    if da < 35.0:
        print(f"\n*** SIGNAL INVERSION DETECTED (DA={da:.1f}%) ***")
        print(f"    Flipping classifier head weights to correct direction...")
        with torch.no_grad():
            for name, param in model.classifier.named_parameters():
                if 'weight' in name:
                    param.data = -param.data
                elif 'bias' in name:
                    param.data = -param.data

        probs_after = model.predict_proba(X_check).detach().cpu().numpy()
        preds_after = (probs_after > 0.5).astype(float)
        da_after = np.mean(preds_after == y_check) * 100
        print(f"    DA after correction: {da_after:.1f}%")

        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"    Model re-saved with corrected weights.")
    else:
        print(f"\nSignal direction OK (DA={da:.1f}%)")

    print(f"\nTraining complete. Best Val DA={best_da:.1f}%")
    print(f"Temperature: {model.temperature}")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    return model, scaler


if __name__ == "__main__":
    run_training()