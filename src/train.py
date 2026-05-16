"""
train.py — Training script for the AI Portfolio Analyzer.

Usage (local):
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --ticker AAPL  # single ticker override

Supports:
  - LSTM / Transformer (PyTorch)
  - XGBoost / LightGBM (baseline)
  - W&B experiment tracking
  - Early stopping + checkpoint saving
  - Sentiment feature integration
  - Multi-ticker training (all tickers in config concatenated)
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, set_seed, detect_device, init_wandb, wandb_log, finish_wandb
from src.dataset import (
    build_processed_dataset,
    time_series_split,
    fit_scaler,
    make_dataloaders,
    get_feature_cols,
    TARGET_COLS,
)
from src.model import build_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ============================================================
# Loss & Metrics
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for multi-output predictions."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    metrics = {}
    target_names = ["return", "volatility", "downside"]
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        metrics[f"mse_{name}"] = float(mean_squared_error(yt, yp))
        metrics[f"mae_{name}"] = float(mean_absolute_error(yt, yp))
        metrics[f"r2_{name}"] = float(r2_score(yt, yp))
    return metrics


# ============================================================
# Multi-ticker dataset builder
# ============================================================

def build_multi_ticker_splits(tickers: list, cfg: dict, use_sentiment: bool):
    """
    Download and process every ticker, split each chronologically, then
    concatenate the splits across tickers.

    Each ticker is split independently (preserving time order within each
    ticker) before concatenation, so there is no cross-ticker look-ahead.

    Returns:
        train_df, val_df, test_df  — concatenated DataFrames
        feature_cols               — list of feature column names
        target_cols                — list of target column names
    """
    train_parts, val_parts, test_parts = [], [], []
    feature_cols = None
    target_cols = None
    skipped = []

    for t in tickers:
        try:
            df = build_processed_dataset(t, cfg, sentiment_df=None)

            if feature_cols is None:
                # Determine columns from the first successfully built ticker
                feature_cols = get_feature_cols(use_sentiment=use_sentiment)
                feature_cols = [c for c in feature_cols if c in df.columns]
                target_cols = [c for c in TARGET_COLS if c in df.columns]

            # Only keep columns that exist; silently skip tickers that are too short
            available = [c for c in feature_cols + target_cols if c in df.columns]
            df = df[available].copy()

            if len(df) < cfg["data"]["sequence_length"] * 3:
                logger.warning(f"Skipping {t}: only {len(df)} rows after processing.")
                skipped.append(t)
                continue

            tr, va, te = time_series_split(
                df,
                val_frac=cfg["training"]["val_split"],
                test_frac=cfg["training"]["test_split"],
            )
            train_parts.append(tr)
            val_parts.append(va)
            test_parts.append(te)
            logger.info(f"  {t}: train={len(tr)}  val={len(va)}  test={len(te)}")

        except Exception as e:
            logger.warning(f"Failed to build dataset for {t}: {e}")
            skipped.append(t)

    if not train_parts:
        raise RuntimeError("No ticker data was successfully processed.")

    if skipped:
        logger.warning(f"Skipped tickers: {skipped}")

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df   = pd.concat(val_parts,   ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)

    logger.info(
        f"Combined dataset — train={len(train_df)}  val={len(val_df)}  test={len(test_df)}"
        f"  ({len(train_parts)} tickers)"
    )
    return train_df, val_df, test_df, feature_cols, target_cols


# ============================================================
# Deep-learning training loop
# ============================================================

def train_dl_model(model, train_loader, val_loader, cfg: dict, device: str, run=None):
    """Train a PyTorch model (LSTM or Transformer).

    Returns: (trained_model, history dict)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    tcfg = cfg["training"]
    model = model.to(device)

    # Loss: MSE for return & vol, BCEWithLogitsLoss for downside probability.
    # NOTE: BCEWithLogitsLoss expects raw logits. At inference time we must
    # apply torch.sigmoid() before treating output[2] as a probability.
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    opt_name = tcfg.get("optimizer", "adam").lower()
    lr = tcfg.get("learning_rate", 1e-3)
    wd = tcfg.get("weight_decay", 1e-4)

    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    epochs = tcfg.get("epochs", 100)
    patience = tcfg.get("early_stopping_patience", 15)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pt"

    log_freq = cfg.get("wandb", {}).get("log_freq", 10)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)

            loss_ret = mse_loss(pred[:, 0], yb[:, 0])
            loss_vol = mse_loss(pred[:, 1], yb[:, 1])
            # pred[:, 2] is a raw logit; BCEWithLogitsLoss applies sigmoid internally
            loss_dn  = bce_loss(pred[:, 2], yb[:, 2])
            loss = loss_ret + loss_vol + 0.5 * loss_dn

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

            if run and step % log_freq == 0:
                wandb_log(run, {"batch_loss": loss.item()})

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                l = (
                    mse_loss(pred[:, 0], yb[:, 0])
                    + mse_loss(pred[:, 1], yb[:, 1])
                    + 0.5 * bce_loss(pred[:, 2], yb[:, 2])
                )
                val_losses.append(l.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(f"Epoch {epoch:03d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")
        wandb_log(run, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # ── Checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "cfg": cfg,
            }, best_ckpt)
            logger.info(f"  ✓ New best checkpoint saved (val_loss={val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {epoch} epochs.")
                break

    # Load best weights back
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Best model loaded from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")
    return model, history


# ============================================================
# Baseline training
# ============================================================

def _flatten_for_baseline(df: pd.DataFrame, feature_cols: list, target_cols: list,
                           seq_len: int, scaler) -> tuple:
    """
    Convert a DataFrame into (X, y) arrays for XGBoost/LightGBM.

    Instead of collapsing the sequence to a single mean (which destroys all
    ordering information), we concatenate lag features: the flattened window
    gives the model visibility into the full sequence in one flat vector.
    Shape: X → (n_samples, seq_len * n_features), y → (n_samples, n_targets)
    """
    Xarr = scaler.transform(df[feature_cols].values)
    yarr = df[target_cols].values
    X, y = [], []
    for i in range(seq_len, len(Xarr)):
        # Flatten the full window rather than averaging, preserving time order
        X.append(Xarr[i - seq_len:i].flatten())
        y.append(yarr[i])
    return np.array(X), np.array(y)


def train_baseline(model, X_train, y_train, X_val, y_val, cfg: dict, run=None):
    """Train an XGBoost or LightGBM model."""
    bcfg = cfg.get("baseline", {})
    model.fit(
        X_train, y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=bcfg.get("early_stopping_rounds", 50),
    )
    preds = model.predict(X_train)
    train_metrics = compute_metrics(y_train, preds)
    val_preds = model.predict(X_val)
    val_metrics = {"val_" + k: v for k, v in compute_metrics(y_val, val_preds).items()}
    all_metrics = {**train_metrics, **val_metrics}
    logger.info(f"Baseline metrics: {all_metrics}")
    wandb_log(run, all_metrics)

    # Save
    ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "baseline_model.pkl"
    model.save(str(ckpt_path))
    return model, all_metrics


# ============================================================
# Main entry point
# ============================================================

def train(cfg: dict, ticker: str = None):
    """Full training pipeline across all tickers in config (or a single override ticker).

    When `ticker` is supplied the run trains on that ticker only (useful for
    quick experiments). Otherwise every ticker in cfg['data']['tickers'] is
    processed and their data is concatenated before training.
    """
    set_seed(cfg["training"].get("seed", 42))
    device = detect_device()
    run = init_wandb(cfg, job_type="train")

    tickers = [ticker] if ticker else cfg["data"]["tickers"]
    model_type = cfg["model"]["type"].lower()
    use_sentiment = cfg["features"].get("use_sentiment", True)

    logger.info(f"Training on {len(tickers)} ticker(s): {tickers}")
    logger.info(f"Model type: {model_type}")

    # ── Build multi-ticker dataset ──
    train_df, val_df, test_df, feature_cols, target_cols = build_multi_ticker_splits(
        tickers, cfg, use_sentiment
    )

    # Fit scaler on combined training data so normalisation is consistent
    # across all tickers. The scaler only sees training rows — no look-ahead.
    scaler = fit_scaler(train_df, feature_cols)

    # Persist scaler for inference
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = ckpt_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
    logger.info(f"Scaler saved → {scaler_path}")

    input_size = len(feature_cols)
    model, _ = build_model(cfg, input_size=input_size)

    seq_len = cfg["data"]["sequence_length"]

    if model_type in ("xgboost", "lightgbm"):
        X_train, y_train = _flatten_for_baseline(train_df, feature_cols, target_cols, seq_len, scaler)
        X_val,   y_val   = _flatten_for_baseline(val_df,   feature_cols, target_cols, seq_len, scaler)
        X_test,  y_test  = _flatten_for_baseline(test_df,  feature_cols, target_cols, seq_len, scaler)

        model, metrics = train_baseline(model, X_train, y_train, X_val, y_val, cfg, run)
        test_preds = model.predict(X_test)
        test_metrics = {"test_" + k: v for k, v in compute_metrics(y_test, test_preds).items()}
        wandb_log(run, test_metrics)
        logger.info(f"Test metrics: {test_metrics}")

    else:
        # Deep-learning path
        batch_size = cfg["training"]["batch_size"]

        train_loader, val_loader, test_loader = make_dataloaders(
            train_df, val_df, test_df,
            feature_cols, target_cols,
            seq_len=seq_len,
            batch_size=batch_size,
            scaler=scaler,
        )

        model, history = train_dl_model(model, train_loader, val_loader, cfg, device, run)

        # Evaluate on test set
        import torch
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                # Apply sigmoid to the downside logit head before metric computation
                raw = model(xb.to(device))
                raw[:, 2] = torch.sigmoid(raw[:, 2])
                all_preds.append(raw.cpu().numpy())
                all_targets.append(yb.numpy())

        if all_preds:
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            test_metrics = {"test_" + k: v for k, v in compute_metrics(y_true, y_pred).items()}
            wandb_log(run, test_metrics)
            logger.info(f"Test metrics: {test_metrics}")

    finish_wandb(run)
    logger.info("Training complete.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Portfolio Analyzer model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--ticker", default=None,
        help="Train on a single ticker instead of all tickers in config"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, ticker=args.ticker)