"""
evaluate.py — Evaluation utilities for the AI Portfolio Analyzer.

Responsibilities:
- Load trained checkpoint (DL or baseline)
- Run predictions on test set
- Compute regression + financial metrics
- Generate prediction plots
- Log everything to W&B
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import load_config, detect_device, init_wandb, wandb_log, finish_wandb
from src.dataset import (
    build_processed_dataset,
    time_series_split,
    get_feature_cols,
    TARGET_COLS,
    StockSequenceDataset,
    make_dataloaders,
)
from src.model import build_model, BaselineModel

logger = logging.getLogger(__name__)


# ============================================================
# Metric helpers
# ============================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    prefix = f"{label}_" if label else ""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    ic   = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae":  mae,
        f"{prefix}r2":   r2,
        f"{prefix}ic":   ic,   # Information Coefficient — Pearson corr
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples where predicted sign matches true sign."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute metrics for all three output heads."""
    targets = ["return", "volatility", "downside"]
    metrics: Dict = {}
    for i, name in enumerate(targets):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        metrics.update(regression_metrics(yt, yp, label=name))
        if name == "return":
            metrics["directional_accuracy"] = directional_accuracy(yt, yp)
    return metrics


# ============================================================
# Load checkpoint helpers
# ============================================================

def load_dl_model(cfg: Dict, device: str):
    """Load best PyTorch checkpoint and return (model, scaler, feature_cols)."""
    import torch

    ckpt_dir    = Path(cfg["training"]["checkpoint_dir"])
    ckpt_path   = ckpt_dir / "best_model.pt"
    scaler_path = ckpt_dir / "scaler.pkl"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
    feature_cols = scaler_data["feature_cols"]
    scaler       = scaler_data["scaler"]

    input_size = len(feature_cols)
    model, _ = build_model(cfg, input_size=input_size)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    logger.info(f"DL model loaded from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")
    return model, scaler, feature_cols


def load_baseline_model(cfg: Dict):
    """Load pickled baseline model and return (model, scaler, feature_cols)."""
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    model = BaselineModel.load(str(ckpt_dir / "baseline_model.pkl"))

    scaler_path = ckpt_dir / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)

    return model, scaler_data["scaler"], scaler_data["feature_cols"]


# ============================================================
# Prediction runners
# ============================================================

def predict_dl(model, test_loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader; return (predictions, targets).

    BUG FIX: the model's third output is a raw logit (trained with
    BCEWithLogitsLoss).  Apply sigmoid here so callers always receive a
    proper probability in [0, 1] for the downside head.
    """
    import torch

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            raw = model(xb.to(device))
            # Convert downside logit → probability
            raw[:, 2] = torch.sigmoid(raw[:, 2])
            all_preds.append(raw.cpu().numpy())
            all_targets.append(yb.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def predict_baseline(model, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    preds = model.predict(X)
    return preds, y


# ============================================================
# Plot generation (Plotly → HTML saved to reports/figures)
# ============================================================

def plot_predictions(
    dates: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str,
    output_dir: str = "reports/figures",
) -> Optional[str]:
    """Generate an interactive HTML prediction chart and save it."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=["Return", "Volatility", "Downside Probability"],
            shared_xaxes=True,
            vertical_spacing=0.07,
        )
        labels = ["Return", "Volatility", "Downside Prob"]
        colors = ["#00d4aa", "#ff6b6b", "#ffd166"]

        for i, (label, color) in enumerate(zip(labels, colors)):
            row    = i + 1
            n      = len(dates)
            x_dates = dates[-n:]
            fig.add_trace(
                go.Scatter(x=x_dates, y=y_true[:, i], name=f"True {label}",
                           line=dict(color="#aaaaaa", width=1.5), opacity=0.8),
                row=row, col=1,
            )
            fig.add_trace(
                go.Scatter(x=x_dates, y=y_pred[:, i], name=f"Pred {label}",
                           line=dict(color=color, width=2)),
                row=row, col=1,
            )

        fig.update_layout(
            title=f"{ticker} — Model Predictions vs Actual",
            template="plotly_dark",
            height=700,
            showlegend=True,
        )

        out  = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = str(out / f"{ticker}_predictions.html")
        fig.write_html(path)
        logger.info(f"Prediction chart saved → {path}")
        return path
    except Exception as e:
        logger.warning(f"Could not generate prediction plot: {e}")
        return None


def plot_loss_curve(
    history: Dict,
    ticker: str,
    output_dir: str = "reports/figures",
) -> Optional[str]:
    """Plot training / validation loss curves."""
    if not history or "train_loss" not in history:
        return None
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history["train_loss"], name="Train Loss",
                                 line=dict(color="#00d4aa")))
        fig.add_trace(go.Scatter(y=history["val_loss"], name="Val Loss",
                                 line=dict(color="#ff6b6b")))
        fig.update_layout(
            title=f"{ticker} — Loss Curves",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark",
        )
        out  = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = str(out / f"{ticker}_loss_curve.html")
        fig.write_html(path)
        logger.info(f"Loss curve saved → {path}")
        return path
    except Exception as e:
        logger.warning(f"Could not generate loss curve: {e}")
        return None


# ============================================================
# Full evaluation pipeline
# ============================================================

def evaluate(cfg: Dict, ticker: str = None):
    """End-to-end evaluation of the trained model on the held-out test set."""
    device     = detect_device()
    run        = init_wandb(cfg, job_type="evaluate")
    model_type = cfg["model"]["type"].lower()
    use_sentiment = cfg["features"].get("use_sentiment", True)

    target_ticker = ticker or cfg["data"]["tickers"][0]
    logger.info(f"Evaluating: {target_ticker}  model: {model_type}")

    # Rebuild test split (same seed → identical dates)
    df = build_processed_dataset(target_ticker, cfg, sentiment_df=None)
    _, _, test_df = time_series_split(
        df,
        val_frac=cfg["training"]["val_split"],
        test_frac=cfg["training"]["test_split"],
    )

    seq_len = cfg["data"]["sequence_length"]

    if model_type in ("xgboost", "lightgbm"):
        model, scaler, feature_cols = load_baseline_model(cfg)
        feature_cols = [c for c in feature_cols if c in test_df.columns]
        target_cols  = [c for c in TARGET_COLS if c in test_df.columns]

        X_list, y_list = [], []
        Xarr = scaler.transform(test_df[feature_cols].values)
        yarr = test_df[target_cols].values
        for i in range(seq_len, len(Xarr)):
            # Flatten the full window (consistent with train.py's _flatten_for_baseline)
            X_list.append(Xarr[i - seq_len:i].flatten())
            y_list.append(yarr[i])
        X_test = np.array(X_list)
        y_test = np.array(y_list)

        y_pred, y_true = predict_baseline(model, X_test, y_test)
        test_dates = test_df.index[seq_len:]

    else:
        model, scaler, feature_cols = load_dl_model(cfg, device)
        feature_cols = [c for c in feature_cols if c in test_df.columns]
        target_cols  = [c for c in TARGET_COLS if c in test_df.columns]

        # BUG FIX: the original code constructed the test DataLoader by passing
        # val_df_dummy (a tiny 1% slice) as both train and val, which was a hack
        # to avoid needing to re-split.  Instead, build the test Dataset directly
        # and wrap it in a DataLoader — no need to construct train/val loaders at all.
        from torch.utils.data import DataLoader

        test_ds = StockSequenceDataset(
            test_df, feature_cols, target_cols, seq_len, scaler
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
        )

        y_pred, y_true = predict_dl(model, test_loader, device)
        test_dates = test_df.index[seq_len: seq_len + len(y_pred)]

    # Compute metrics
    metrics = all_metrics(y_true, y_pred)
    logger.info(f"Test metrics: {metrics}")
    wandb_log(run, {f"eval/{k}": v for k, v in metrics.items()})

    # Generate charts
    plot_predictions(test_dates, y_true, y_pred, target_ticker)

    finish_wandb(run)
    return metrics, y_pred, y_true


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, ticker=args.ticker)