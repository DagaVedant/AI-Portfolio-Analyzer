"""
model.py — Model definitions for the AI Portfolio Analyzer.

Supported architectures:
  - LSTM          (PyTorch) — sequential time-series model
  - Transformer   (PyTorch) — attention-based time-series model
  - XGBoost       (sklearn-compatible) — gradient-boosted tree baseline
  - LightGBM      (sklearn-compatible) — fast gradient-boosted tree baseline

All deep-learning models output three values per sample:
    [expected_return, expected_volatility, downside_probability]
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — deep-learning models unavailable.")

# ── Gradient boosting ────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


# ============================================================
# LSTM Model
# ============================================================

if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """Multi-layer LSTM with optional MC-Dropout for uncertainty estimation.

        Args:
            input_size:  number of input features per time step
            hidden_size: LSTM hidden dimension
            num_layers:  stacked LSTM layers
            output_size: number of prediction targets (default 3)
            dropout:     dropout probability applied between LSTM layers
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            output_size: int = 3,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden_size)
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_size // 2, output_size)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, input_size)
            out, _ = self.lstm(x)          # (batch, seq_len, hidden)
            out = out[:, -1, :]            # last time step
            out = self.norm(out)
            out = self.dropout(out)
            out = self.act(self.fc1(out))
            out = self.fc2(out)            # (batch, output_size)
            return out

        def predict_with_uncertainty(
            self,
            x: "torch.Tensor",
            n_samples: int = 50,
        ) -> tuple:
            """MC-Dropout uncertainty estimation.

            Runs n_samples forward passes with dropout enabled to estimate
            a predictive mean and standard deviation.
            """
            self.train()  # enable dropout
            preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
            self.eval()
            mean = preds.mean(dim=0)
            std = preds.std(dim=0)
            return mean, std


# ============================================================
# Transformer Model
# ============================================================

if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)


    class TransformerModel(nn.Module):
        """Encoder-only Transformer for time-series regression.

        Input projection maps raw features → d_model, then n_layers of
        multi-head attention + FFN, then mean pooling → prediction head.
        """

        def __init__(
            self,
            input_size: int,
            d_model: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            ffn_dim: int = 256,
            output_size: int = 3,
            dropout: float = 0.1,
            max_seq_len: int = 512,
        ):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,        # pre-LN for stable training
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.pool = nn.AdaptiveAvgPool1d(1)   # mean pooling over time
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_size),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, input_size)
            x = self.input_proj(x)          # → (batch, seq, d_model)
            x = self.pos_enc(x)
            x = self.encoder(x)             # → (batch, seq, d_model)
            x = self.pool(x.transpose(1, 2)).squeeze(-1)  # → (batch, d_model)
            return self.head(x)             # → (batch, output_size)

        def predict_with_uncertainty(
            self, x: "torch.Tensor", n_samples: int = 50
        ) -> tuple:
            self.train()
            preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
            self.eval()
            return preds.mean(dim=0), preds.std(dim=0)


# ============================================================
# Baseline: XGBoost / LightGBM wrapper
# ============================================================

class BaselineModel:
    """Thin wrapper around XGBoost or LightGBM for multi-output regression.

    Trains three separate regressors — one per target column.
    """

    def __init__(self, model_type: str = "xgboost", **kwargs):
        self.model_type = model_type.lower()
        self.kwargs = kwargs
        self.models = []   # one model per output
        self._is_fitted = False

    def _make_estimator(self):
        default_params = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0,
        )
        default_params.update(self.kwargs)

        if self.model_type == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("xgboost not installed.")
            return xgb.XGBRegressor(**default_params)
        elif self.model_type == "lightgbm":
            if not LGB_AVAILABLE:
                raise ImportError("lightgbm not installed.")
            lgb_params = {k: v for k, v in default_params.items() if k != "verbosity"}
            lgb_params["verbose"] = -1
            return lgb.LGBMRegressor(**lgb_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
    ):
        """Fit one regressor per output column."""
        n_outputs = y_train.shape[1] if y_train.ndim > 1 else 1
        self.models = []
        for i in range(n_outputs):
            y_i = y_train[:, i] if y_train.ndim > 1 else y_train
            m = self._make_estimator()
            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                y_vi = y_val[:, i] if y_val.ndim > 1 else y_val
                eval_set = [(X_val, y_vi)]
                if self.model_type == "xgboost":
                    fit_kwargs = dict(
                        eval_set=eval_set,
                        verbose=False,
                        early_stopping_rounds=early_stopping_rounds,
                    )
                elif self.model_type == "lightgbm":
                    fit_kwargs = dict(
                        eval_set=eval_set,
                        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                    )
            m.fit(X_train, y_i, **fit_kwargs)
            self.models.append(m)
        self._is_fitted = True
        logger.info(f"BaselineModel ({self.model_type}) trained on {X_train.shape}.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet.")
        preds = np.stack([m.predict(X) for m in self.models], axis=1)
        return preds  # (n_samples, n_outputs)

    def feature_importances(self) -> list:
        """Return per-output feature importance arrays."""
        return [m.feature_importances_ for m in self.models]

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"BaselineModel saved → {path}")

    @staticmethod
    def load(path: str) -> "BaselineModel":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"BaselineModel loaded ← {path}")
        return obj


# ============================================================
# Model Factory
# ============================================================

def build_model(cfg: dict, input_size: int):
    """Create and return a model based on config['model']['type'].

    Returns a (model, model_type_str) tuple.
    """
    mcfg = cfg.get("model", {})
    model_type = mcfg.get("type", "lstm").lower()

    if model_type in ("xgboost", "lightgbm"):
        bcfg = cfg.get("baseline", {})
        model = BaselineModel(
            model_type=model_type,
            n_estimators=bcfg.get("n_estimators", 500),
            max_depth=bcfg.get("max_depth", 6),
            learning_rate=bcfg.get("learning_rate", 0.05),
        )
        return model, model_type

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for lstm/transformer models.")

    output_size = mcfg.get("output_size", 3)
    dropout = mcfg.get("dropout", 0.2)

    if model_type == "lstm":
        model = LSTMModel(
            input_size=input_size,
            hidden_size=mcfg.get("hidden_size", 128),
            num_layers=mcfg.get("num_layers", 2),
            output_size=output_size,
            dropout=dropout,
        )
    elif model_type == "transformer":
        model = TransformerModel(
            input_size=input_size,
            d_model=mcfg.get("hidden_size", 128),
            num_heads=mcfg.get("attention_heads", 4),
            num_layers=mcfg.get("num_layers", 2),
            ffn_dim=mcfg.get("ffn_dim", 256),
            output_size=output_size,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Built {model_type.upper()} model | params={n_params:,} | input_size={input_size}")
    return model, model_type
