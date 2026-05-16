"""
dataset.py — Market data pipeline for the AI Portfolio Analyzer.

Responsibilities:
- Download OHLCV data with yfinance
- Compute technical indicators
- Merge sentiment features
- Build train / val / test splits (time-series safe)
- Expose PyTorch Dataset / DataLoader objects
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Try PyTorch — gracefully skip if not installed
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed; Dataset/DataLoader unavailable.")


# ============================================================
# 1. Data Download
# ============================================================

def download_ticker(
    ticker: str,
    start: str,
    end: str,
    raw_dir: str = "data/raw",
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker using yfinance.

    Returns a cleaned DataFrame indexed by Date.
    """
    raw_path = Path(raw_dir) / f"{ticker}.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        logger.info(f"Loaded cached data for {ticker} from {raw_path}")
        return df

    logger.info(f"Downloading {ticker} from {start} to {end} …")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.to_datetime(raw.index)
    raw.dropna(how="all", inplace=True)
    raw.to_parquet(raw_path)
    logger.info(f"Saved raw data → {raw_path}  ({len(raw)} rows)")
    return raw

def download_intraday(
    ticker: str,
    interval: str = "1h",
    period: str = "2y",
    raw_dir: str = "data/raw",
) -> pd.DataFrame:
    """Download intraday OHLCV bars via yfinance.

    interval options : 1m (7 days max), 5m, 15m, 30m, 1h (2 years max)
    period   options : 7d, 60d, 1y, 2y

    Saves to data/raw/{ticker}_{interval}.parquet
    """
    raw_path = Path(raw_dir) / f"{ticker}_{interval}.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        logger.info(f"Loaded cached intraday data for {ticker} ({interval}) from {raw_path}")
        return df

    logger.info(f"Downloading {ticker} intraday ({interval}, {period}) ...")
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.to_datetime(raw.index)
    raw.dropna(how="all", inplace=True)

    if raw.empty:
        raise ValueError(f"No intraday data returned for {ticker} ({interval})")

    raw.to_parquet(raw_path)
    logger.info(f"Saved intraday data → {raw_path}  ({len(raw)} rows)")
    return raw


def download_all_intraday(
    cfg: Dict,
    interval: str = "1h",
    period: str = "2y",
) -> Dict[str, pd.DataFrame]:
    """Download intraday data for every ticker in config."""
    Path(cfg["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    results = {}
    for ticker in cfg["data"]["tickers"]:
        try:
            df = download_intraday(
                ticker,
                interval=interval,
                period=period,
                raw_dir=cfg["data"]["raw_dir"],
            )
            results[ticker] = df
        except Exception as e:
            logger.warning(f"Failed intraday download for {ticker}: {e}")
    return results


def download_all(cfg: Dict) -> Dict[str, pd.DataFrame]:
    """Download data for every ticker in the config."""
    Path(cfg["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    results = {}
    for ticker in cfg["data"]["tickers"]:
        try:
            df = download_ticker(
                ticker,
                start=cfg["data"]["start_date"],
                end=cfg["data"]["end_date"],
                raw_dir=cfg["data"]["raw_dir"],
            )
            results[ticker] = df
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")
    return results


# ============================================================
# 2. Technical Indicators
# ============================================================

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_moving_averages(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
        df[f"vol_{w}"] = df["daily_return"].rolling(w).std()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    mid = df["Close"].rolling(period).mean()
    sigma = df["Close"].rolling(period).std()
    df["bb_upper"] = mid + std * sigma
    df["bb_lower"] = mid - std * sigma
    df["bb_mid"] = mid
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (mid + 1e-10)
    df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(period).mean()
    df["atr_pct"] = df["atr"] / (df["Close"] + 1e-10)
    return df


def add_momentum(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"mom_{w}"] = df["Close"].pct_change(w)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index)
    df["day_of_week"]    = idx.dayofweek / 4.0        # normalise 0-1
    df["day_of_month"]   = idx.day / 31.0
    df["month"]          = idx.month / 12.0
    df["quarter"]        = idx.quarter / 4.0
    df["is_month_end"]   = idx.is_month_end.astype(float)
    df["is_quarter_end"] = idx.is_quarter_end.astype(float)
    return df

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch and merge macro indicators via yfinance.

    Adds daily returns and normalised levels for:
    VIX, 10Y yield, 3M yield, Dollar index, Gold, Oil.
    Missing dates are forward-filled.
    """
    import yfinance as yf

    macro_tickers = {
        "^VIX":     "vix",
        "^TNX":     "yield_10y",
        "^IRX":     "yield_3m",
        "DX-Y.NYB": "dxy",
        "GC=F":     "gold",
        "CL=F":     "oil",
    }

    start = df.index[0]
    end   = df.index[-1]

    for yticker, col_name in macro_tickers.items():
        try:
            raw = yf.download(
                yticker, start=start, end=end,
                auto_adjust=True, progress=False
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if raw.empty:
                raise ValueError("empty")

            series = raw["Close"].reindex(df.index, method="ffill")
            df[col_name]              = series.pct_change()
            df[f"{col_name}_level"]   = series / (series.rolling(252).mean() + 1e-10)
        except Exception as e:
            logger.warning(f"Macro fetch failed for {yticker}: {e} — filling zeros.")
            df[col_name]            = 0.0
            df[f"{col_name}_level"] = 1.0

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_change"]   = df["Volume"].pct_change()
    df["vol_ratio_20"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-10)
    df["obv"]          = (np.sign(df["daily_return"]) * df["Volume"]).cumsum()
    return df


def build_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Apply all technical indicator functions to a raw OHLCV DataFrame."""
    fcfg = cfg.get("features", {})
    windows = fcfg.get("rolling_windows", [5, 10, 20, 50, 200])

    df = add_returns(df)
    df = add_moving_averages(df, windows)
    df = add_rsi(df, period=fcfg.get("rsi_period", 14))
    df = add_macd(
        df,
        fast=fcfg.get("macd_fast", 12),
        slow=fcfg.get("macd_slow", 26),
        signal=fcfg.get("macd_signal", 9),
    )
    df = add_bollinger_bands(
        df,
        period=fcfg.get("bb_period", 20),
        std=fcfg.get("bb_std", 2.0),
    )
    df = add_atr(df, period=fcfg.get("atr_period", 14))
    df = add_momentum(df, windows=[5, 10, 20])
    df = add_time_features(df)
    df = add_volume_features(df)
    df = add_macro_features(df)

    # ── Target columns (forward-looking) ─────────────────────────────────
    # BUG FIX: the original code used df["log_return"].shift(-horizon).rolling(horizon)
    # which double-shifts — the rolling window then operates on already-future values,
    # spanning rows t+horizon to t+2*horizon instead of t+1 to t+horizon.
    #
    # Correct approach: sum/std/min the NEXT `horizon` log returns by rolling
    # forward and then aligning back to the current row with a single shift.
    #
    #   future_returns[t] = log_return[t+1] + ... + log_return[t+horizon]
    #
    # Implemented as: reverse the series, rolling sum of `horizon` rows,
    # reverse back — which is equivalent to a trailing sum on the reversed
    # series and therefore a leading sum on the original.
    horizon = cfg["data"]["forecast_horizon"]
    downside_threshold = -0.05

    # Rolling sum/std/min of the NEXT `horizon` daily log returns
    lr = df["log_return"]
    df["target_return"]     = lr[::-1].rolling(horizon).sum().shift(-(horizon - 1))[::-1]
    df["target_volatility"] = lr[::-1].rolling(horizon).std().shift(-(horizon - 1))[::-1]
    df["target_downside"]   = (
        (lr[::-1].rolling(horizon).min().shift(-(horizon - 1))[::-1] < downside_threshold)
        .astype(float)
    )

    return df


def merge_sentiment(df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Left-join ticker-level sentiment features onto the price DataFrame.

    sentiment_df must have a DatetimeIndex and numeric sentiment columns.
    Missing dates are forward-filled (sentiment persists until refreshed).
    """
    if sentiment_df is None or sentiment_df.empty:
        logger.info("No sentiment data to merge; using zeros.")
        sent_cols = [
            "weighted_sentiment", "positive_news_ratio",
            "negative_news_ratio", "neutral_news_ratio",
            "news_volume", "sentiment_volatility",
            "sentiment_momentum", "latest_news_sentiment",
        ]
        for col in sent_cols:
            df[col] = 0.0
        return df

    # Resample sentiment to daily, ffill
    sentiment_df = sentiment_df.reindex(df.index, method="ffill").fillna(0)
    df = df.join(sentiment_df, how="left")
    df[sentiment_df.columns] = df[sentiment_df.columns].fillna(0)
    return df


# ============================================================
# 3. Feature Columns
# ============================================================

PRICE_FEATURES = [
    "daily_return", "log_return", "Volume",
    "vol_change", "vol_ratio_20", "obv",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct",
    "atr_pct",
    "mom_5", "mom_10", "mom_20",
    "day_of_week", "day_of_month", "month", "quarter",
    "is_month_end", "is_quarter_end",
    # Macro features
    "vix", "vix_level",
    "yield_10y", "yield_10y_level",
    "yield_3m", "yield_3m_level",
    "dxy", "dxy_level",
    "gold", "gold_level",
    "oil", "oil_level",
]


WINDOW_FEATURES = []
for w in [5, 10, 20, 50, 200]:
    WINDOW_FEATURES += [f"sma_{w}", f"ema_{w}", f"vol_{w}"]

SENTIMENT_FEATURES = [
    "weighted_sentiment", "positive_news_ratio",
    "negative_news_ratio", "neutral_news_ratio",
    "news_volume", "sentiment_volatility",
    "sentiment_momentum", "latest_news_sentiment",
]

TARGET_COLS = ["target_return", "target_volatility", "target_downside"]


def get_feature_cols(use_sentiment: bool = True) -> List[str]:
    cols = PRICE_FEATURES + WINDOW_FEATURES
    if use_sentiment:
        cols += SENTIMENT_FEATURES
    return cols


# ============================================================
# 4. Processed Dataset
# ============================================================

def build_processed_dataset(
    ticker: str,
    cfg: Dict,
    sentiment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Full pipeline: download → features → merge sentiment → clean → save."""
    raw = download_ticker(
        ticker,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        raw_dir=cfg["data"]["raw_dir"],
    )
    df = build_features(raw, cfg)
    if cfg["features"].get("use_sentiment", True):
        df = merge_sentiment(df, sentiment_df)

    feature_cols = get_feature_cols(use_sentiment=cfg["features"].get("use_sentiment", True))
    keep_cols = feature_cols + TARGET_COLS
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Convert inf/-inf to NaN, then drop all missing values
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()

    out_path = Path(cfg["data"]["processed_dir"]) / f"{ticker}_processed.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info(f"Saved processed features → {out_path}  ({len(df)} rows, {len(keep_cols)} cols)")
    return df


# ============================================================
# 5. Train / Val / Test Splits
# ============================================================

def time_series_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split — NO shuffling, NO look-ahead."""
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start  = int(n * (1 - test_frac - val_frac))
    train = df.iloc[:val_start]
    val   = df.iloc[val_start:test_start]
    test  = df.iloc[test_start:]
    logger.info(
        f"Split: train={len(train)} val={len(val)} test={len(test)}"
    )
    return train, val, test


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def scale_df(df: pd.DataFrame, feature_cols: List[str], scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[feature_cols].values)


# ============================================================
# 6. PyTorch Dataset
# ============================================================

if TORCH_AVAILABLE:
    class StockSequenceDataset(Dataset):
        """Sliding-window sequence dataset for time-series models (LSTM / Transformer).

        Each sample is a window of shape (seq_len, n_features) with three targets:
            [expected_return, expected_volatility, downside_probability]
        """

        def __init__(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            target_cols: List[str],
            seq_len: int,
            scaler: Optional[StandardScaler] = None,
        ):
            self.seq_len = seq_len
            self.feature_cols = feature_cols
            self.target_cols = target_cols

            X = df[feature_cols].values.astype(np.float32)
            y = df[target_cols].values.astype(np.float32)

            if scaler is not None:
                X = scaler.transform(X).astype(np.float32)

            self.X = X
            self.y = y

        def __len__(self) -> int:
            # BUG FIX: original returned len(X) - seq_len which is correct for
            # valid indices 0 .. len(X)-seq_len-1, but we add an explicit max(0,…)
            # guard so an undersized split never returns a negative length.
            return max(0, len(self.X) - self.seq_len)

        def __getitem__(self, idx: int):
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            x_seq = self.X[idx: idx + self.seq_len]       # (seq_len, n_features)
            y_val = self.y[idx + self.seq_len]             # (3,)
            return torch.tensor(x_seq), torch.tensor(y_val)


    def make_dataloaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        seq_len: int,
        batch_size: int,
        scaler: Optional[StandardScaler] = None,
    ) -> Tuple["DataLoader", "DataLoader", "DataLoader"]:
        """Create train / val / test DataLoaders."""
        train_ds = StockSequenceDataset(train_df, feature_cols, target_cols, seq_len, scaler)
        val_ds   = StockSequenceDataset(val_df,   feature_cols, target_cols, seq_len, scaler)
        test_ds  = StockSequenceDataset(test_df,  feature_cols, target_cols, seq_len, scaler)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        logger.info(
            f"DataLoaders: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches"
        )
        return train_loader, val_loader, test_loader