"""
inference.py — Live inference pipeline for the AI Portfolio Analyzer.

For a given ticker this module:
  1. Fetches recent market data (yfinance)
  2. Engineers all technical features
  3. Fetches current news articles
  4. Scores news sentiment with FinBERT (or lexicon fallback)
  5. Builds the latest feature vector
  6. Runs the trained model (DL or baseline)
  7. Applies sentiment-based risk adjustments
  8. Computes VaR, CVaR, max drawdown estimate, beta, risk score
  9. Runs portfolio optimisation against a peer set
 10. Returns a fully-typed results dict ready for the dashboard

Disclaimer: All outputs are educational estimates, not financial advice.
"""

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import load_config, detect_device
from src.dataset import build_features, get_feature_cols, TARGET_COLS
from src.news import fetch_news
from src.sentiment import score_articles, aggregate_sentiment, sentiment_risk_adjustment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------

RISK_BANDS = [
    (0,   25,  "Low",     "#00d4aa"),
    (25,  50,  "Medium",  "#ffd166"),
    (50,  75,  "High",    "#ff9f43"),
    (75,  101, "Extreme", "#ff6b6b"),
]


def _risk_category(score: float) -> tuple:
    for lo, hi, label, color in RISK_BANDS:
        if lo <= score < hi:
            return label, color
    return "Extreme", "#ff6b6b"


def compute_var_cvar(
    daily_returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 21,
) -> Dict:
    """Historical Value-at-Risk and Conditional VaR.

    Returns daily and horizon-scaled figures (assuming iid returns).
    """
    r = daily_returns.dropna().values
    if len(r) < 20:
        return dict(var_1d=0.0, cvar_1d=0.0, var_h=0.0, cvar_h=0.0)

    var_1d = float(np.percentile(r, (1 - confidence) * 100))
    tail   = r[r <= var_1d]
    cvar_1d = float(tail.mean()) if len(tail) > 0 else var_1d

    # Scale to horizon (sqrt-of-time approximation)
    scale  = np.sqrt(horizon_days)
    var_h  = var_1d  * scale
    cvar_h = cvar_1d * scale

    return dict(
        var_1d=round(var_1d,  5),
        cvar_1d=round(cvar_1d, 5),
        var_h=round(var_h,   5),
        cvar_h=round(cvar_h,  5),
    )


def compute_max_drawdown(prices: pd.Series) -> float:
    """Historical maximum drawdown from a price series."""
    roll_max = prices.cummax()
    dd = (prices - roll_max) / (roll_max + 1e-10)
    return float(dd.min())


def compute_beta(ticker_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Rolling 252-day beta vs benchmark."""
    aligned = pd.concat([ticker_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 20:
        return 1.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / (cov[1, 1] + 1e-10))


def composite_risk_score(
    ann_vol: float,
    pred_vol: float,
    var_1d: float,
    max_dd: float,
    beta: float,
    downside_prob: float,
    sentiment_adj: Dict,
) -> float:
    """Produce a 0-100 composite risk score from multiple risk factors.

    Higher = riskier.
    """
    vol_score  = min(1.0, ann_vol / 0.6)
    var_score  = min(1.0, abs(var_1d) / 0.05)
    dd_score   = min(1.0, abs(max_dd) / 0.5)
    beta_score = min(1.0, abs(beta - 1) / 1.5)
    dn_score   = min(1.0, downside_prob)

    sent_vol_factor = min(1.0, sentiment_adj.get("vol_adj", 0) * 10)
    dn_sent_factor  = min(1.0, sentiment_adj.get("downside_adj", 0) * 5)

    raw = (
        vol_score        * 0.25 +
        var_score        * 0.20 +
        dd_score         * 0.20 +
        dn_score         * 0.15 +
        beta_score       * 0.10 +
        sent_vol_factor  * 0.05 +
        dn_sent_factor   * 0.05
    )
    return round(raw * 100, 1)


# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_model_cache: Dict = {}


def _load_model(cfg: Dict):
    """Lazy-load and cache the trained model + scaler."""
    global _model_cache
    model_type = cfg["model"]["type"].lower()
    ckpt_dir   = Path(cfg["training"]["checkpoint_dir"])
    scaler_path = ckpt_dir / "scaler.pkl"

    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["scaler"], _model_cache["feature_cols"]

    if not scaler_path.exists():
        logger.warning("No trained checkpoint found. Returning None model.")
        return None, None, None

    with open(scaler_path, "rb") as f:
        sd = pickle.load(f)
    scaler       = sd["scaler"]
    feature_cols = sd["feature_cols"]

    if model_type in ("xgboost", "lightgbm"):
        from src.model import BaselineModel
        model_path = ckpt_dir / "baseline_model.pkl"
        if not model_path.exists():
            return None, scaler, feature_cols
        model = BaselineModel.load(str(model_path))
    else:
        try:
            import torch
            from src.model import build_model
            model_path = ckpt_dir / "best_model.pt"
            if not model_path.exists():
                return None, scaler, feature_cols
            model, _ = build_model(cfg, input_size=len(feature_cols))
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
        except Exception as e:
            logger.warning(f"Could not load DL model: {e}")
            return None, scaler, feature_cols

    _model_cache = {"model": model, "scaler": scaler, "feature_cols": feature_cols}
    return model, scaler, feature_cols


# ---------------------------------------------------------------------------
# Feature vector builder for live inference
# ---------------------------------------------------------------------------

def _build_live_features(
    ticker: str,
    cfg: Dict,
    sentiment_features: Dict,
    lookback_days: int = 300,
) -> Optional[pd.DataFrame]:
    """Download recent market data, build features, and append sentiment."""
    end       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start     = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    start_str = start.strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            ticker,
            period=f"{lookback_days}d",
            auto_adjust=True,
            progress=False
            )
        
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index)
        if raw.empty:
            return None
    except Exception as e:
        logger.error(f"yfinance download failed for {ticker}: {e}")
        return None

    df = build_features(raw, cfg)

    # Attach sentiment columns
    for k, v in sentiment_features.items():
        df[k] = float(v)

    # Drop target cols (forward-looking; not available at inference time)
    for col in TARGET_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Core inference runner
# ---------------------------------------------------------------------------

def run_inference(ticker: str, cfg: Dict) -> Dict:
    """Full live inference pipeline.

    Returns a results dict consumed directly by the dashboard.
    """
    ticker  = ticker.upper().strip()
    horizon = cfg["data"]["forecast_horizon"]
    use_sentiment = cfg["features"].get("use_sentiment", True)

    # ── 1. Fetch news ──────────────────────────────────────────────────────
    logger.info(f"[{ticker}] Fetching news …")
    articles = fetch_news(ticker, cfg)

    # ── 2. Score sentiment ─────────────────────────────────────────────────
    logger.info(f"[{ticker}] Scoring sentiment ({len(articles)} articles) …")
    if articles:
        articles = score_articles(articles, cfg)

    sentiment_features = aggregate_sentiment(articles, cfg) if use_sentiment else {}
    sent_adj = sentiment_risk_adjustment(sentiment_features)

    # ── 3. Build live features ─────────────────────────────────────────────
    logger.info(f"[{ticker}] Building features …")
    df = _build_live_features(ticker, cfg, sentiment_features)
    if df is None or df.empty:
        return _error_result(ticker, "Could not fetch market data.")

    # daily_return lives in the feature DataFrame (it IS a feature col)
    daily_ret = df["daily_return"].dropna() if "daily_return" in df.columns else pd.Series(dtype=float)

    # ── 4. Market summary stats ────────────────────────────────────────────
    # BUG FIX: initialise current_price to 0 unconditionally; the try block
    # below always overwrites it from a dedicated price download.  The original
    # code had an inverted condition that always produced 0.0 anyway, and also
    # referenced df["Close"] which is not in the features DataFrame.
    current_price = 0.0
    ret_1d = ret_30d = ret_90d = 0.0
    volume = 0
    sma_20 = sma_50 = 0.0
    raw_for_price = None   # declared here so the max_dd block below is safe

    try:
        raw_for_price = yf.download(ticker, period="90d", auto_adjust=True, progress=False)
        if isinstance(raw_for_price.columns, pd.MultiIndex):
            raw_for_price.columns = raw_for_price.columns.get_level_values(0)
        current_price = float(raw_for_price["Close"].iloc[-1])
        ret_1d  = float(raw_for_price["Close"].pct_change().iloc[-1])
        ret_30d = float(raw_for_price["Close"].pct_change(21).iloc[-1])
        ret_90d = float(raw_for_price["Close"].pct_change(63).iloc[-1])
        volume  = int(raw_for_price["Volume"].iloc[-1])
        sma_20  = float(raw_for_price["Close"].rolling(20).mean().iloc[-1])
        sma_50  = float(raw_for_price["Close"].rolling(50).mean().iloc[-1])
    except Exception as e:
        logger.warning(f"Could not fetch price summary for {ticker}: {e}")

    # ── 5. Risk calculations ───────────────────────────────────────────────
    ann_vol  = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 5 else 0.1
    var_cvar = compute_var_cvar(daily_ret, confidence=0.95, horizon_days=horizon)

    # Benchmark beta
    try:
        bm_raw = yf.download("SPY", period="1y", auto_adjust=True, progress=False)
        if isinstance(bm_raw.columns, pd.MultiIndex):
            bm_raw.columns = bm_raw.columns.get_level_values(0)
        bm_ret = bm_raw["Close"].pct_change().dropna()
        beta   = compute_beta(daily_ret, bm_ret)
    except Exception:
        beta = 1.0

    # BUG FIX: raw_for_price is now guaranteed to be defined (either a
    # DataFrame from the try block above, or None).  Check before using.
    try:
        if raw_for_price is not None and not raw_for_price.empty:
            max_dd = compute_max_drawdown(raw_for_price["Close"])
        else:
            max_dd = 0.0
    except Exception:
        max_dd = 0.0

    # ── 6. ML model inference ──────────────────────────────────────────────
    model, scaler, feature_cols = _load_model(cfg)
    pred_return = pred_vol = downside_prob = confidence = 0.0
    model_used = "None (no checkpoint)"

    if model is not None and scaler is not None and feature_cols is not None:
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) < len(feature_cols) * 0.7:
            logger.warning("Too many missing feature columns — skipping model inference.")
        else:
            try:
                seq_len    = cfg["data"]["sequence_length"]
                model_type = cfg["model"]["type"].lower()

                # Pad missing cols with 0
                for col in feature_cols:
                    if col not in df.columns:
                        df[col] = 0.0

                X = scaler.transform(df[feature_cols].values[-seq_len:])

                if model_type in ("xgboost", "lightgbm"):
                    X_flat = X.flatten().reshape(1, -1)
                    preds  = model.predict(X_flat)[0]
                else:
                    import torch
                    x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
                    device   = detect_device()
                    model    = model.to(device)
                    with torch.no_grad():
                        if hasattr(model, "predict_with_uncertainty"):
                            mean, std = model.predict_with_uncertainty(x_tensor.to(device))
                            raw_preds = mean.cpu().numpy()[0]
                            conf_std  = std.cpu().numpy()[0]
                            confidence = float(np.clip(1.0 - conf_std.mean() * 5, 0, 1))
                        else:
                            raw_preds = model(x_tensor.to(device)).cpu().numpy()[0]

                    # BUG FIX: output[2] is a raw logit (BCEWithLogitsLoss was used
                    # during training).  Apply sigmoid here to convert it to a
                    # probability in [0, 1] before any downstream use.
                    preds = raw_preds.copy()
                    preds[2] = float(torch.sigmoid(torch.tensor(raw_preds[2])).item())

                pred_return   = float(preds[0]) if len(preds) > 0 else 0.0
                pred_vol      = float(preds[1]) if len(preds) > 1 else ann_vol
                downside_prob = float(np.clip(preds[2], 0, 1)) if len(preds) > 2 else 0.3
                confidence    = confidence or float(np.clip(1.0 - abs(pred_return) * 3, 0.2, 0.95))
                model_used    = model_type.upper()

            except Exception as e:
                logger.warning(f"Model inference failed: {e}")

    # ── 7. Sentiment-adjusted predictions ─────────────────────────────────
    adj_return   = pred_return + sent_adj.get("return_adj", 0.0)
    adj_vol      = max(0.001, pred_vol + sent_adj.get("vol_adj", 0.0))
    adj_downside = min(1.0, downside_prob + sent_adj.get("downside_adj", 0.0))

    # Forecast price range (±1.96σ over horizon)
    daily_vol_est = adj_vol / np.sqrt(252)
    horizon_vol   = daily_vol_est * np.sqrt(horizon)
    price_low  = current_price * np.exp(adj_return - 1.96 * horizon_vol)
    price_high = current_price * np.exp(adj_return + 1.96 * horizon_vol)
    price_mid  = current_price * np.exp(adj_return)

    # ── 8. Composite risk score ────────────────────────────────────────────
    risk_score = composite_risk_score(
        ann_vol, adj_vol,
        var_cvar["var_1d"],
        max_dd, beta,
        adj_downside,
        sent_adj,
    )
    risk_label, risk_color = _risk_category(risk_score)

    # Outlook label
    if adj_return > 0.02 and sentiment_features.get("weighted_sentiment", 0) > 0.1:
        outlook = "Bullish"
    elif adj_return < -0.02 or sentiment_features.get("weighted_sentiment", 0) < -0.1:
        outlook = "Bearish"
    else:
        outlook = "Neutral"

    # ── 9. Portfolio suggestion (single ticker vs peers) ──────────────────
    portfolio_result = _single_ticker_portfolio(ticker, cfg, pred_return)

    # ── 10. Assemble result dict ───────────────────────────────────────────
    result = dict(
        # Identity
        ticker=ticker,
        timestamp=datetime.now(timezone.utc).isoformat(),
        # Market summary
        current_price=round(current_price, 2),
        daily_return=round(ret_1d, 4),
        return_30d=round(ret_30d, 4),
        return_90d=round(ret_90d, 4),
        volume=volume,
        sma_20=round(sma_20, 2),
        sma_50=round(sma_50, 2),
        # Forecast
        pred_return=round(adj_return, 4),
        pred_volatility=round(adj_vol, 4),
        pred_downside_prob=round(adj_downside, 4),
        price_forecast_low=round(price_low, 2),
        price_forecast_mid=round(price_mid, 2),
        price_forecast_high=round(price_high, 2),
        forecast_horizon_days=horizon,
        model_confidence=round(confidence, 3),
        model_used=model_used,
        outlook=outlook,
        # Risk metrics
        ann_volatility=round(ann_vol, 4),
        var_1d=round(var_cvar["var_1d"], 4),
        cvar_1d=round(var_cvar["cvar_1d"], 4),
        var_horizon=round(var_cvar["var_h"], 4),
        cvar_horizon=round(var_cvar["cvar_h"], 4),
        max_drawdown=round(max_dd, 4),
        beta=round(beta, 3),
        risk_score=risk_score,
        risk_label=risk_label,
        risk_color=risk_color,
        # Sentiment
        sentiment_features=sentiment_features,
        sentiment_adjustment=sent_adj,
        articles=articles[:20],   # top 20 articles for display
        # Portfolio
        portfolio=portfolio_result,
    )
    return result


def _single_ticker_portfolio(ticker: str, cfg: Dict, pred_return: float) -> Dict:
    """Quick portfolio optimisation for a peer set including the queried ticker."""
    from src.optimize import optimize_portfolio

    peers = [ticker] + [
        t for t in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        if t != ticker
    ][:4]

    try:
        end   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (pd.Timestamp.now() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        raw   = yf.download(peers, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw
        close.dropna(how="all", inplace=True)
        close.dropna(axis=1, inplace=True)

        result = optimize_portfolio(
            close,
            cfg,
            model_preds={ticker: pred_return},
        )
        return result
    except Exception as e:
        logger.warning(f"Portfolio optimization failed: {e}")
        return dict(
            tickers=peers,
            weights=[1 / len(peers)] * len(peers),
            weight_dict={t: round(1 / len(peers), 3) for t in peers},
            expected_return=pred_return,
            expected_volatility=0.15,
            sharpe_ratio=0.5,
            risk_contributions={},
            method="equal_weight_fallback",
        )


def _error_result(ticker: str, message: str) -> Dict:
    logger.error(f"[{ticker}] {message}")
    return dict(
        ticker=ticker,
        error=message,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )