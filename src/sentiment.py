"""
sentiment.py — News sentiment analysis for the AI Portfolio Analyzer.

Uses FinBERT (ProsusAI/finbert) from Hugging Face to score each article.
Falls back to a rule-based lexicon scorer when transformers is not available.

For each article produces:
    positive   float  0-1
    negative   float  0-1
    neutral    float  0-1
    score      float  -1 to +1   (positive - negative)
    label      str    bullish | bearish | neutral

Then aggregates to ticker-level features with recency weighting:
    average_sentiment      float
    weighted_sentiment     float   (exponential recency decay)
    positive_news_ratio    float
    negative_news_ratio    float
    neutral_news_ratio     float
    news_volume            int
    sentiment_volatility   float
    sentiment_momentum     float   (recent vs older articles)
    latest_news_sentiment  float
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FinBERT loader
# ---------------------------------------------------------------------------

_finbert_pipeline = None


def _load_finbert(model_name: str = "ProsusAI/finbert", device: int = -1):
    """Lazy-load the FinBERT pipeline.

    device: -1 = CPU, 0 = first CUDA GPU
    Falls back to lexicon scorer if transformers is not installed.
    """
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        logger.info(f"Loading sentiment model: {model_name} …")
        _finbert_pipeline = hf_pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,           # return all labels
            device=device,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded successfully.")
        return _finbert_pipeline
    except Exception as e:
        logger.warning(f"Could not load FinBERT ({e}). Using lexicon fallback.")
        return None


# ---------------------------------------------------------------------------
# Lexicon-based fallback scorer
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "beat", "beats", "surge", "surged", "gain", "gains", "profit", "profits",
    "growth", "grew", "rise", "rises", "rose", "record", "strong", "strength",
    "outperform", "upgrade", "raised", "higher", "bullish", "rally", "rallied",
    "positive", "exceeds", "exceeded", "expansion", "opportunity", "optimistic",
    "dividend", "recovery", "booming", "upside", "success", "achieve", "win",
    "breakthrough", "momentum", "accelerate", "robust", "solid",
}

NEGATIVE_WORDS = {
    "miss", "misses", "decline", "declined", "loss", "losses", "drop", "drops",
    "fell", "fall", "falls", "weak", "weakness", "downgrade", "cut", "lower",
    "bearish", "slump", "slumped", "disappointing", "disappoint", "concern",
    "risk", "uncertainty", "warning", "warned", "negative", "below", "shortfall",
    "layoff", "layoffs", "debt", "lawsuit", "investigation", "recall", "crisis",
    "recession", "inflation", "volatile", "volatility", "sell-off",
}


def _lexicon_score(text: str) -> Dict:
    """Simple word-count sentiment scorer (no ML)."""
    words = set(text.lower().split())
    pos   = len(words & POSITIVE_WORDS)
    neg   = len(words & NEGATIVE_WORDS)
    total = pos + neg + 1  # +1 avoids division by zero

    positive = pos / total
    negative = neg / total
    neutral  = max(0.0, 1.0 - positive - negative)

    # Normalise to sum to 1
    s = positive + negative + neutral
    positive, negative, neutral = positive / s, negative / s, neutral / s
    score = positive - negative

    if score > 0.1:
        label = "bullish"
    elif score < -0.1:
        label = "bearish"
    else:
        label = "neutral"

    return dict(positive=round(positive, 4), negative=round(negative, 4),
                neutral=round(neutral, 4), score=round(score, 4), label=label)


# ---------------------------------------------------------------------------
# Score a single article
# ---------------------------------------------------------------------------

def score_article(text: str, pipeline=None) -> Dict:
    """Return sentiment dict for a single text string.

    Uses FinBERT pipeline when available, otherwise lexicon fallback.
    """
    if not text or not text.strip():
        return dict(positive=0.33, negative=0.33, neutral=0.34, score=0.0, label="neutral")

    if pipeline is not None:
        try:
            results = pipeline(text[:512])
            # results is a list of dicts: [{"label": ..., "score": ...}, ...]
            # Flatten nested list if needed
            if results and isinstance(results[0], list):
                results = results[0]
            probs    = {r["label"].lower(): r["score"] for r in results}
            positive = probs.get("positive", 0.0)
            negative = probs.get("negative", 0.0)
            neutral  = probs.get("neutral",  1.0 - positive - negative)
            score    = positive - negative
            label    = "bullish" if score > 0.1 else ("bearish" if score < -0.1 else "neutral")
            return dict(positive=round(positive, 4), negative=round(negative, 4),
                        neutral=round(neutral, 4), score=round(score, 4), label=label)
        except Exception as e:
            logger.debug(f"FinBERT inference failed: {e} — using lexicon fallback.")

    return _lexicon_score(text)


# ---------------------------------------------------------------------------
# Score a list of articles
# ---------------------------------------------------------------------------

def score_articles(
    articles: List[Dict],
    cfg: Dict,
) -> List[Dict]:
    """Add sentiment fields to each article dict in-place and return the list.

    Text used for scoring = title + " " + summary.
    """
    scfg       = cfg.get("sentiment", {})
    model_name = scfg.get("model_name", "ProsusAI/finbert")
    batch_size = scfg.get("batch_size", 8)

    # Device: use CUDA if available
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except ImportError:
        device = -1

    pipeline = _load_finbert(model_name, device=device)

    texts = [f"{a['title']} {a['summary']}" for a in articles]

    if pipeline is not None:
        # Batch inference
        try:
            batch_results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                batch_results.extend(pipeline(batch, truncation=True, max_length=512))
            # batch_results is a list of lists when top_k=None
            for idx, res in enumerate(batch_results):
                if isinstance(res, list):
                    probs = {r["label"].lower(): r["score"] for r in res}
                else:
                    probs = {res["label"].lower(): res["score"]}
                pos = probs.get("positive", 0.0)
                neg = probs.get("negative", 0.0)
                neu = max(0.0, 1.0 - pos - neg)
                sc  = pos - neg
                lbl = "bullish" if sc > 0.1 else ("bearish" if sc < -0.1 else "neutral")
                articles[idx]["sentiment"] = dict(
                    positive=round(pos, 4), negative=round(neg, 4),
                    neutral=round(neu, 4), score=round(sc, 4), label=lbl,
                )
        except Exception as e:
            logger.warning(f"Batch FinBERT failed ({e}), scoring individually.")
            for idx, text in enumerate(texts):
                articles[idx]["sentiment"] = score_article(text, pipeline=None)
    else:
        for idx, text in enumerate(texts):
            articles[idx]["sentiment"] = score_article(text, pipeline=None)

    return articles


# ---------------------------------------------------------------------------
# Ticker-level sentiment aggregation
# ---------------------------------------------------------------------------

def aggregate_sentiment(articles: List[Dict], cfg: Dict) -> Dict:
    """Produce ticker-level sentiment features from a list of scored articles.

    Recency weighting: weight = exp(-lambda * age_hours)
    where lambda = cfg.sentiment.recency_decay (default 0.1).
    """
    scfg                = cfg.get("sentiment", {})
    decay               = scfg.get("recency_decay", 0.1)
    high_vol_threshold  = scfg.get("high_volume_threshold", 10)

    if not articles:
        return _empty_sentiment_features()

    # Only articles that have been scored
    scored = [a for a in articles if "sentiment" in a]
    if not scored:
        return _empty_sentiment_features()

    # BUG FIX: the original code assumed `scored` was already sorted newest-first
    # (articles[0] = newest), but articles arriving from mixed providers may not
    # be in that order.  Explicitly sort by age_hours ascending so that index 0
    # is always the most recent article — making the momentum and latest_score
    # calculations correct regardless of the caller's ordering.
    scored = sorted(scored, key=lambda a: a.get("age_hours", 0.0))

    scores = np.array([a["sentiment"]["score"] for a in scored])
    ages   = np.array([a.get("age_hours", 0.0) for a in scored])
    labels = [a["sentiment"]["label"] for a in scored]

    # Recency weights (exponential decay — smaller age_hours = newer = higher weight)
    weights  = np.exp(-decay * ages)
    weights /= weights.sum() + 1e-10

    weighted_sentiment = float(np.dot(weights, scores))
    average_sentiment  = float(np.mean(scores))

    n         = len(scored)
    pos_ratio = labels.count("bullish") / n
    neg_ratio = labels.count("bearish") / n
    neu_ratio = labels.count("neutral") / n

    sent_vol = float(np.std(scores))

    # Momentum: mean of newest half (index 0…mid) vs oldest half (index mid…n)
    # Now guaranteed correct because we sorted by age ascending above.
    mid         = max(1, n // 2)
    recent_mean = float(np.mean(scores[:mid]))
    older_mean  = float(np.mean(scores[mid:]) if n > 1 else scores[0])
    sentiment_momentum = recent_mean - older_mean

    # Latest score = the most recent article (index 0 after sorting)
    latest_score = float(scores[0]) if len(scores) > 0 else 0.0

    news_volume        = n
    volume_risk_factor = min(1.0, n / high_vol_threshold)

    return dict(
        average_sentiment=round(average_sentiment, 4),
        weighted_sentiment=round(weighted_sentiment, 4),
        positive_news_ratio=round(pos_ratio, 4),
        negative_news_ratio=round(neg_ratio, 4),
        neutral_news_ratio=round(neu_ratio, 4),
        news_volume=news_volume,
        sentiment_volatility=round(sent_vol, 4),
        sentiment_momentum=round(sentiment_momentum, 4),
        latest_news_sentiment=round(latest_score, 4),
        volume_risk_factor=round(volume_risk_factor, 4),
    )


def _empty_sentiment_features() -> Dict:
    return dict(
        average_sentiment=0.0,
        weighted_sentiment=0.0,
        positive_news_ratio=0.0,
        negative_news_ratio=0.0,
        neutral_news_ratio=1.0,
        news_volume=0,
        sentiment_volatility=0.0,
        sentiment_momentum=0.0,
        latest_news_sentiment=0.0,
        volume_risk_factor=0.0,
    )


# ---------------------------------------------------------------------------
# Sentiment-adjusted risk modifier
# ---------------------------------------------------------------------------

def sentiment_risk_adjustment(sentiment_features: Dict) -> Dict:
    """Compute additive/multiplicative adjustments to model risk outputs.

    Rules (simple, transparent, educational):
    - Very negative sentiment  → increase downside risk, increase volatility
    - Very positive sentiment  → slightly improve return forecast, raise vol
    - High sentiment volatility → raise uncertainty
    - High news volume         → raise uncertainty
    """
    ws  = sentiment_features.get("weighted_sentiment", 0.0)
    sv  = sentiment_features.get("sentiment_volatility", 0.0)
    vrf = sentiment_features.get("volume_risk_factor", 0.0)

    # Return boost: positive news mildly lifts expected return
    return_adj = ws * 0.02           # ±2% max at |ws|=1

    # Volatility penalty: negative news or high vol → raise vol forecast
    vol_adj = abs(ws) * 0.005 + sv * 0.01 + vrf * 0.005

    # Downside probability: negative sentiment and high volume raise it
    neg_ratio    = sentiment_features.get("negative_news_ratio", 0.0)
    downside_adj = max(0.0, -ws * 0.1 + neg_ratio * 0.05 + vrf * 0.03)

    # Overall uncertainty multiplier (for confidence interval width)
    uncertainty_mult = 1.0 + sv * 0.3 + vrf * 0.2

    return dict(
        return_adj=round(return_adj, 5),
        vol_adj=round(vol_adj, 5),
        downside_adj=round(downside_adj, 5),
        uncertainty_mult=round(uncertainty_mult, 4),
    )