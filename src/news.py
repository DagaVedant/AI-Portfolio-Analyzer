"""
news.py — Financial news pipeline for the AI Portfolio Analyzer.

Supports (in priority order, based on available API keys):
  1. NewsAPI       — broad financial news (requires NEWSAPI_KEY)
  2. Finnhub       — company-specific news (requires FINNHUB_KEY)
  3. Alpha Vantage — ticker news sentiment feed (requires AV_KEY)
  4. Yahoo Finance RSS — free, no key required (always available)
  5. GDELT         — free global event database (no key required)

All providers return a unified list of article dicts with these keys:
    title       str
    summary     str
    source      str
    published   datetime (UTC, timezone-aware)
    url         str
    ticker      str
    age_hours   float
"""

import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YAHOO_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GDELT_TEMPLATE = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query={query}&mode=artlist&maxrecords=25&format=json&timespan=3days"
)
REQUEST_TIMEOUT = 10   # seconds


# ---------------------------------------------------------------------------
# Unified article schema
# ---------------------------------------------------------------------------

def _article(
    title: str,
    summary: str,
    source: str,
    published: datetime,
    url: str,
    ticker: str,
) -> Dict:
    """Return a clean, normalised article dict."""
    now = datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_hours = (now - published).total_seconds() / 3600
    return dict(
        title=title.strip(),
        summary=(summary or "").strip(),
        source=source.strip(),
        published=published,
        url=url.strip(),
        ticker=ticker.upper(),
        age_hours=round(age_hours, 2),
    )


def _parse_rss_date(date_str: str) -> datetime:
    """Parse RFC-822 date strings from RSS feeds."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Provider: Yahoo Finance RSS (FREE — always available)
# ---------------------------------------------------------------------------

def fetch_yahoo_rss(ticker: str, max_articles: int = 30) -> List[Dict]:
    """Fetch news from Yahoo Finance RSS feed. No API key required."""
    url = YAHOO_RSS_TEMPLATE.format(ticker=ticker)
    articles = []
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        for item in root.iter("item"):
            title_el = item.find("title")
            desc_el = item.find("description")
            pub_el = item.find("pubDate")
            link_el = item.find("link")
            source_el = item.find("source")

            title = title_el.text if title_el is not None else ""
            summary = desc_el.text if desc_el is not None else ""
            pub_str = pub_el.text if pub_el is not None else ""
            link = link_el.text if link_el is not None else ""
            source = source_el.text if source_el is not None else "Yahoo Finance"

            published = _parse_rss_date(pub_str)
            articles.append(_article(title, summary, source, published, link, ticker))

            if len(articles) >= max_articles:
                break

        logger.info(f"Yahoo RSS: {len(articles)} articles for {ticker}")
    except Exception as e:
        logger.warning(f"Yahoo RSS failed for {ticker}: {e}")

    return articles


# ---------------------------------------------------------------------------
# Provider: NewsAPI
# ---------------------------------------------------------------------------

def fetch_newsapi(
    ticker: str,
    api_key: str,
    lookback_hours: int = 72,
    max_articles: int = 30,
    base_url: str = "https://newsapi.org/v2/everything",
) -> List[Dict]:
    """Fetch news from NewsAPI.org. Requires NEWSAPI_KEY."""
    from_dt = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "q": ticker,
        "from": from_dt,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": api_key,
    }
    articles = []
    try:
        resp = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        for art in data.get("articles", []):
            pub_str = art.get("publishedAt", "")
            published = _parse_rss_date(pub_str) if pub_str else datetime.now(timezone.utc)
            articles.append(_article(
                title=art.get("title", ""),
                summary=art.get("description", "") or art.get("content", ""),
                source=art.get("source", {}).get("name", "NewsAPI"),
                published=published,
                url=art.get("url", ""),
                ticker=ticker,
            ))
        logger.info(f"NewsAPI: {len(articles)} articles for {ticker}")
    except Exception as e:
        logger.warning(f"NewsAPI failed for {ticker}: {e}")
    return articles


# ---------------------------------------------------------------------------
# Provider: Finnhub
# ---------------------------------------------------------------------------

def fetch_finnhub(
    ticker: str,
    api_key: str,
    lookback_days: int = 3,
    max_articles: int = 30,
    base_url: str = "https://finnhub.io/api/v1/company-news",
) -> List[Dict]:
    """Fetch company news from Finnhub. Requires FINNHUB_KEY."""
    today = datetime.now(timezone.utc)
    from_dt = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_dt = today.strftime("%Y-%m-%d")
    params = {
        "symbol": ticker,
        "from": from_dt,
        "to": to_dt,
        "token": api_key,
    }
    articles = []
    try:
        resp = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        for art in data[:max_articles]:
            pub_ts = art.get("datetime", 0)
            published = datetime.fromtimestamp(pub_ts, tz=timezone.utc) if pub_ts else datetime.now(timezone.utc)
            articles.append(_article(
                title=art.get("headline", ""),
                summary=art.get("summary", ""),
                source=art.get("source", "Finnhub"),
                published=published,
                url=art.get("url", ""),
                ticker=ticker,
            ))
        logger.info(f"Finnhub: {len(articles)} articles for {ticker}")
    except Exception as e:
        logger.warning(f"Finnhub failed for {ticker}: {e}")
    return articles


# ---------------------------------------------------------------------------
# Provider: Alpha Vantage News Sentiment
# ---------------------------------------------------------------------------

def fetch_alphavantage(
    ticker: str,
    api_key: str,
    max_articles: int = 30,
    base_url: str = "https://www.alphavantage.co/query",
) -> List[Dict]:
    """Fetch news from Alpha Vantage News Sentiment endpoint."""
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": max_articles,
        "apikey": api_key,
    }
    articles = []
    try:
        resp = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        for art in data.get("feed", []):
            pub_str = art.get("time_published", "")
            try:
                published = datetime.strptime(pub_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                published = datetime.now(timezone.utc)

            articles.append(_article(
                title=art.get("title", ""),
                summary=art.get("summary", ""),
                source=art.get("source", "Alpha Vantage"),
                published=published,
                url=art.get("url", ""),
                ticker=ticker,
            ))
        logger.info(f"Alpha Vantage: {len(articles)} articles for {ticker}")
    except Exception as e:
        logger.warning(f"Alpha Vantage failed for {ticker}: {e}")
    return articles


# ---------------------------------------------------------------------------
# Provider: GDELT (FREE — no key)
# ---------------------------------------------------------------------------

def fetch_gdelt(ticker: str, company_name: str = "", max_articles: int = 25) -> List[Dict]:
    """Fetch news from GDELT Project API. Completely free."""
    query = f'"{company_name or ticker}" finance stock'
    url = GDELT_TEMPLATE.format(query=quote_plus(query))
    articles = []
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT * 2)
        resp.raise_for_status()
        data = resp.json()

        for art in (data.get("articles") or [])[:max_articles]:
            pub_str = art.get("seendate", "")
            try:
                published = datetime.strptime(pub_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                published = datetime.now(timezone.utc)

            articles.append(_article(
                title=art.get("title", ""),
                summary=art.get("title", ""),  # GDELT doesn't provide summaries
                source=art.get("domain", "GDELT"),
                published=published,
                url=art.get("url", ""),
                ticker=ticker,
            ))
        logger.info(f"GDELT: {len(articles)} articles for {ticker}")
    except Exception as e:
        logger.warning(f"GDELT failed for {ticker}: {e}")
    return articles


# ---------------------------------------------------------------------------
# Main public function — auto-selects provider
# ---------------------------------------------------------------------------

# Map common tickers to company names (for GDELT queries)
TICKER_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
    "TSLA": "Tesla", "AMZN": "Amazon", "META": "Meta",
    "GOOGL": "Google Alphabet", "AMD": "AMD", "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq ETF",
}


def fetch_news(
    ticker: str,
    cfg: Dict,
    max_articles: Optional[int] = None,
) -> List[Dict]:
    """Fetch news for a ticker from ALL available providers and merge results.

    Runs every provider that has a valid API key (or no key required),
    then deduplicates by URL and sorts by recency. This means Yahoo RSS
    and NewsAPI (and Finnhub if keyed) all contribute articles.

    All API keys are read from environment variables:
        NEWSAPI_KEY, FINNHUB_KEY, AV_KEY
    """
    ncfg = cfg.get("news", {})
    max_art = max_articles or ncfg.get("max_articles", 30)
    lookback = ncfg.get("lookback_hours", 72)

    newsapi_key = os.getenv("NEWSAPI_KEY", "").strip()
    finnhub_key = os.getenv("FINNHUB_KEY", "").strip()
    av_key = os.getenv("AV_KEY", "").strip()

    all_articles: List[Dict] = []

    # 1. Yahoo RSS — always free, no key needed
    yahoo = fetch_yahoo_rss(ticker, max_articles=max_art)
    if yahoo:
        logger.info(f"Yahoo RSS contributed {len(yahoo)} articles")
        all_articles.extend(yahoo)

    # 2. NewsAPI — if key available
    if newsapi_key:
        newsapi = fetch_newsapi(ticker, newsapi_key, lookback_hours=lookback, max_articles=max_art)
        if newsapi:
            logger.info(f"NewsAPI contributed {len(newsapi)} articles")
            all_articles.extend(newsapi)

    # 3. Finnhub — if key available
    if finnhub_key:
        finnhub = fetch_finnhub(ticker, finnhub_key, lookback_days=lookback // 24, max_articles=max_art)
        if finnhub:
            logger.info(f"Finnhub contributed {len(finnhub)} articles")
            all_articles.extend(finnhub)

    # 4. Alpha Vantage — if key available
    if av_key:
        av = fetch_alphavantage(ticker, av_key, max_articles=max_art)
        if av:
            logger.info(f"Alpha Vantage contributed {len(av)} articles")
            all_articles.extend(av)

    # 5. GDELT fallback — only if nothing else worked
    if not all_articles:
        company = TICKER_NAMES.get(ticker.upper(), ticker)
        gdelt = fetch_gdelt(ticker, company_name=company, max_articles=max_art)
        if gdelt:
            logger.info(f"GDELT fallback contributed {len(gdelt)} articles")
            all_articles.extend(gdelt)

    # Deduplicate by URL, sort newest first, cap at max_art
    seen_urls = set()
    unique = []
    for a in sorted(all_articles, key=lambda x: x["published"], reverse=True):
        if a["url"] not in seen_urls and a["title"]:
            seen_urls.add(a["url"])
            unique.append(a)

    logger.info(f"Total unique articles for {ticker}: {len(unique)} (from {len(all_articles)} raw)")
    return unique[:max_art]
