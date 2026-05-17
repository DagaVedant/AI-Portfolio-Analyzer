"""
components.py — Reusable Streamlit UI components for the AI Portfolio Analyzer.

All components accept a pre-computed result dict from src/inference.py.
No emojis used — clean text labels only.
"""

import streamlit as st
from typing import Dict, List, Optional

from dashboard.dashboard_utils import (
    fmt_pct, fmt_price, fmt_num,
    sentiment_label_color, outlook_emoji,
    ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW, ACCENT_BLUE,
    MUTED_COLOR, TEXT_COLOR,
)

# Override theme constants to match new design
ACCENT_GREEN  = "#3ef5c8"
ACCENT_RED    = "#ff6b6b"
ACCENT_YELLOW = "#ffd166"
ACCENT_BLUE   = "#58a6ff"
TEXT_COLOR    = "#dde4f0"
MUTED_COLOR   = "#4e6080"
CARD_BG       = "#0f1824"
BORDER_COLOR  = "#1e2d45"


# ---------------------------------------------------------------------------
# Generic card renderer
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "", color: str = TEXT_COLOR):
    """Render a single styled metric card."""
    delta_html = ""
    if delta:
        d_color = ACCENT_GREEN if delta.startswith("+") else ACCENT_RED
        delta_html = f'<div style="color:{d_color};font-size:0.76rem;margin-top:3px;font-family:\'JetBrains Mono\',monospace">{delta}</div>'

    st.markdown(
        f"""
        <div style="
            background:{CARD_BG};
            border:1px solid {BORDER_COLOR};
            border-radius:10px;
            padding:14px 18px;
            text-align:center;
            transition: border-color 0.2s ease;
        ">
            <div style="color:{MUTED_COLOR};font-size:0.68rem;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:5px;
                        font-family:'JetBrains Mono',monospace">{label}</div>
            <div style="color:{color};font-size:1.35rem;font-weight:700;
                        font-family:'JetBrains Mono',monospace">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="margin:24px 0 12px 0;padding-bottom:10px;
                    border-bottom:1px solid {BORDER_COLOR}">
            <span style="color:{TEXT_COLOR};font-size:1.05rem;font-weight:700;
                         letter-spacing:0.02em;font-family:'Syne',sans-serif">{title}</span>
            {f"<span style='color:{MUTED_COLOR};font-size:0.78rem;margin-left:10px;font-family:JetBrains Mono,monospace'>{subtitle}</span>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def tag(text: str, color: str = ACCENT_BLUE):
    return (
        f'<span style="background:{color}22;color:{color};'
        f'padding:2px 8px;border-radius:6px;font-size:0.7rem;'
        f'font-weight:700;letter-spacing:0.06em;font-family:JetBrains Mono,monospace">{text}</span>'
    )


# ---------------------------------------------------------------------------
# Market overview section
# ---------------------------------------------------------------------------

def render_market_overview(result: Dict):
    section_header("Market Overview", result.get("ticker", ""))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Current Price", fmt_price(result.get("current_price", 0)))
    with c2:
        r1d = result.get("daily_return", 0)
        metric_card("1-Day Return", fmt_pct(r1d),
                    color=ACCENT_GREEN if r1d >= 0 else ACCENT_RED)
    with c3:
        r30 = result.get("return_30d", 0)
        metric_card("30-Day Return", fmt_pct(r30),
                    color=ACCENT_GREEN if r30 >= 0 else ACCENT_RED)
    with c4:
        r90 = result.get("return_90d", 0)
        metric_card("90-Day Return", fmt_pct(r90),
                    color=ACCENT_GREEN if r90 >= 0 else ACCENT_RED)
    with c5:
        metric_card("Volume", fmt_num(result.get("volume", 0), 1))


# ---------------------------------------------------------------------------
# Forecast panel
# ---------------------------------------------------------------------------

def render_forecast_panel(result: Dict):
    outlook = result.get("outlook", "Neutral")
    outlook_icon = {"Bullish": "Bull", "Bearish": "Bear", "Neutral": "Neutral"}.get(outlook, outlook)
    section_header(f"Forecast — {outlook_icon}", f"Horizon: {result.get('forecast_horizon_days', 21)} trading days")

    c1, c2, c3, c4 = st.columns(4)
    ret = result.get("pred_return", 0)
    with c1:
        metric_card("Predicted Return", fmt_pct(ret),
                    color=ACCENT_GREEN if ret >= 0 else ACCENT_RED)
    with c2:
        metric_card("Forecast Low", fmt_price(result.get("price_forecast_low", 0)),
                    color=ACCENT_RED)
    with c3:
        metric_card("Forecast Mid", fmt_price(result.get("price_forecast_mid", 0)),
                    color=ACCENT_YELLOW)
    with c4:
        metric_card("Forecast High", fmt_price(result.get("price_forecast_high", 0)),
                    color=ACCENT_GREEN)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        conf = result.get("model_confidence", 0)
        metric_card("Model Confidence", f"{conf*100:.0f}%",
                    color=ACCENT_GREEN if conf > 0.6 else ACCENT_YELLOW)
    with c6:
        outlook_color = {"Bullish": ACCENT_GREEN, "Bearish": ACCENT_RED, "Neutral": ACCENT_YELLOW}
        metric_card("Outlook", outlook, color=outlook_color.get(outlook, TEXT_COLOR))
    with c7:
        metric_card("Model", result.get("model_used", "N/A"), color=ACCENT_BLUE)


# ---------------------------------------------------------------------------
# Risk panel
# ---------------------------------------------------------------------------

def render_risk_panel(result: Dict):
    section_header("Risk Analysis")

    risk_score = result.get("risk_score", 0)
    risk_label = result.get("risk_label", "N/A")
    risk_color = result.get("risk_color", MUTED_COLOR)

    st.markdown(
        f"""
        <div style="background:{CARD_BG};border:1px solid {risk_color}33;
                    border-radius:12px;padding:16px;text-align:center;margin-bottom:12px">
            <div style="color:{MUTED_COLOR};font-size:0.68rem;text-transform:uppercase;
                        letter-spacing:0.14em;font-family:'JetBrains Mono',monospace">Composite Risk Score</div>
            <div style="color:{risk_color};font-size:2.8rem;font-weight:800;
                        font-family:'JetBrains Mono',monospace;line-height:1.1;margin:8px 0">{risk_score:.0f}</div>
            <div style="color:{risk_color};font-size:0.92rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.08em">{risk_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vol = result.get("ann_volatility", 0)
        metric_card("Annual Volatility", fmt_pct(vol),
                    color=ACCENT_YELLOW if vol < 0.3 else ACCENT_RED)
    with c2:
        var = result.get("var_1d", 0)
        metric_card("VaR 1-Day 95%", fmt_pct(abs(var)),
                    color=ACCENT_YELLOW)
    with c3:
        cvar = result.get("cvar_1d", 0)
        metric_card("CVaR 1-Day 95%", fmt_pct(abs(cvar)),
                    color=ACCENT_RED)
    with c4:
        dd = result.get("max_drawdown", 0)
        metric_card("Max Drawdown", fmt_pct(abs(dd)),
                    color=ACCENT_RED)

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5:
        metric_card("Beta vs SPY", f"{result.get('beta', 1):.2f}",
                    color=ACCENT_YELLOW)
    with c6:
        dp = result.get("pred_downside_prob", 0)
        metric_card("Downside Prob", fmt_pct(dp),
                    color=ACCENT_RED if dp > 0.5 else ACCENT_YELLOW)
    with c7:
        var_h = result.get("var_horizon", 0)
        metric_card(f"VaR {result.get('forecast_horizon_days',21)}-Day", fmt_pct(abs(var_h)),
                    color=ACCENT_RED)


# ---------------------------------------------------------------------------
# News sentiment panel
# ---------------------------------------------------------------------------

def render_sentiment_panel(result: Dict):
    sf = result.get("sentiment_features", {})
    articles = result.get("articles", [])

    section_header("News Sentiment", f"{sf.get('news_volume', 0)} articles via NewsAPI + Yahoo RSS")

    c1, c2, c3, c4 = st.columns(4)
    ws = sf.get("weighted_sentiment", 0)
    with c1:
        metric_card("Weighted Sentiment", f"{ws:+.3f}",
                    color=ACCENT_GREEN if ws > 0.1 else (ACCENT_RED if ws < -0.1 else ACCENT_YELLOW))
    with c2:
        metric_card("Bullish", fmt_pct(sf.get("positive_news_ratio", 0), 0),
                    color=ACCENT_GREEN)
    with c3:
        metric_card("Bearish", fmt_pct(sf.get("negative_news_ratio", 0), 0),
                    color=ACCENT_RED)
    with c4:
        metric_card("Sent. Volatility", f"{sf.get('sentiment_volatility', 0):.3f}",
                    color=ACCENT_YELLOW)

    if not articles:
        st.info("No news articles fetched. Check NEWSAPI_KEY in .env")
        return

    bullish = [a for a in articles if a.get("sentiment", {}).get("label") == "bullish"]
    bearish = [a for a in articles if a.get("sentiment", {}).get("label") == "bearish"]

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown(
            f'<div style="color:{ACCENT_GREEN};font-family:JetBrains Mono,monospace;'
            f'font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;'
            f'margin-bottom:10px">Bullish Articles</div>',
            unsafe_allow_html=True,
        )
        for a in bullish[:4]:
            _render_article_card(a)

    with col_neg:
        st.markdown(
            f'<div style="color:{ACCENT_RED};font-family:JetBrains Mono,monospace;'
            f'font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;'
            f'margin-bottom:10px">Bearish Articles</div>',
            unsafe_allow_html=True,
        )
        for a in bearish[:4]:
            _render_article_card(a)

    neutral = [a for a in articles if a.get("sentiment", {}).get("label") == "neutral"]
    if neutral:
        with st.expander(f"Neutral Articles ({len(neutral)})"):
            for a in neutral[:5]:
                _render_article_card(a)


def _render_article_card(article: Dict):
    sent = article.get("sentiment", {})
    label = sent.get("label", "neutral")
    score = sent.get("score", 0)
    color = sentiment_label_color(label)
    # Override with new theme colors
    if label == "bullish":
        color = ACCENT_GREEN
    elif label == "bearish":
        color = ACCENT_RED
    else:
        color = ACCENT_YELLOW

    age = article.get("age_hours", 0)
    age_str = f"{int(age)}h ago" if age < 48 else f"{int(age/24)}d ago"
    url = article.get("url", "#")
    title = article.get("title", "No title")[:90]
    source = article.get("source", "")

    st.markdown(
        f"""
        <div style="background:{CARD_BG};border-left:2px solid {color};
                    border-radius:6px;padding:10px 14px;margin-bottom:8px;
                    transition: transform 0.15s ease">
            <a href="{url}" target="_blank" style="color:{TEXT_COLOR};
               text-decoration:none;font-size:0.82rem;font-weight:600;
               line-height:1.45;font-family:'Syne',sans-serif">{title}</a>
            <div style="margin-top:6px">
                <span style="color:{MUTED_COLOR};font-size:0.7rem;
                             font-family:'JetBrains Mono',monospace">{source} &middot; {age_str}</span>
                <span style="float:right;background:{color}18;color:{color};
                             padding:1px 8px;border-radius:4px;font-size:0.68rem;
                             font-weight:700;font-family:'JetBrains Mono',monospace;
                             letter-spacing:0.04em">{label.upper()} {score:+.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Model info footer
# ---------------------------------------------------------------------------

def render_model_info(result: Dict):
    section_header("Model Information")
    st.markdown(
        f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:10px;padding:14px 18px">
            <div style="display:flex;gap:24px;flex-wrap:wrap;font-family:'JetBrains Mono',monospace;font-size:0.8rem">
                <span>Model: <span style="color:{ACCENT_BLUE}">{result.get('model_used','N/A')}</span></span>
                <span>Timestamp: <span style="color:{MUTED_COLOR}">{result.get('timestamp','N/A')[:19]} UTC</span></span>
                <span>Horizon: <span style="color:{ACCENT_YELLOW}">{result.get('forecast_horizon_days',21)} days</span></span>
            </div>
            <div style="color:{MUTED_COLOR};font-size:0.72rem;margin-top:10px;line-height:1.6;
                        font-family:'JetBrains Mono',monospace">
                Educational Disclaimer: All predictions, risk metrics and portfolio suggestions
                are generated by machine-learning models for educational purposes only and do NOT
                constitute financial advice. Past performance is no guarantee of future results.
                Always consult a qualified financial advisor before making investment decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
