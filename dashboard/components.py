"""
components.py — Reusable Streamlit UI components for the AI Portfolio Analyzer.

All components accept a pre-computed result dict from src/inference.py
and render polished, card-style sections.
"""

import streamlit as st
from typing import Dict, List, Optional

from dashboard.dashboard_utils import (
    fmt_pct, fmt_price, fmt_num,
    sentiment_label_color, outlook_emoji,
    ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW, ACCENT_BLUE,
    MUTED_COLOR, TEXT_COLOR,
)


# ---------------------------------------------------------------------------
# Generic card renderer
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "", color: str = TEXT_COLOR):
    """Render a single styled metric card."""
    delta_html = ""
    if delta:
        d_color = ACCENT_GREEN if delta.startswith("+") else ACCENT_RED
        delta_html = f'<div style="color:{d_color};font-size:0.78rem;margin-top:2px">{delta}</div>'

    st.markdown(
        f"""
        <div style="
            background:#161b22;
            border:1px solid #21262d;
            border-radius:10px;
            padding:14px 18px;
            text-align:center;
        ">
            <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.08em;margin-bottom:4px">{label}</div>
            <div style="color:{color};font-size:1.4rem;font-weight:700;
                        font-family:'IBM Plex Mono',monospace">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="margin:28px 0 10px 0;padding-bottom:8px;
                    border-bottom:1px solid #21262d">
            <span style="color:#e6edf3;font-size:1.1rem;font-weight:700;
                         letter-spacing:0.04em">{title}</span>
            {"<span style='color:#8b949e;font-size:0.82rem;margin-left:10px'>" + subtitle + "</span>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def tag(text: str, color: str = ACCENT_BLUE):
    return (
        f'<span style="background:{color}22;color:{color};'
        f'padding:2px 8px;border-radius:20px;font-size:0.72rem;'
        f'font-weight:600;letter-spacing:0.05em">{text}</span>'
    )


# ---------------------------------------------------------------------------
# Market overview section
# ---------------------------------------------------------------------------

def render_market_overview(result: Dict):
    section_header("📊 Market Overview", result.get("ticker", ""))

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

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6:
        sma20 = result.get("sma_20", 0)
        price = result.get("current_price", sma20)
        color = ACCENT_GREEN if price > sma20 else ACCENT_RED
        metric_card("SMA 20", fmt_price(sma20), color=color)
    with c7:
        sma50 = result.get("sma_50", 0)
        color = ACCENT_GREEN if price > sma50 else ACCENT_RED
        metric_card("SMA 50", fmt_price(sma50), color=color)


# ---------------------------------------------------------------------------
# Forecast panel
# ---------------------------------------------------------------------------

def render_forecast_panel(result: Dict):
    outlook = result.get("outlook", "Neutral")
    section_header(f"🔮 Forecast Panel  {outlook_emoji(outlook)}", f"Horizon: {result.get('forecast_horizon_days', 21)} trading days")

    c1, c2, c3, c4 = st.columns(4)
    ret = result.get("pred_return", 0)
    with c1:
        metric_card("Predicted Return",
                    fmt_pct(ret),
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
    section_header("⚠️ Risk Panel")

    risk_score = result.get("risk_score", 0)
    risk_label = result.get("risk_label", "N/A")
    risk_color = result.get("risk_color", MUTED_COLOR)

    # Big risk score
    st.markdown(
        f"""
        <div style="background:#161b22;border:1px solid {risk_color}44;
                    border-radius:12px;padding:16px;text-align:center;margin-bottom:12px">
            <div style="color:#8b949e;font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.1em">Composite Risk Score</div>
            <div style="color:{risk_color};font-size:2.8rem;font-weight:800;
                        font-family:'IBM Plex Mono',monospace;line-height:1.1">{risk_score:.0f}</div>
            <div style="color:{risk_color};font-size:1rem;font-weight:600;margin-top:2px">{risk_label}</div>
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
        metric_card("VaR (1-Day, 95%)", fmt_pct(abs(var)),
                    color=ACCENT_YELLOW)
    with c3:
        cvar = result.get("cvar_1d", 0)
        metric_card("CVaR (1-Day, 95%)", fmt_pct(abs(cvar)),
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
        metric_card(f"VaR ({result.get('forecast_horizon_days',21)}-Day)", fmt_pct(abs(var_h)),
                    color=ACCENT_RED)


# ---------------------------------------------------------------------------
# News sentiment panel
# ---------------------------------------------------------------------------

def render_sentiment_panel(result: Dict):
    sf = result.get("sentiment_features", {})
    articles = result.get("articles", [])

    section_header("📰 News Sentiment", f"{sf.get('news_volume', 0)} articles analysed")

    c1, c2, c3, c4 = st.columns(4)
    ws = sf.get("weighted_sentiment", 0)
    with c1:
        metric_card("Weighted Sentiment", f"{ws:+.3f}",
                    color=ACCENT_GREEN if ws > 0.1 else (ACCENT_RED if ws < -0.1 else ACCENT_YELLOW))
    with c2:
        metric_card("Bullish %", fmt_pct(sf.get("positive_news_ratio", 0), 0),
                    color=ACCENT_GREEN)
    with c3:
        metric_card("Bearish %", fmt_pct(sf.get("negative_news_ratio", 0), 0),
                    color=ACCENT_RED)
    with c4:
        metric_card("Sentiment Volatility", f"{sf.get('sentiment_volatility', 0):.3f}",
                    color=ACCENT_YELLOW)

    if not articles:
        st.info("No news articles fetched.")
        return

    # Split articles by label
    bullish = [a for a in articles if a.get("sentiment", {}).get("label") == "bullish"]
    bearish = [a for a in articles if a.get("sentiment", {}).get("label") == "bearish"]

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown(f"**<span style='color:{ACCENT_GREEN}'>📈 Top Bullish Articles</span>**",
                    unsafe_allow_html=True)
        for a in bullish[:4]:
            _render_article_card(a)

    with col_neg:
        st.markdown(f"**<span style='color:{ACCENT_RED}'>📉 Top Bearish Articles</span>**",
                    unsafe_allow_html=True)
        for a in bearish[:4]:
            _render_article_card(a)

    # Neutral
    neutral = [a for a in articles if a.get("sentiment", {}).get("label") == "neutral"]
    if neutral:
        with st.expander("➡️ Neutral Articles"):
            for a in neutral[:5]:
                _render_article_card(a)


def _render_article_card(article: Dict):
    sent = article.get("sentiment", {})
    label = sent.get("label", "neutral")
    score = sent.get("score", 0)
    color = sentiment_label_color(label)
    age = article.get("age_hours", 0)
    age_str = f"{int(age)}h ago" if age < 48 else f"{int(age/24)}d ago"
    url = article.get("url", "#")
    title = article.get("title", "No title")[:90]
    source = article.get("source", "")

    st.markdown(
        f"""
        <div style="background:#161b22;border-left:3px solid {color};
                    border-radius:6px;padding:10px 12px;margin-bottom:8px">
            <a href="{url}" target="_blank" style="color:{TEXT_COLOR};
               text-decoration:none;font-size:0.84rem;font-weight:600;
               line-height:1.4">{title}</a>
            <div style="margin-top:5px">
                <span style="color:#8b949e;font-size:0.72rem">{source} · {age_str}</span>
                <span style="float:right;background:{color}22;color:{color};
                             padding:1px 7px;border-radius:10px;font-size:0.70rem;
                             font-weight:700">{label.upper()} {score:+.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Portfolio recommendation panel
# ---------------------------------------------------------------------------

def render_portfolio_panel(result: Dict):
    port = result.get("portfolio", {})
    if not port:
        return

    section_header("💼 Portfolio Recommendation")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Expected Return", fmt_pct(port.get("expected_return", 0)),
                    color=ACCENT_GREEN)
    with c2:
        metric_card("Expected Volatility", fmt_pct(port.get("expected_volatility", 0)),
                    color=ACCENT_YELLOW)
    with c3:
        metric_card("Sharpe Ratio", f"{port.get('sharpe_ratio', 0):.2f}",
                    color=ACCENT_BLUE)

    weight_dict = port.get("weight_dict", {})
    if weight_dict:
        st.markdown("**Suggested Allocation:**")
        for t, w in sorted(weight_dict.items(), key=lambda x: -x[1]):
            pct_val = w * 100
            bar_color = ACCENT_GREEN if t == result.get("ticker") else ACCENT_BLUE
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;margin-bottom:5px">
                    <span style="color:{TEXT_COLOR};width:60px;font-size:0.85rem;
                                 font-weight:600">{t}</span>
                    <div style="flex:1;background:#21262d;border-radius:4px;height:18px;margin:0 10px">
                        <div style="width:{min(pct_val,100):.1f}%;background:{bar_color};
                                    height:100%;border-radius:4px"></div>
                    </div>
                    <span style="color:{bar_color};font-size:0.85rem;width:45px;
                                 text-align:right">{pct_val:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Model info footer
# ---------------------------------------------------------------------------

def render_model_info(result: Dict):
    section_header("ℹ️ Model Information")
    st.markdown(
        f"""
        <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;padding:14px 18px">
            <div style="display:flex;gap:20px;flex-wrap:wrap">
                <span>🤖 <b>Model:</b> <span style="color:{ACCENT_BLUE}">{result.get('model_used','N/A')}</span></span>
                <span>🕐 <b>Timestamp:</b> <span style="color:{MUTED_COLOR}">{result.get('timestamp','N/A')[:19]}</span></span>
                <span>📅 <b>Horizon:</b> <span style="color:{ACCENT_YELLOW}">{result.get('forecast_horizon_days',21)} days</span></span>
            </div>
            <div style="color:#8b949e;font-size:0.75rem;margin-top:10px;font-style:italic">
                ⚠️ <b>Educational Disclaimer:</b> All predictions, risk metrics and portfolio suggestions 
                are generated by machine-learning models for <b>educational purposes only</b> and do NOT 
                constitute financial advice. Past performance is no guarantee of future results. 
                Always consult a qualified financial advisor before making investment decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
