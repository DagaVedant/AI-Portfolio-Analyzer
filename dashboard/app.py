import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="AI Portfolio Analyzer",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    background-color: #080c12 !important;
    color: #dde4f0 !important;
    font-family: 'Syne', sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Loading overlay ── */
.loading-overlay {
    position: fixed;
    inset: 0;
    background: rgba(8,12,18,0.96);
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.loader-ring {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    border: 3px solid #1a2236;
    border-top-color: #3ef5c8;
    border-right-color: #3ef5c8;
    animation: spin 1s cubic-bezier(0.68,-0.55,0.27,1.55) infinite;
    margin-bottom: 24px;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.loader-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #3ef5c8;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    animation: blink 1.4s ease-in-out infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.loader-sub {
    font-size: 0.72rem;
    color: #2e4060;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
}

/* ── Header ── */
.header-banner {
    background: linear-gradient(160deg, #0d1320 0%, #111827 60%, #0d1320 100%);
    border: 1px solid #1e2d45;
    border-left: 3px solid #3ef5c8;
    border-radius: 10px;
    padding: 22px 30px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    animation: slideDown 0.5s ease;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-16px); }
    to { opacity: 1; transform: translateY(0); }
}

.header-banner::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 280px; height: 100%;
    background: radial-gradient(ellipse at 80% 50%, rgba(62,245,200,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #dde4f0;
    letter-spacing: -0.03em;
    margin-bottom: 4px;
}

.header-title .accent { color: #3ef5c8; }

.header-sub {
    color: #4e6080;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #080c12 !important;
    border-right: 1px solid #141e2e;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* ── Inputs ── */
.stTextInput input {
    background-color: #0f1824 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 7px !important;
    color: #dde4f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput input:focus {
    border-color: #3ef5c8 !important;
    box-shadow: 0 0 0 3px rgba(62,245,200,0.1) !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3ef5c8, #00c9a7) !important;
    color: #080c12 !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 0.55rem 1.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(62,245,200,0.25) !important;
}
.stButton > button {
    background-color: #0f1824 !important;
    color: #dde4f0 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 7px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: #3ef5c8 !important;
    color: #3ef5c8 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #0f1824;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
    color: #4e6080 !important;
    background: transparent !important;
    border-radius: 6px !important;
    padding: 7px 18px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    color: #dde4f0 !important;
    background: #1a2840 !important;
    border-bottom: 2px solid #3ef5c8 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0f1824;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 14px 16px;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stMetricLabel"] {
    color: #4e6080 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: #dde4f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Alert ── */
.stAlert {
    border-radius: 8px !important;
    background-color: #0f1824 !important;
    border-color: #1e2d45 !important;
}

/* ── Spinner ── */
.stSpinner > div > div {
    border-top-color: #3ef5c8 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: #0f1824 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #dde4f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c12; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3ef5c8; }

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Section animations ── */
.section-animate {
    animation: fadeUp 0.45s ease;
}

/* ── Status dot ── */
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3ef5c8;
    animation: pulse-dot 2s infinite;
    display: inline-block;
    margin-right: 8px;
    vertical-align: middle;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(62,245,200,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 5px rgba(62,245,200,0); }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Imports ────────────────────────────────────────────────────────────────
from src.utils import load_config, load_env
from src.inference import run_inference
from dashboard.components import (
    render_market_overview,
    render_forecast_panel,
    render_risk_panel,
    render_sentiment_panel,
    render_model_info,
    section_header,
    metric_card,
)
from dashboard.dashboard_utils import (
    price_chart, forecast_chart, volatility_chart,
    drawdown_chart, sentiment_gauge,
    risk_breakdown_chart,
)

# ── Load config & env ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_config():
    load_env()
    cfg = load_config("configs/config.yaml")
    # Force NewsAPI as preferred provider if key is available
    if os.getenv("NEWSAPI_KEY", "").strip():
        cfg["news"]["provider"] = "newsapi"
    return cfg


# ── Session state ──────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-banner">
        <div class="header-title">AI Portfolio <span class="accent">Analyzer</span></div>
        <div class="header-sub">
            ML-driven forecasts &nbsp;&middot;&nbsp; News sentiment via NewsAPI
            &nbsp;&middot;&nbsp; Risk projection
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="color:#3ef5c8;font-family:JetBrains Mono,monospace;'
        'font-size:0.82rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'
        'margin-bottom:18px">Controls</div>',
        unsafe_allow_html=True,
    )

    ticker_input = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter a US stock or ETF ticker (e.g. AAPL, TSLA, SPY)",
        placeholder="AAPL",
    ).upper().strip()

    popular = st.multiselect(
        "Quick-pick tickers",
        ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", "SPY", "QQQ"],
        default=[],
    )
    if popular:
        ticker_input = popular[-1]

    horizon = st.slider("Forecast horizon (days)", min_value=5, max_value=63, value=21, step=1)

    st.markdown("---")
    run_btn = st.button("Analyse", width= 'content', type="primary")
    clear_btn = st.button("Clear", width= 'content')

    if clear_btn:
        st.session_state.result = None
        st.session_state.last_ticker = ""
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="color:#2e4060;font-size:0.68rem;line-height:1.75;'
        'font-family:JetBrains Mono,monospace">'
        'Educational purposes only. No financial advice given or implied. '
        'Consult a qualified financial advisor before investing.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Loading overlay + analysis trigger ────────────────────────────────────
if run_btn and ticker_input:
    cfg = get_config()
    cfg["data"]["forecast_horizon"] = horizon

    # Show the custom loading overlay
    loading_html = f"""
    <div id="custom-loading" class="loading-overlay">
        <div class="loader-ring"></div>
        <div class="loader-text">Analysing {ticker_input}</div>
        <div class="loader-sub">NewsAPI &middot; FinBERT &middot; LSTM model</div>
    </div>
    """
    loading_placeholder = st.empty()
    loading_placeholder.markdown(loading_html, unsafe_allow_html=True)

    try:
        result = run_inference(ticker_input, cfg)
        if "error" in result:
            loading_placeholder.empty()
            st.error(result["error"])
            st.session_state.result = None
        else:
            st.session_state.result = result
            st.session_state.last_ticker = ticker_input
            loading_placeholder.empty()
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Unexpected error: {e}")
        st.session_state.result = None


# ── Results ────────────────────────────────────────────────────────────────
result = st.session_state.result

if result is None:
    st.markdown(
        """
        <div style="text-align:center;padding:90px 20px">
            <div style="font-size:1.9rem;font-weight:800;color:#dde4f0;margin-bottom:12px;
                        font-family:'Syne',sans-serif;letter-spacing:-0.03em;animation:fadeUp 0.5s ease">
                Enter a ticker and click Analyse
            </div>
            <div style="font-size:0.8rem;line-height:1.9;font-family:'JetBrains Mono',monospace;
                        color:#2e4060">
                ML-powered return forecasts &nbsp;&middot;&nbsp; risk projections
                &nbsp;&middot;&nbsp; live NewsAPI sentiment
            </div>
            <div style="margin-top:28px;display:flex;justify-content:center;gap:10px;flex-wrap:wrap">
        """ +
        "".join([
            f'<span style="background:#0f1824;border:1px solid #1e2d45;border-radius:6px;'
            f'padding:5px 14px;font-size:0.76rem;color:#3ef5c8;font-family:JetBrains Mono,monospace">{t}</span>'
            for t in ["AAPL", "MSFT", "NVDA", "TSLA", "META", "SPY", "QQQ"]
        ]) +
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Ticker title bar ───────────────────────────────────────────────────────
ticker = result.get("ticker", "")
price = result.get("current_price", 0)
r1d = result.get("daily_return", 0)
r1d_color = "#3ef5c8" if r1d >= 0 else "#ff6b6b"
r1d_sign = "+" if r1d >= 0 else ""
outlook = result.get("outlook", "Neutral")
outlook_colors = {"Bullish": "#3ef5c8", "Bearish": "#ff6b6b", "Neutral": "#ffd166"}
oc = outlook_colors.get(outlook, "#8b949e")
sf = result.get("sentiment_features", {})
news_count = sf.get("news_volume", 0)

st.markdown(
    f"""
    <div style="background:#0f1824;border:1px solid #1e2d45;border-radius:10px;
                padding:14px 24px;margin-bottom:18px;display:flex;
                align-items:center;gap:24px;flex-wrap:wrap;animation:slideDown 0.4s ease">
        <div>
            <span class="status-dot"></span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;
                         font-weight:700;color:#dde4f0">{ticker}</span>
            <span style="font-size:1.2rem;font-weight:700;color:#dde4f0;margin-left:14px">${price:,.2f}</span>
            <span style="color:{r1d_color};font-size:0.88rem;margin-left:10px;
                         font-family:'JetBrains Mono',monospace">
                {r1d_sign}{abs(r1d)*100:.2f}%
            </span>
        </div>
        <div style="background:{oc}18;color:{oc};padding:4px 14px;border-radius:6px;
                    font-size:0.75rem;font-weight:700;border:1px solid {oc}30;
                    font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;text-transform:uppercase">
            {outlook}
        </div>
        <div style="color:#2e4060;font-size:0.7rem;margin-left:auto;
                    font-family:'JetBrains Mono',monospace">
            NewsAPI + Yahoo RSS &middot; {news_count} articles &nbsp;|&nbsp; {result.get('timestamp','')[:19]} UTC
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Tabs (no Portfolio tab) ────────────────────────────────────────────────
tab_overview, tab_forecast, tab_risk, tab_news, tab_charts = st.tabs([
    "Overview", "Forecast", "Risk", "Sentiment", "Charts"
])


# ── Tab 1: Overview ────────────────────────────────────────────────────────
with tab_overview:
    # Key metrics row
    section_header("Market Snapshot", ticker)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Current Price", f"${price:,.2f}")
    with c2:
        r1d_val = result.get("daily_return", 0)
        metric_card("1-Day Return", f"{'+' if r1d_val>=0 else ''}{r1d_val*100:.2f}%",
                    color="#3ef5c8" if r1d_val >= 0 else "#ff6b6b")
    with c3:
        r30 = result.get("return_30d", 0)
        metric_card("30-Day Return", f"{'+' if r30>=0 else ''}{r30*100:.2f}%",
                    color="#3ef5c8" if r30 >= 0 else "#ff6b6b")
    with c4:
        r90 = result.get("return_90d", 0)
        metric_card("90-Day Return", f"{'+' if r90>=0 else ''}{r90*100:.2f}%",
                    color="#3ef5c8" if r90 >= 0 else "#ff6b6b")

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # Prediction prices
    section_header("Price Predictions", f"{result.get('forecast_horizon_days', 21)}-Day Horizon")

    low_p = result.get("price_forecast_low", 0)
    mid_p = result.get("price_forecast_mid", 0)
    high_p = result.get("price_forecast_high", 0)
    conf = result.get("model_confidence", 0)

    c_low, c_mid, c_high, c_conf = st.columns(4)
    with c_low:
        low_chg = (low_p - price) / price if price else 0
        metric_card("Bear Case", f"${low_p:,.2f}",
                    delta=f"{'+' if low_chg>=0 else ''}{low_chg*100:.1f}%",
                    color="#ff6b6b")
    with c_mid:
        mid_chg = (mid_p - price) / price if price else 0
        metric_card("Base Case", f"${mid_p:,.2f}",
                    delta=f"{'+' if mid_chg>=0 else ''}{mid_chg*100:.1f}%",
                    color="#ffd166")
    with c_high:
        high_chg = (high_p - price) / price if price else 0
        metric_card("Bull Case", f"${high_p:,.2f}",
                    delta=f"{'+' if high_chg>=0 else ''}{high_chg*100:.1f}%",
                    color="#3ef5c8")
    with c_conf:
        metric_card("Model Confidence", f"{conf*100:.0f}%",
                    color="#3ef5c8" if conf > 0.6 else "#ffd166")

    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # Price chart + sentiment gauge
    col_chart, col_side = st.columns([2, 1])
    with col_chart:
        fig = price_chart(ticker, lookback_days=180)
        if fig:
            st.plotly_chart(fig, width= 'content', config={"displayModeBar": False})
    with col_side:
        ws = sf.get("weighted_sentiment", 0.0)
        fig_g = sentiment_gauge(ws, label="Sentiment Score")
        if fig_g:
            st.plotly_chart(fig_g, width= 'content', config={"displayModeBar": False})

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        risk_score = result.get("risk_score", 0)
        risk_label = result.get("risk_label", "")
        risk_color = result.get("risk_color", "#8b949e")
        st.markdown(
            f"""
            <div style="background:#0f1824;border:1px solid {risk_color}33;
                        border-radius:10px;padding:14px;text-align:center">
                <div style="color:#2e4060;font-size:0.66rem;text-transform:uppercase;
                            letter-spacing:0.14em;font-family:'JetBrains Mono',monospace">Risk Score</div>
                <div style="color:{risk_color};font-size:2.2rem;font-weight:800;
                            font-family:'JetBrains Mono',monospace;margin:6px 0">{risk_score:.0f}</div>
                <div style="color:{risk_color};font-size:0.82rem;font-weight:700;
                            letter-spacing:0.06em;text-transform:uppercase">{risk_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_model_info(result)


# ── Tab 2: Forecast ────────────────────────────────────────────────────────
with tab_forecast:
    render_forecast_panel(result)
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    fig_fc = forecast_chart(result)
    if fig_fc:
        st.plotly_chart(fig_fc, width= 'content', config={"displayModeBar": False})
    else:
        st.info("Forecast chart unavailable — check market data.")

    sa = result.get("sentiment_adjustment", {})
    if sa:
        section_header("Sentiment Adjustments Applied")
        c1, c2, c3, c4 = st.columns(4)
        for col, (lbl, val, fmt) in zip(
            [c1, c2, c3, c4],
            [
                ("Return Adj", sa.get("return_adj", 0), "{:+.3%}"),
                ("Vol Adj", sa.get("vol_adj", 0), "+{:.3%}"),
                ("Downside Adj", sa.get("downside_adj", 0), "+{:.3%}"),
                ("Uncertainty", sa.get("uncertainty_mult", 1), "{:.2f}x"),
            ],
        ):
            with col:
                metric_card(lbl, fmt.format(val))


# ── Tab 3: Risk ────────────────────────────────────────────────────────────
with tab_risk:
    render_risk_panel(result)
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    fig_rb = risk_breakdown_chart(result)
    if fig_rb:
        st.plotly_chart(fig_rb, width= 'content', config={"displayModeBar": False})


# ── Tab 4: News / Sentiment ────────────────────────────────────────────────
with tab_news:
    render_sentiment_panel(result)


# ── Tab 5: Charts ──────────────────────────────────────────────────────────
with tab_charts:
    section_header("Technical Charts")

    col_vol, col_dd = st.columns(2)
    with col_vol:
        fig_vol = volatility_chart(ticker)
        if fig_vol:
            st.plotly_chart(fig_vol, width= 'content', config={"displayModeBar": False})
        else:
            st.info("Volatility chart unavailable.")
    with col_dd:
        fig_dd = drawdown_chart(ticker)
        if fig_dd:
            st.plotly_chart(fig_dd, width= 'content', config={"displayModeBar": False})
        else:
            st.info("Drawdown chart unavailable.")

    fig_price_full = price_chart(ticker, lookback_days=365)
    if fig_price_full:
        st.plotly_chart(fig_price_full, width= 'content', config={"displayModeBar": False})