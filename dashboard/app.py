"""
app.py — Main Streamlit dashboard for the AI Portfolio Analyzer.

Run with:
    streamlit run dashboard/app.py

Features:
  - Dark mode with custom CSS
  - Ticker input + horizon selector
  - Live inference pipeline call
  - Charts: price, forecast, volatility, drawdown, sentiment gauge, portfolio pie, risk breakdown
  - News sentiment panel with article cards
  - Portfolio recommendation with allocation bars
  - Model info footer with disclaimer
"""

import sys
import os

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="AI Portfolio Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Root overrides ── */
html, body, [class*="css"] {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Main container ── */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-top: 3px solid #00d4aa;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,212,170,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.header-sub {
    color: #8b949e;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
}
.accent-dot {
    color: #00d4aa;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* ── Input widgets ── */
.stTextInput input, .stSelectbox select, .stSlider {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stTextInput input:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.12) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4aa, #00b894) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.8rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3) !important;
}
.stButton > button {
    background-color: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    background: transparent !important;
    border-radius: 8px !important;
    padding: 6px 16px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.83rem !important;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important;
    background: #21262d !important;
    border-bottom: 2px solid #00d4aa !important;
}

/* ── Spinner ── */
.stSpinner > div > div {
    border-top-color: #00d4aa !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'IBM Plex Mono', monospace !important; }

/* ── Info / warning boxes ── */
.stAlert {
    border-radius: 8px !important;
    background-color: #161b22 !important;
    border-color: #30363d !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

/* ── Loading pulse animation ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.loading-text { animation: pulse 1.5s infinite; color: #00d4aa; font-family: 'IBM Plex Mono', monospace; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Imports (after path setup) ─────────────────────────────────────────────
from src.utils import load_config, load_env
from src.inference import run_inference
from dashboard.components import (
    render_market_overview,
    render_forecast_panel,
    render_risk_panel,
    render_sentiment_panel,
    render_portfolio_panel,
    render_model_info,
    section_header,
    metric_card,
)
from dashboard.dashboard_utils import (
    price_chart, forecast_chart, volatility_chart,
    drawdown_chart, sentiment_gauge, portfolio_pie,
    risk_breakdown_chart,
)

# ── Load config & env ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_config():
    load_env()
    return load_config("configs/config.yaml")


# ── Session state ──────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""


# ────────────────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-banner">
        <div class="header-title">AI Portfolio Analyzer<span class="accent-dot">.</span></div>
        <div class="header-sub">
            Machine-learning driven stock analysis · News sentiment · Risk projection · Portfolio optimization
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Controls
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="color:#00d4aa;font-family:IBM Plex Mono,monospace;'
        'font-size:1rem;font-weight:700;margin-bottom:16px">⚙️ Controls</div>',
        unsafe_allow_html=True,
    )

    ticker_input = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter a US stock or ETF ticker (e.g. AAPL, TSLA, SPY)",
        placeholder="AAPL",
    ).upper().strip()

    popular = st.multiselect(
        "Quick-pick popular tickers",
        ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", "SPY", "QQQ"],
        default=[],
        help="Click a ticker to override the input above",
    )
    if popular:
        ticker_input = popular[-1]

    horizon = st.slider("Forecast horizon (trading days)", min_value=5, max_value=63, value=21, step=1)

    st.markdown("---")
    run_btn = st.button("🚀 Analyse", use_container_width=True, type="primary")
    clear_btn = st.button("🗑 Clear", use_container_width=True)

    if clear_btn:
        st.session_state.result = None
        st.session_state.last_ticker = ""
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="color:#8b949e;font-size:0.72rem;line-height:1.6">'
        '⚠️ <b>Disclaimer:</b> This tool is for <b>educational purposes only</b>. '
        'No financial advice is given or implied. Always consult a qualified '
        'financial advisor before investing.'
        '</div>',
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────────────────
# ANALYSIS TRIGGER
# ────────────────────────────────────────────────────────────────────────────
if run_btn and ticker_input:
    cfg = get_config()
    cfg["data"]["forecast_horizon"] = horizon

    with st.spinner(f"Analysing {ticker_input} …"):
        try:
            result = run_inference(ticker_input, cfg)
            if "error" in result:
                st.error(f"❌ {result['error']}")
                st.session_state.result = None
            else:
                st.session_state.result = result
                st.session_state.last_ticker = ticker_input
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.session_state.result = None

# ────────────────────────────────────────────────────────────────────────────
# RESULTS — Rendered only when data is available
# ────────────────────────────────────────────────────────────────────────────
result = st.session_state.result

if result is None:
    # Landing / empty state
    st.markdown(
        """
        <div style="text-align:center;padding:80px 20px;color:#8b949e">
            <div style="font-size:3.5rem;margin-bottom:16px">📈</div>
            <div style="font-size:1.15rem;color:#e6edf3;font-weight:600;margin-bottom:8px">
                Enter a ticker and click Analyse
            </div>
            <div style="font-size:0.88rem;line-height:1.7">
                Get ML-powered return forecasts, risk projections, live news sentiment,<br>
                and portfolio optimisation for any US stock or ETF.
            </div>
            <div style="margin-top:24px;display:flex;justify-content:center;gap:10px;flex-wrap:wrap">
        """ +
        "".join([
            f'<span style="background:#161b22;border:1px solid #21262d;border-radius:20px;'
            f'padding:5px 14px;font-size:0.80rem;color:#58a6ff">{t}</span>'
            for t in ["AAPL", "MSFT", "NVDA", "TSLA", "META", "SPY", "QQQ"]
        ]) +
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ────────────────────────────────────────────────────────────────────────────
# TICKER TITLE BAR
# ────────────────────────────────────────────────────────────────────────────
ticker = result.get("ticker", "")
price = result.get("current_price", 0)
r1d = result.get("daily_return", 0)
r1d_color = "#00d4aa" if r1d >= 0 else "#ff6b6b"
r1d_sign = "▲" if r1d >= 0 else "▼"
outlook = result.get("outlook", "Neutral")
outlook_colors = {"Bullish": "#00d4aa", "Bearish": "#ff6b6b", "Neutral": "#ffd166"}
oc = outlook_colors.get(outlook, "#8b949e")

st.markdown(
    f"""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:10px;
                padding:14px 24px;margin-bottom:16px;display:flex;
                align-items:center;gap:24px;flex-wrap:wrap">
        <div>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;
                         font-weight:800;color:#e6edf3">{ticker}</span>
            <span style="font-size:1.3rem;font-weight:700;color:#e6edf3;
                         margin-left:14px">${price:,.2f}</span>
            <span style="color:{r1d_color};font-size:0.95rem;margin-left:8px">
                {r1d_sign} {abs(r1d)*100:.2f}%
            </span>
        </div>
        <div style="background:{oc}22;color:{oc};padding:4px 14px;border-radius:20px;
                    font-size:0.82rem;font-weight:700;border:1px solid {oc}44">
            {outlook}
        </div>
        <div style="color:#8b949e;font-size:0.78rem;margin-left:auto">
            Updated: {result.get('timestamp','')[:19]} UTC
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ────────────────────────────────────────────────────────────────────────────
# TABBED SECTIONS
# ────────────────────────────────────────────────────────────────────────────
tab_overview, tab_forecast, tab_risk, tab_news, tab_portfolio, tab_charts = st.tabs([
    "📊 Overview", "🔮 Forecast", "⚠️ Risk", "📰 Sentiment", "💼 Portfolio", "📉 Charts"
])


# ── Tab 1: Overview ──────────────────────────────────────────────────────
with tab_overview:
    render_market_overview(result)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    col_chart, col_sent = st.columns([2, 1])
    with col_chart:
        fig = price_chart(ticker, lookback_days=180)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col_sent:
        sf = result.get("sentiment_features", {})
        ws = sf.get("weighted_sentiment", 0.0)
        fig_g = sentiment_gauge(ws, label="Sentiment Score")
        if fig_g:
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        risk_score = result.get("risk_score", 0)
        risk_label = result.get("risk_label", "")
        risk_color = result.get("risk_color", "#8b949e")
        st.markdown(
            f"""
            <div style="background:#161b22;border:1px solid {risk_color}44;
                        border-radius:10px;padding:14px;text-align:center">
                <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;
                            letter-spacing:0.1em">Risk Score</div>
                <div style="color:{risk_color};font-size:2rem;font-weight:800;
                            font-family:'IBM Plex Mono',monospace">{risk_score:.0f}</div>
                <div style="color:{risk_color};font-size:0.88rem;font-weight:600">{risk_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_model_info(result)


# ── Tab 2: Forecast ──────────────────────────────────────────────────────
with tab_forecast:
    render_forecast_panel(result)
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    fig_fc = forecast_chart(result)
    if fig_fc:
        st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Forecast chart unavailable — check market data.")

    # Sentiment adjustment table
    sa = result.get("sentiment_adjustment", {})
    if sa:
        section_header("Sentiment Adjustments Applied")
        c1, c2, c3, c4 = st.columns(4)
        cols = [c1, c2, c3, c4]
        items = [
            ("Return Adj", sa.get("return_adj", 0), "±{:.3%}"),
            ("Vol Adj", sa.get("vol_adj", 0), "+{:.3%}"),
            ("Downside Adj", sa.get("downside_adj", 0), "+{:.3%}"),
            ("Uncertainty Mult", sa.get("uncertainty_mult", 1), "{:.2f}×"),
        ]
        for col, (lbl, val, fmt) in zip(cols, items):
            with col:
                metric_card(lbl, fmt.format(val))


# ── Tab 3: Risk ──────────────────────────────────────────────────────────
with tab_risk:
    render_risk_panel(result)
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    fig_rb = risk_breakdown_chart(result)
    if fig_rb:
        st.plotly_chart(fig_rb, use_container_width=True, config={"displayModeBar": False})


# ── Tab 4: News / Sentiment ──────────────────────────────────────────────
with tab_news:
    render_sentiment_panel(result)


# ── Tab 5: Portfolio ─────────────────────────────────────────────────────
with tab_portfolio:
    render_portfolio_panel(result)
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    port = result.get("portfolio", {})
    weight_dict = port.get("weight_dict", {})

    if weight_dict:
        col_pie, col_rc = st.columns(2)
        with col_pie:
            fig_pie = portfolio_pie(weight_dict)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        with col_rc:
            section_header("Risk Contributions")
            rc = port.get("risk_contributions", {})
            for t, rv in sorted(rc.items(), key=lambda x: -x[1]):
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;margin-bottom:5px">
                        <span style="color:#e6edf3;width:60px;font-size:0.85rem;font-weight:600">{t}</span>
                        <div style="flex:1;background:#21262d;border-radius:4px;height:16px;margin:0 10px">
                            <div style="width:{min(rv*100,100):.1f}%;background:#bc8cff;height:100%;border-radius:4px"></div>
                        </div>
                        <span style="color:#bc8cff;font-size:0.82rem;width:50px;text-align:right">{rv*100:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    render_model_info(result)


# ── Tab 6: Charts ────────────────────────────────────────────────────────
with tab_charts:
    section_header("📉 Technical Charts")

    col_vol, col_dd = st.columns(2)
    with col_vol:
        fig_vol = volatility_chart(ticker)
        if fig_vol:
            st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Volatility chart unavailable.")
    with col_dd:
        fig_dd = drawdown_chart(ticker)
        if fig_dd:
            st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Drawdown chart unavailable.")

    # Full price chart
    fig_price_full = price_chart(ticker, lookback_days=365)
    if fig_price_full:
        st.plotly_chart(fig_price_full, use_container_width=True, config={"displayModeBar": False})
