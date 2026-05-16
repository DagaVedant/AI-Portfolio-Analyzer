"""
dashboard_utils.py — Chart generation and formatting helpers for the dashboard.

All Plotly figures use a consistent dark theme.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

DARK_BG = "#0d1117"
CARD_BG = "#161b22"
ACCENT_GREEN = "#00d4aa"
ACCENT_RED = "#ff6b6b"
ACCENT_YELLOW = "#ffd166"
ACCENT_BLUE = "#58a6ff"
ACCENT_PURPLE = "#bc8cff"
TEXT_COLOR = "#e6edf3"
MUTED_COLOR = "#8b949e"
GRID_COLOR = "#21262d"

BASE_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="'IBM Plex Mono', monospace", color=TEXT_COLOR, size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT_COLOR, bordercolor=ACCENT_GREEN),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR),
)


def _apply_base(fig, title: str = "", height: int = 350):
    fig.update_layout(**BASE_LAYOUT, title=dict(text=title, font=dict(color=TEXT_COLOR, size=14)), height=height)
    return fig


# ---------------------------------------------------------------------------
# 1. Price chart with SMA overlays
# ---------------------------------------------------------------------------

def price_chart(ticker: str, lookback_days: int = 180) -> Optional[object]:
    if not PLOTLY_OK:
        return None
    try:
        import yfinance as yf
        raw = yf.download(ticker, period=f"{lookback_days}d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            return None

        sma20 = raw["Close"].rolling(20).mean()
        sma50 = raw["Close"].rolling(50).mean()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=raw.index, open=raw["Open"], high=raw["High"],
            low=raw["Low"], close=raw["Close"],
            name="OHLC",
            increasing_line_color=ACCENT_GREEN,
            decreasing_line_color=ACCENT_RED,
        ))
        fig.add_trace(go.Scatter(x=raw.index, y=sma20, name="SMA 20",
                                 line=dict(color=ACCENT_YELLOW, width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=raw.index, y=sma50, name="SMA 50",
                                 line=dict(color=ACCENT_PURPLE, width=1.5, dash="dot")))
        _apply_base(fig, f"{ticker} — Price History", height=420)
        fig.update_layout(xaxis_rangeslider_visible=False)
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Forecast chart (current price + confidence cone)
# ---------------------------------------------------------------------------

def forecast_chart(result: Dict) -> Optional[object]:
    if not PLOTLY_OK:
        return None
    try:
        import yfinance as yf

        ticker = result["ticker"]
        horizon = result.get("forecast_horizon_days", 21)
        price_low = result.get("price_forecast_low", 0)
        price_mid = result.get("price_forecast_mid", 0)
        price_high = result.get("price_forecast_high", 0)
        current = result.get("current_price", price_mid)

        # Historical tail
        raw = yf.download(ticker, period="60d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        hist_dates = raw.index[-30:]
        hist_prices = raw["Close"].iloc[-30:]

        # Forecast dates
        last_date = pd.Timestamp.now().normalize()
        future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

        # Build cone by interpolating
        n = len(future_dates)
        lows = np.linspace(current, price_low, n)
        highs = np.linspace(current, price_high, n)
        mids = np.linspace(current, price_mid, n)

        fig = go.Figure()
        # Historical
        fig.add_trace(go.Scatter(x=hist_dates, y=hist_prices,
                                 name="Historical", line=dict(color=TEXT_COLOR, width=2)))
        # Forecast band
        fig.add_trace(go.Scatter(
            x=np.concatenate([future_dates, future_dates[::-1]]),
            y=np.concatenate([highs, lows[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 212, 170, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence",
        ))
        # Midline
        fig.add_trace(go.Scatter(x=future_dates, y=mids, name="Forecast",
                                 line=dict(color=ACCENT_GREEN, width=2, dash="dash")))
        _apply_base(fig, f"{ticker} — {horizon}-Day Price Forecast", height=380)
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. Volatility chart (rolling 20-day)
# ---------------------------------------------------------------------------

def volatility_chart(ticker: str) -> Optional[object]:
    if not PLOTLY_OK:
        return None
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            return None

        daily_ret = raw["Close"].pct_change()
        vol_20 = daily_ret.rolling(20).std() * np.sqrt(252)
        vol_60 = daily_ret.rolling(60).std() * np.sqrt(252)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw.index, y=vol_20 * 100,
                                 name="20-Day Vol %", line=dict(color=ACCENT_YELLOW, width=2)))
        fig.add_trace(go.Scatter(x=raw.index, y=vol_60 * 100,
                                 name="60-Day Vol %", line=dict(color=ACCENT_PURPLE, width=1.5, dash="dot")))
        fig.add_hline(y=20, line=dict(color=MUTED_COLOR, dash="dot", width=1),
                      annotation_text="20% baseline", annotation_font_color=MUTED_COLOR)
        _apply_base(fig, f"{ticker} — Rolling Volatility (%)", height=300)
        fig.update_layout(yaxis_title="Annualised Vol (%)")
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. Drawdown chart
# ---------------------------------------------------------------------------

def drawdown_chart(ticker: str) -> Optional[object]:
    if not PLOTLY_OK:
        return None
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            return None

        prices = raw["Close"]
        dd = (prices - prices.cummax()) / prices.cummax() * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=raw.index, y=dd,
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.18)",
            line=dict(color=ACCENT_RED, width=1.5),
            name="Drawdown %",
        ))
        _apply_base(fig, f"{ticker} — Drawdown (%)", height=300)
        fig.update_layout(yaxis_title="Drawdown (%)")
        return fig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 5. Sentiment gauge
# ---------------------------------------------------------------------------

def sentiment_gauge(score: float, label: str = "") -> Optional[object]:
    if not PLOTLY_OK:
        return None

    score_pct = (score + 1) / 2 * 100   # map -1..1 to 0..100

    if score > 0.15:
        bar_color = ACCENT_GREEN
    elif score < -0.15:
        bar_color = ACCENT_RED
    else:
        bar_color = ACCENT_YELLOW

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 3),
        delta={"reference": 0, "valueformat": ".3f"},
        number={"font": {"color": bar_color, "size": 28}},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": MUTED_COLOR,
                     "tickfont": {"color": MUTED_COLOR}},
            "bar": {"color": bar_color, "thickness": 0.25},
            "bgcolor": CARD_BG,
            "bordercolor": GRID_COLOR,
            "steps": [
                {"range": [-1, -0.15], "color": "rgba(255,107,107,0.15)"},
                {"range": [-0.15, 0.15], "color": "rgba(255,209,102,0.10)"},
                {"range": [0.15, 1], "color": "rgba(0,212,170,0.15)"},
            ],
            "threshold": {
                "line": {"color": TEXT_COLOR, "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": label or "Sentiment Score", "font": {"color": MUTED_COLOR, "size": 13}},
    ))
    _apply_base(fig, height=250)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig


# ---------------------------------------------------------------------------
# 6. Portfolio allocation pie
# ---------------------------------------------------------------------------

def portfolio_pie(weight_dict: Dict[str, float]) -> Optional[object]:
    if not PLOTLY_OK or not weight_dict:
        return None

    labels = list(weight_dict.keys())
    values = [v * 100 for v in weight_dict.values()]
    colors = [
        ACCENT_GREEN, ACCENT_BLUE, ACCENT_YELLOW,
        ACCENT_PURPLE, ACCENT_RED, "#fd7e14", "#20c997",
    ][:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors, line=dict(color=CARD_BG, width=2)),
        textfont=dict(color=TEXT_COLOR),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    _apply_base(fig, "Portfolio Allocation", height=300)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", font=dict(size=11)),
        annotations=[dict(
            text=f"<b>{len(labels)}</b><br>assets",
            x=0.5, y=0.5,
            font=dict(color=TEXT_COLOR, size=14),
            showarrow=False,
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Risk breakdown bar
# ---------------------------------------------------------------------------

def risk_breakdown_chart(result: Dict) -> Optional[object]:
    if not PLOTLY_OK:
        return None

    labels = ["Annual Vol", "VaR (1d)", "CVaR (1d)", "Max DD", "Downside Prob"]
    raw_vals = [
        abs(result.get("ann_volatility", 0)),
        abs(result.get("var_1d", 0)),
        abs(result.get("cvar_1d", 0)),
        abs(result.get("max_drawdown", 0)),
        abs(result.get("pred_downside_prob", 0)),
    ]
    pct_vals = [v * 100 for v in raw_vals]
    colors = [
        ACCENT_YELLOW if v < 20 else (ACCENT_RED if v > 40 else "#fd7e14")
        for v in pct_vals
    ]

    fig = go.Figure(go.Bar(
        x=labels,
        y=pct_vals,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in pct_vals],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR),
    ))
    _apply_base(fig, "Risk Breakdown (%)", height=300)
    fig.update_layout(yaxis_title="%", showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# 8. Backtest equity curve
# ---------------------------------------------------------------------------

def backtest_chart(equity_curve: pd.Series, benchmark_curve: Optional[pd.Series] = None) -> Optional[object]:
    if not PLOTLY_OK or equity_curve is None:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve,
        name="Portfolio", line=dict(color=ACCENT_GREEN, width=2),
    ))
    if benchmark_curve is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_curve.index, y=benchmark_curve,
            name="Benchmark", line=dict(color=ACCENT_BLUE, width=1.5, dash="dot"),
        ))
    _apply_base(fig, "Backtest — Equity Curve", height=380)
    fig.update_layout(yaxis_title="Portfolio Value ($)")
    return fig


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pct(v: float, decimals: int = 2) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.{decimals}f}%"


def fmt_price(v: float) -> str:
    return f"${v:,.2f}"


def fmt_num(v: float, decimals: int = 2) -> str:
    if abs(v) >= 1e9:
        return f"{v/1e9:.{decimals}f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.{decimals}f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.{decimals}f}K"
    return f"{v:.{decimals}f}"


def sentiment_label_color(label: str) -> str:
    mapping = {"bullish": ACCENT_GREEN, "bearish": ACCENT_RED, "neutral": ACCENT_YELLOW}
    return mapping.get(label.lower(), MUTED_COLOR)


def outlook_emoji(outlook: str) -> str:
    return {"Bullish": "📈", "Bearish": "📉", "Neutral": "➡️"}.get(outlook, "")
