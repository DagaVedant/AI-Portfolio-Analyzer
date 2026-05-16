"""
backtest.py — Walk-forward backtesting engine for the AI Portfolio Analyzer.

Key design principles:
  - NO look-ahead bias: models only trained on past data at each rebalance date
  - Monthly or weekly rebalancing
  - Transaction costs deducted on every trade
  - Full equity-curve tracking
  - Benchmark (SPY) comparison
  - Performance metrics: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate

Educational disclaimer: Past backtest results do not guarantee future performance.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def get_rebalance_dates(
    index: pd.DatetimeIndex,
    freq: str = "monthly",
) -> List[pd.Timestamp]:
    """Return list of rebalance dates from a DatetimeIndex.

    freq: 'monthly' | 'weekly'
    """
    if freq == "weekly":
        # Every Friday (or last trading day of week)
        return list(index[index.to_series().dt.dayofweek == 4])
    else:
        # Last trading day of each month
        return list(
            index.to_series()
            .groupby([index.year, index.month])
            .last()
            .values
        )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf: float = 0.05,
) -> Dict:
    """Compute a comprehensive set of performance metrics from an equity curve."""
    daily_ret = equity_curve.pct_change().dropna()
    n_days    = len(daily_ret)

    if n_days < 2:
        return {}

    # CAGR
    total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)
    years = n_days / 252
    cagr  = float((1 + total_return) ** (1 / max(years, 0.01)) - 1)

    # Annualised volatility
    ann_vol = float(daily_ret.std() * np.sqrt(252))

    # Sharpe ratio (annualised)
    excess = daily_ret - rf / 252
    sharpe = float(excess.mean() / (excess.std() + 1e-10) * np.sqrt(252))

    # Sortino ratio
    downside_ret  = daily_ret[daily_ret < 0]
    sortino_denom = float(downside_ret.std() * np.sqrt(252)) if len(downside_ret) > 0 else 1e-10
    sortino       = float((daily_ret.mean() * 252 - rf) / (sortino_denom + 1e-10))

    # Maximum drawdown
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / (rolling_max + 1e-10)
    max_dd      = float(drawdown.min())

    # Calmar ratio
    calmar = float(cagr / (abs(max_dd) + 1e-10))

    # Win rate (fraction of positive days)
    win_rate = float((daily_ret > 0).mean())

    metrics = dict(
        total_return=round(total_return, 4),
        cagr=round(cagr, 4),
        ann_volatility=round(ann_vol, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 4),
        calmar_ratio=round(calmar, 4),
        win_rate=round(win_rate, 4),
        n_trading_days=n_days,
    )

    # Benchmark comparison
    if benchmark is not None and len(benchmark) > 1:
        bm_ret     = benchmark.pct_change().dropna()
        bm_aligned = bm_ret.reindex(daily_ret.index).fillna(0)

        if len(bm_aligned) > 1:
            beta        = float(np.cov(daily_ret, bm_aligned)[0, 1] / (bm_aligned.var() + 1e-10))
            alpha_daily = float(daily_ret.mean() - beta * bm_aligned.mean())
            alpha_annual = float(alpha_daily * 252)
            corr        = float(daily_ret.corr(bm_aligned))

            bm_total = float((benchmark.iloc[-1] / benchmark.iloc[0]) - 1)
            bm_years = len(bm_aligned) / 252
            bm_cagr  = float((1 + bm_total) ** (1 / max(bm_years, 0.01)) - 1)

            metrics.update(dict(
                beta=round(beta, 4),
                alpha_annual=round(alpha_annual, 4),
                benchmark_correlation=round(corr, 4),
                benchmark_total_return=round(bm_total, 4),
                benchmark_cagr=round(bm_cagr, 4),
            ))

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward backtesting engine
# ---------------------------------------------------------------------------

class Backtester:
    """Walk-forward backtester.

    At each rebalance date:
      1. Use ALL data up to (but not including) that date to estimate
         returns and covariance (no look-ahead).
      2. Compute portfolio weights via the selected optimizer.
      3. Hold positions until the next rebalance date.
      4. Deduct transaction costs.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        bcfg = cfg.get("backtest", {})
        pcfg = cfg.get("portfolio", {})

        self.start_date       = pd.Timestamp(bcfg.get("start_date", "2021-01-01"))
        self.end_date         = pd.Timestamp(bcfg.get("end_date", "2024-12-31"))
        self.initial_capital  = bcfg.get("initial_capital", 100_000)
        self.rebalance_freq   = bcfg.get("rebalance_freq", "monthly")
        self.transaction_cost = bcfg.get("transaction_cost", 0.001)
        self.benchmark_ticker = bcfg.get("benchmark", "SPY")
        self.rf               = pcfg.get("risk_free_rate", 0.05)
        self.max_w            = pcfg.get("max_weight", 0.40)
        self.min_w            = pcfg.get("min_weight", 0.01)
        self.opt_method       = pcfg.get("method", "max_sharpe")

        # Results populated by run()
        self.equity_curve:    Optional[pd.Series] = None
        self.benchmark_curve: Optional[pd.Series] = None
        self.weight_history:  List[Dict] = []
        self.metrics:         Dict = {}

    def run(
        self,
        price_df: pd.DataFrame,
        benchmark_series: Optional[pd.Series] = None,
        model_preds_history: Optional[Dict[pd.Timestamp, Dict[str, float]]] = None,
    ) -> Dict:
        """Execute the walk-forward backtest.

        Args:
            price_df:             Daily adjusted-close prices, columns = tickers
            benchmark_series:     Benchmark (SPY) daily adjusted-close
            model_preds_history:  Optional {date: {ticker: predicted_return}} for ML-guided allocation

        Returns:
            dict with equity_curve, benchmark_curve, metrics, weight_history
        """
        from src.optimize import optimize_portfolio, compute_expected_returns, compute_covariance

        # Restrict to backtest period
        price_df = price_df.loc[self.start_date: self.end_date]
        if price_df.empty:
            raise ValueError("price_df is empty for the backtest period.")

        tickers         = list(price_df.columns)
        dates           = price_df.index
        rebalance_dates = set(get_rebalance_dates(dates, freq=self.rebalance_freq))

        # Initialise portfolio
        capital          = float(self.initial_capital)
        weights          = np.ones(len(tickers)) / len(tickers)
        current_weights  = weights.copy()

        equity_values = []

        # BUG FIX: the original code used `i >= warmup_days` where `i` is the
        # integer row index in `dates`.  This is wrong when the price_df has
        # gaps (halted stocks, holiday variations) because row 252 may represent
        # fewer than 252 actual trading days of history for some tickers.
        # Instead, count the non-NaN rows in hist_prices at each rebalance to
        # confirm we have enough clean history before optimising.
        warmup_days = 252

        for i, date in enumerate(dates):
            # ── Rebalance ──
            if date in rebalance_dates:
                hist_prices = price_df.iloc[:i]    # strictly past data only

                # Count actual trading days with full data available
                clean_days = hist_prices.dropna().shape[0]
                if clean_days >= warmup_days:
                    # Get model predictions for this date if available
                    mp = None
                    if model_preds_history is not None:
                        past_pred_dates = [d for d in model_preds_history if d <= date]
                        if past_pred_dates:
                            mp = model_preds_history[max(past_pred_dates)]

                    try:
                        opt_result  = optimize_portfolio(hist_prices, self.cfg, model_preds=mp)
                        new_weights = np.array([
                            opt_result["weight_dict"].get(t, 1 / len(tickers)) for t in tickers
                        ])
                    except Exception as e:
                        logger.warning(f"Optimization failed on {date}: {e} — keeping equal weights.")
                        new_weights = np.ones(len(tickers)) / len(tickers)

                    # Transaction cost: proportional to turnover
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    tc_cost  = turnover * self.transaction_cost * capital
                    capital -= tc_cost
                    current_weights = new_weights

                    self.weight_history.append({
                        "date":             date,
                        "weights":          {t: float(current_weights[j]) for j, t in enumerate(tickers)},
                        "transaction_cost": tc_cost,
                    })

            # ── Daily return ──
            if i > 0:
                prev_prices = price_df.iloc[i - 1][tickers].values
                curr_prices = price_df.iloc[i][tickers].values
                valid       = prev_prices > 0
                daily_rets  = np.where(valid, (curr_prices - prev_prices) / prev_prices, 0.0)
                port_ret    = float(current_weights @ daily_rets)
                capital    *= (1 + port_ret)

            equity_values.append(capital)

        self.equity_curve = pd.Series(equity_values, index=dates, name="Portfolio")

        # Benchmark
        if benchmark_series is not None:
            bm = benchmark_series.loc[self.start_date: self.end_date]
            self.benchmark_curve = bm / bm.iloc[0] * self.initial_capital
        else:
            self.benchmark_curve = None

        # Metrics
        self.metrics = compute_performance_metrics(
            self.equity_curve, self.benchmark_curve, rf=self.rf
        )

        logger.info(
            f"Backtest complete | "
            f"CAGR={self.metrics.get('cagr', 0):.2%} | "
            f"Sharpe={self.metrics.get('sharpe_ratio', 0):.2f} | "
            f"MaxDD={self.metrics.get('max_drawdown', 0):.2%}"
        )

        return dict(
            equity_curve=self.equity_curve,
            benchmark_curve=self.benchmark_curve,
            metrics=self.metrics,
            weight_history=self.weight_history,
        )


# ---------------------------------------------------------------------------
# Drawdown series helper
# ---------------------------------------------------------------------------

def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return drawdown (as fraction) at each point in the equity curve."""
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / (rolling_max + 1e-10)


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run_backtest(cfg: Dict) -> Dict:
    """Standalone helper to run a backtest from config."""
    from src.dataset import download_ticker

    tickers          = [t for t in cfg["data"]["tickers"] if t != cfg["backtest"]["benchmark"]]
    benchmark_ticker = cfg["backtest"]["benchmark"]

    price_data = {}
    for t in tickers + [benchmark_ticker]:
        try:
            raw = download_ticker(
                t,
                start=cfg["backtest"]["start_date"],
                end=cfg["backtest"]["end_date"],
                raw_dir=cfg["data"]["raw_dir"],
            )
            price_data[t] = raw["Close"]
        except Exception as e:
            logger.warning(f"Could not load {t}: {e}")

    if not price_data:
        raise RuntimeError("No price data available for backtesting.")

    price_df  = pd.DataFrame({t: v for t, v in price_data.items() if t in tickers}).dropna()
    bm_series = price_data.get(benchmark_ticker)

    bt     = Backtester(cfg)
    result = bt.run(price_df, benchmark_series=bm_series)
    return result


if __name__ == "__main__":
    import argparse
    from src.utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg_    = load_config(args.config)
    result_ = run_backtest(cfg_)
    print("\nBacktest Metrics:")
    for k, v in result_["metrics"].items():
        print(f"  {k}: {v}")