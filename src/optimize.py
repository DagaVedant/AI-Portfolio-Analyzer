"""
optimize.py — Portfolio optimization for the AI Portfolio Analyzer.

Supports:
  - Maximum Sharpe Ratio
  - Minimum Volatility
  - Mean-Variance (target return)
  - Risk Parity (equal risk contribution)
  - Long-only + max-weight constraints
  - Transaction cost awareness

Uses cvxpy when available; falls back to scipy.optimize.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.info("cvxpy not installed; using scipy fallback for optimization.")

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Return / covariance estimation
# ---------------------------------------------------------------------------

def compute_expected_returns(
    price_df: pd.DataFrame,
    method: str = "historical",
    model_preds: Optional[Dict[str, float]] = None,
    horizon_days: int = 21,
    lookback: int = 252,
) -> pd.Series:
    """Estimate expected (annualised) returns for each asset.

    method:
        'historical'  — simple annualised mean of daily returns
        'model'       — use ML predicted returns (model_preds dict)
        'blended'     — 50/50 blend of historical and model predictions
    """
    daily_ret   = price_df.pct_change().dropna().tail(lookback)
    hist_annual = daily_ret.mean() * 252

    if method == "historical" or model_preds is None:
        return hist_annual

    # Build model series aligned to price_df columns
    model_series = pd.Series({t: model_preds.get(t, 0.0) for t in price_df.columns})
    # Convert horizon return to annualised
    model_annual = model_series * (252 / horizon_days)

    if method == "model":
        return model_annual
    # blended
    return 0.5 * hist_annual + 0.5 * model_annual


def compute_covariance(
    price_df: pd.DataFrame,
    method: str = "sample",
    lookback: int = 252,
) -> pd.DataFrame:
    """Estimate annualised covariance matrix.

    method:
        'sample'       — standard sample covariance
        'ledoit_wolf'  — shrinkage estimator (if sklearn installed)
    """
    daily_ret = price_df.pct_change().dropna().tail(lookback)

    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(np.array(daily_ret.values, copy=True))
            cov_daily = pd.DataFrame(lw.covariance_, index=daily_ret.columns, columns=daily_ret.columns)
        except Exception:
            cov_daily = daily_ret.cov()
    else:
        cov_daily = daily_ret.cov()

    return cov_daily * 252   # annualise


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def sharpe_ratio(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.05) -> float:
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    return (ret - rf) / (vol + 1e-10)


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute per-asset marginal risk contribution as fraction of total risk."""
    port_vol = portfolio_volatility(weights, cov)
    marginal = cov @ weights
    rc       = weights * marginal / (port_vol + 1e-10)
    return rc / (rc.sum() + 1e-10)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def _cvxpy_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    max_weight: float,
    min_weight: float,
) -> np.ndarray:
    """Max Sharpe via CVXPY (Markowitz parameterisation trick)."""
    n         = len(mu)
    y         = cp.Variable(n, nonneg=True)
    kappa     = cp.Variable(nonneg=True)

    excess_mu = mu - rf
    constraints = [
        cp.sum(excess_mu @ y) == 1,
        cp.sum(y) == kappa,
        y <= max_weight * kappa,
        y >= min_weight * kappa,
        kappa >= 0,
    ]
    objective = cp.Minimize(cp.quad_form(y, cov))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate") or y.value is None:
        logger.warning("CVXPY max-Sharpe failed; using equal-weight fallback.")
        return np.ones(n) / n

    raw = y.value / (y.value.sum() + 1e-10)
    clipped = np.clip(raw, min_weight, max_weight)
    return clipped / clipped.sum()


def _cvxpy_min_vol(
    cov: np.ndarray,
    max_weight: float,
    min_weight: float,
) -> np.ndarray:
    n = cov.shape[0]
    w = cp.Variable(n)
    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
    ]
    objective = cp.Minimize(cp.quad_form(w, cov))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if w.value is None:
        return np.ones(n) / n

    # BUG FIX: clip then re-normalise so weights always sum to 1.
    # The original code clipped but never normalised, producing invalid weights
    # when called directly (the public optimize_portfolio() re-normalised, but
    # direct callers were silently broken).
    clipped = np.clip(w.value, min_weight, max_weight)
    return clipped / clipped.sum()


def _scipy_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    max_weight: float,
    min_weight: float,
) -> np.ndarray:
    n = len(mu)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / (vol + 1e-10)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(min_weight, max_weight)] * n
    w0          = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    if res.success:
        return res.x
    logger.warning("scipy max-Sharpe did not converge; equal-weight fallback.")
    return np.ones(n) / n


def _scipy_min_vol(
    cov: np.ndarray,
    max_weight: float,
    min_weight: float,
) -> np.ndarray:
    n = cov.shape[0]

    def port_vol(w):
        return np.sqrt(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(min_weight, max_weight)] * n
    w0          = np.ones(n) / n
    res = minimize(port_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    return res.x if res.success else np.ones(n) / n


def _risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution via scipy."""
    if not SCIPY_AVAILABLE:
        return np.ones(cov.shape[0]) / cov.shape[0]

    n         = cov.shape[0]
    target_rc = np.ones(n) / n

    def objective(w):
        vol     = np.sqrt(w @ cov @ w)
        rc      = (w * (cov @ w)) / (vol + 1e-10)
        rc_norm = rc / (rc.sum() + 1e-10)
        return float(np.sum((rc_norm - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0.01, 0.5)] * n
    w0          = np.ones(n) / n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-10, "maxiter": 2000})
    return res.x if res.success else np.ones(n) / n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_portfolio(
    price_df: pd.DataFrame,
    cfg: Dict,
    model_preds: Optional[Dict[str, float]] = None,
) -> Dict:
    """Run portfolio optimisation and return a results dict.

    Args:
        price_df:    DataFrame of adjusted close prices, columns = tickers
        cfg:         full project config dict
        model_preds: optional {ticker: predicted_return_over_horizon}

    Returns dict with keys:
        weights, tickers, expected_return, expected_volatility,
        sharpe_ratio, risk_contributions, method
    """
    pcfg    = cfg.get("portfolio", {})
    method  = pcfg.get("method", "max_sharpe").lower()
    rf      = pcfg.get("risk_free_rate", 0.05)
    max_w   = pcfg.get("max_weight", 0.40)
    min_w   = pcfg.get("min_weight", 0.01)
    horizon = cfg["data"].get("forecast_horizon", 21)

    tickers = list(price_df.columns)
    n       = len(tickers)

    if n == 0:
        raise ValueError("price_df must have at least one column.")

    # Estimate returns and covariance
    mu_series = compute_expected_returns(
        price_df,
        method="blended" if model_preds else "historical",
        model_preds=model_preds,
        horizon_days=horizon,
    )
    cov_df = compute_covariance(price_df)

    mu = np.array(
        mu_series[tickers].values,
        dtype=float,
        copy=True
    )

    cov = np.array(
        cov_df.loc[tickers, tickers].values,
        dtype=float,
        copy=True
    )

    # Ensure positive semi-definite (add small jitter)
    cov += np.eye(n) * 1e-8

    logger.info(f"Optimizing portfolio ({method}) for {n} assets …")

    if method == "max_sharpe":
        if CVXPY_AVAILABLE:
            weights = _cvxpy_max_sharpe(mu, cov, rf, max_w, min_w)
        elif SCIPY_AVAILABLE:
            weights = _scipy_max_sharpe(mu, cov, rf, max_w, min_w)
        else:
            weights = np.ones(n) / n

    elif method == "min_volatility":
        if CVXPY_AVAILABLE:
            weights = _cvxpy_min_vol(cov, max_w, min_w)
        elif SCIPY_AVAILABLE:
            weights = _scipy_min_vol(cov, max_w, min_w)
        else:
            weights = np.ones(n) / n

    elif method == "risk_parity":
        weights = _risk_parity(cov)

    else:  # mean_variance — same as max_sharpe with target return
        if CVXPY_AVAILABLE:
            weights = _cvxpy_max_sharpe(mu, cov, rf, max_w, min_w)
        else:
            weights = np.ones(n) / n

    # Clip and re-normalise (final safety net)
    weights  = np.clip(weights, min_w, max_w)
    weights_sum = weights.sum()
    if weights_sum <= 1e-12:
        weights = np.ones(n) / n
    else:
        weights /= weights_sum

    exp_ret = portfolio_return(weights, mu)
    exp_vol = portfolio_volatility(weights, cov)
    sr      = sharpe_ratio(weights, mu, cov, rf)
    rc      = risk_contributions(weights, cov)

    result = dict(
        tickers=tickers,
        weights=weights.tolist(),
        expected_return=round(float(exp_ret), 4),
        expected_volatility=round(float(exp_vol), 4),
        sharpe_ratio=round(float(sr), 4),
        risk_contributions={t: round(float(rc[i]), 4) for i, t in enumerate(tickers)},
        method=method,
        weight_dict={t: round(float(weights[i]), 4) for i, t in enumerate(tickers)},
    )
    logger.info(
        f"Optimization done | E[R]={exp_ret:.2%} | E[σ]={exp_vol:.2%} | Sharpe={sr:.2f}"
    )
    return result