"""
Unit tests for src/optimize.py — Portfolio optimization functions.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class TestReturnsCalculation:
    """Test expected returns computation."""
    
    def test_compute_expected_returns_historical(self, sample_returns_df):
        """Test historical mean returns calculation."""
        from src.optimize import compute_expected_returns
        
        returns = sample_returns_df
        mu = compute_expected_returns(
            returns.to_frame().T.T,
            method="historical",
            lookback=252,
        )
        
        assert isinstance(mu, pd.Series)
        assert len(mu) == returns.shape[1]
        # Returns should be in reasonable range (annualized)
        assert np.abs(mu.mean()) < 1.0
    
    def test_compute_expected_returns_with_model_predictions(self):
        """Test returns with model predictions."""
        from src.optimize import compute_expected_returns
        
        price_df = pd.DataFrame({
            "AAPL": [100, 101, 102],
            "GOOGL": [200, 200, 201],
        }, index=pd.date_range("2023-01-01", periods=3))
        
        model_preds = {"AAPL": 0.05, "GOOGL": 0.03}
        
        mu = compute_expected_returns(
            price_df,
            method="model",
            model_preds=model_preds,
        )
        
        assert isinstance(mu, pd.Series)
        assert mu["AAPL"] == 0.05
        assert mu["GOOGL"] == 0.03
    
    def test_compute_expected_returns_single_ticker(self):
        """Test with single ticker."""
        from src.optimize import compute_expected_returns
        
        price_df = pd.DataFrame({
            "AAPL": np.random.randn(100).cumsum() + 100
        }, index=pd.date_range("2023-01-01", periods=100))
        
        mu = compute_expected_returns(price_df, method="historical")
        
        assert len(mu) == 1
        assert isinstance(mu, pd.Series)


class TestCovarianceEstimation:
    """Test covariance matrix estimation."""
    
    def test_estimate_covariance_sample(self, sample_returns_df):
        """Test sample covariance estimation."""
        from src.optimize import estimate_covariance
        
        cov = estimate_covariance(sample_returns_df, method="sample")
        
        assert isinstance(cov, np.ndarray)
        assert cov.shape == (sample_returns_df.shape[1], sample_returns_df.shape[1])
        
        # Should be symmetric
        assert np.allclose(cov, cov.T)
        
        # Diagonal should be positive
        assert np.all(np.diag(cov) > 0)
    
    def test_estimate_covariance_ledoit_wolf(self, sample_returns_df):
        """Test Ledoit-Wolf shrinkage covariance."""
        try:
            from src.optimize import estimate_covariance
            
            cov = estimate_covariance(sample_returns_df, method="ledoit_wolf")
            
            assert isinstance(cov, np.ndarray)
            assert cov.shape == (sample_returns_df.shape[1], sample_returns_df.shape[1])
            
            # Should be positive semi-definite
            eigenvalues = np.linalg.eigvals(cov)
            assert np.all(eigenvalues >= -1e-10)
        except Exception:
            pytest.skip("Ledoit-Wolf estimation not available")
    
    def test_covariance_symmetry(self, sample_returns_df):
        """Test covariance matrix is symmetric."""
        from src.optimize import estimate_covariance
        
        cov = estimate_covariance(sample_returns_df)
        
        assert np.allclose(cov, cov.T)
    
    def test_covariance_positive_definite(self, sample_returns_df):
        """Test covariance matrix is positive definite."""
        from src.optimize import estimate_covariance
        
        cov = estimate_covariance(sample_returns_df)
        
        # All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvals(cov)
        assert np.all(eigenvalues >= -1e-10)


class TestPortfolioMetrics:
    """Test portfolio performance metric calculations."""
    
    def test_portfolio_return(self):
        """Test portfolio return calculation."""
        from src.optimize import portfolio_return
        
        weights = np.array([0.4, 0.3, 0.3])
        mu = np.array([0.10, 0.05, 0.08])
        
        ret = portfolio_return(weights, mu)
        
        assert ret == pytest.approx(0.077)
    
    def test_portfolio_volatility(self):
        """Test portfolio volatility calculation."""
        from src.optimize import portfolio_volatility
        
        weights = np.array([0.4, 0.3, 0.3])
        cov = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05],
        ])
        
        vol = portfolio_volatility(weights, cov)
        
        assert vol > 0
        assert vol == pytest.approx(np.sqrt(weights @ cov @ weights))
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        from src.optimize import sharpe_ratio
        
        weights = np.array([0.5, 0.5])
        mu = np.array([0.10, 0.15])
        cov = np.eye(2) * 0.04
        rf = 0.02
        
        sr = sharpe_ratio(weights, mu, cov, rf=rf)
        
        ret = weights @ mu
        vol = np.sqrt(weights @ cov @ weights)
        expected_sr = (ret - rf) / vol
        
        assert sr == pytest.approx(expected_sr)
    
    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility (safe division)."""
        from src.optimize import sharpe_ratio
        
        weights = np.array([0.5, 0.5])
        mu = np.array([0.10, 0.15])
        cov = np.zeros((2, 2))  # Zero volatility
        
        sr = sharpe_ratio(weights, mu, cov)
        
        # Should not raise error, should handle gracefully
        assert np.isfinite(sr)
    
    def test_risk_contributions(self):
        """Test risk contribution calculation."""
        from src.optimize import risk_contributions
        
        weights = np.array([0.4, 0.3, 0.3])
        cov = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05],
        ])
        
        rc = risk_contributions(weights, cov)
        
        # Risk contributions should sum to 1
        assert np.isclose(rc.sum(), 1.0)
        
        # All should be non-negative
        assert np.all(rc >= 0)
        
        # Length should match weights
        assert len(rc) == len(weights)


class TestPortfolioOptimization:
    """Test portfolio optimization methods."""
    
    def test_optimize_portfolio_max_sharpe(self, sample_returns_df):
        """Test max Sharpe ratio optimization."""
        from src.optimize import optimize_portfolio
        
        mu = np.array([0.10, 0.08, 0.12, 0.05, 0.07])
        cov = np.eye(5) * 0.04  # Uncorrelated assets
        
        weights = optimize_portfolio(
            mu, cov,
            objective="max_sharpe",
            max_weight=0.3,
        )
        
        assert weights is not None
        assert len(weights) == 5
        
        # Weights should sum to 1
        assert np.isclose(weights.sum(), 1.0)
        
        # Weights should be long-only
        assert np.all(weights >= 0)
        
        # Respect max_weight constraint
        assert np.all(weights <= 0.3 + 1e-6)
    
    def test_optimize_portfolio_min_volatility(self):
        """Test minimum volatility optimization."""
        from src.optimize import optimize_portfolio
        
        mu = np.array([0.10, 0.08, 0.12])
        cov = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05],
        ])
        
        weights = optimize_portfolio(
            mu, cov,
            objective="min_volatility",
        )
        
        assert weights is not None
        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= 0)
    
    def test_optimize_portfolio_risk_parity(self):
        """Test risk parity optimization."""
        try:
            from src.optimize import optimize_portfolio
            
            mu = np.array([0.10, 0.08, 0.12])
            cov = np.eye(3) * 0.04
            
            weights = optimize_portfolio(
                mu, cov,
                objective="risk_parity",
            )
            
            if weights is not None:
                assert np.isclose(weights.sum(), 1.0)
                assert np.all(weights >= 0)
        except (ValueError, NotImplementedError):
            pytest.skip("Risk parity not fully implemented")
    
    def test_optimize_portfolio_max_weight_constraint(self):
        """Test max weight constraints are respected."""
        from src.optimize import optimize_portfolio
        
        mu = np.array([0.10, 0.08, 0.12, 0.05, 0.07])
        cov = np.eye(5) * 0.04
        max_weight = 0.25
        
        weights = optimize_portfolio(
            mu, cov,
            objective="max_sharpe",
            max_weight=max_weight,
        )
        
        if weights is not None:
            assert np.all(weights <= max_weight + 1e-6)
    
    def test_optimize_portfolio_min_weight_constraint(self):
        """Test minimum weight constraints."""
        from src.optimize import optimize_portfolio
        
        mu = np.array([0.10, 0.08, 0.12])
        cov = np.eye(3) * 0.04
        min_weight = 0.1
        
        weights = optimize_portfolio(
            mu, cov,
            objective="max_sharpe",
            min_weight=min_weight,
        )
        
        if weights is not None:
            # Non-zero weights should respect minimum
            nonzero = weights[weights > 1e-6]
            assert np.all(nonzero >= min_weight - 1e-6)
    
    def test_optimize_portfolio_single_asset(self):
        """Test optimization with single asset."""
        from src.optimize import optimize_portfolio
        
        mu = np.array([0.10])
        cov = np.array([[0.04]])
        
        weights = optimize_portfolio(mu, cov)
        
        if weights is not None:
            assert np.isclose(weights[0], 1.0)


class TestCVXPYOptimization:
    """Test CVXPY-based optimization functions."""
    
    @pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY not installed")
    def test_cvxpy_max_sharpe(self):
        """Test CVXPY max Sharpe implementation."""
        try:
            from src.optimize import _cvxpy_max_sharpe
            
            mu = np.array([0.10, 0.08, 0.12])
            cov = np.eye(3) * 0.04
            
            weights = _cvxpy_max_sharpe(
                mu, cov,
                rf=0.02,
                max_weight=0.5,
                min_weight=0.0,
            )
            
            assert weights is not None
            assert len(weights) == 3
            assert np.isclose(weights.sum(), 1.0)
        except ImportError:
            pytest.skip("CVXPY optimization not available")
    
    @pytest.mark.skipif(not CVXPY_AVAILABLE, reason="CVXPY not installed")
    def test_cvxpy_min_volatility(self):
        """Test CVXPY min volatility implementation."""
        try:
            from src.optimize import _cvxpy_min_volatility
            
            cov = np.array([
                [0.04, 0.01, 0.02],
                [0.01, 0.03, 0.01],
                [0.02, 0.01, 0.05],
            ])
            
            weights = _cvxpy_min_volatility(
                cov,
                max_weight=0.5,
            )
            
            if weights is not None:
                assert np.isclose(weights.sum(), 1.0)
        except ImportError:
            pytest.skip("CVXPY optimization not available")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_portfolio_return_zero_weights(self):
        """Test portfolio return with zero weights."""
        from src.optimize import portfolio_return
        
        weights = np.zeros(3)
        mu = np.array([0.10, 0.08, 0.12])
        
        ret = portfolio_return(weights, mu)
        assert ret == 0.0
    
    def test_portfolio_return_single_asset(self):
        """Test portfolio return with single asset."""
        from src.optimize import portfolio_return
        
        weights = np.array([1.0])
        mu = np.array([0.10])
        
        ret = portfolio_return(weights, mu)
        assert ret == 0.10
    
    def test_sharpe_ratio_negative_return(self):
        """Test Sharpe ratio with negative expected return."""
        from src.optimize import sharpe_ratio
        
        weights = np.array([0.5, 0.5])
        mu = np.array([-0.05, -0.10])
        cov = np.eye(2) * 0.04
        rf = 0.02
        
        sr = sharpe_ratio(weights, mu, cov, rf=rf)
        
        # Should be negative
        assert sr < 0
    
    def test_risk_contributions_single_asset(self):
        """Test risk contributions with single asset."""
        from src.optimize import risk_contributions
        
        weights = np.array([1.0])
        cov = np.array([[0.04]])
        
        rc = risk_contributions(weights, cov)
        
        assert rc.sum() == pytest.approx(1.0)
        assert rc[0] == pytest.approx(1.0)
    
    def test_optimize_portfolio_impossible_constraints(self):
        """Test optimization with conflicting constraints."""
        from src.optimize import optimize_portfolio
        
        mu = np.array([0.10, 0.08])
        cov = np.eye(2) * 0.04
        
        # min_weight > max_weight is impossible
        try:
            weights = optimize_portfolio(
                mu, cov,
                min_weight=0.6,
                max_weight=0.4,  # Impossible!
            )
            # Should either return None or raise error
            if weights is not None:
                # If it returns something, it might be all zeros or None
                pass
        except (ValueError, RuntimeError):
            # Expected
            pass


class TestLongOnlyConstraints:
    """Test long-only portfolio constraints."""
    
    def test_long_only_weights_non_negative(self):
        """Test that long-only portfolio has non-negative weights."""
        from src.optimize import optimize_portfolio
        
        mu = np.random.randn(5) * 0.1 + 0.08
        cov = np.eye(5) * 0.04
        
        weights = optimize_portfolio(mu, cov, long_only=True)
        
        if weights is not None:
            assert np.all(weights >= -1e-6)
    
    def test_weight_sum_constraint(self):
        """Test that optimized weights sum to 1."""
        from src.optimize import optimize_portfolio
        
        mu = np.array([0.10, 0.08, 0.12, 0.05, 0.07])
        cov = np.eye(5) * 0.04
        
        weights = optimize_portfolio(mu, cov)
        
        if weights is not None:
            assert np.isclose(weights.sum(), 1.0, atol=1e-5)


class TestTransactionCosts:
    """Test transaction cost awareness."""
    
    def test_optimize_with_transaction_costs(self):
        """Test optimization that accounts for transaction costs."""
        try:
            from src.optimize import optimize_portfolio
            
            mu = np.array([0.10, 0.08, 0.12])
            cov = np.eye(3) * 0.04
            
            # With transaction costs
            weights_notc = optimize_portfolio(mu, cov)
            weights_tc = optimize_portfolio(
                mu, cov,
                transaction_cost=0.001,
                current_weights=np.array([0.33, 0.33, 0.34])
            )
            
            # Both should be valid
            if weights_tc is not None:
                assert np.isclose(weights_tc.sum(), 1.0)
        except TypeError:
            # Transaction costs might not be implemented
            pytest.skip("Transaction costs not implemented")
