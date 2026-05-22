"""
pytest configuration and fixtures for AI Portfolio Analyzer tests.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Create sample OHLCV price data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    n = len(dates)
    
    data = {
        "Open": np.random.uniform(100, 150, n),
        "High": np.random.uniform(105, 155, n),
        "Low": np.random.uniform(95, 145, n),
        "Close": np.random.uniform(100, 150, n),
        "Volume": np.random.randint(1000000, 10000000, n),
    }
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    # Ensure OHLCV constraints
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
    return df


@pytest.fixture
def sample_multi_ticker_data():
    """Create sample OHLCV data for multiple tickers."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    n = len(dates)
    tickers = ["AAPL", "GOOGL", "MSFT"]
    
    data = {}
    for ticker in tickers:
        df = pd.DataFrame({
            "Open": np.random.uniform(100, 200, n),
            "High": np.random.uniform(105, 205, n),
            "Low": np.random.uniform(95, 195, n),
            "Close": np.random.uniform(100, 200, n),
            "Volume": np.random.randint(1000000, 10000000, n),
            "Ticker": ticker,
        }, index=dates)
        df.index.name = "Date"
        df["High"] = df[["Open", "Close", "High"]].max(axis=1)
        df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
        data[ticker] = df
    
    return data


@pytest.fixture
def sample_features_df():
    """Create sample feature matrix for ML tests."""
    n_samples = 100
    n_features = 20
    
    df = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    return df


@pytest.fixture
def sample_sentiment_articles():
    """Create sample news articles for sentiment testing."""
    articles = [
        {
            "title": "Stock hits all-time high with strong earnings",
            "text": "Positive gains reported quarterly earnings beat analyst expectations",
            "ticker": "AAPL",
            "date": datetime.now() - timedelta(days=1)
        },
        {
            "title": "Market downturn amid recession fears",
            "text": "Stock price falls significantly on macro concerns and layoff announcement",
            "ticker": "AAPL",
            "date": datetime.now() - timedelta(days=2)
        },
        {
            "title": "Trading sideways with mixed sentiment",
            "text": "Stock remains neutral as investors await guidance from management",
            "ticker": "AAPL",
            "date": datetime.now() - timedelta(days=3)
        },
        {
            "title": "Empty article",
            "text": "",
            "ticker": "AAPL",
            "date": datetime.now()
        }
    ]
    return articles


@pytest.fixture
def sample_returns_df():
    """Create sample returns for portfolio optimization tests."""
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    n_days = 252
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")
    returns = pd.DataFrame(
        np.random.randn(n_days, len(tickers)) * 0.02,  # 2% daily volatility
        index=dates,
        columns=tickers
    )
    return returns


@pytest.fixture
def sample_config():
    """Create a minimal config dict for testing."""
    return {
        "data": {
            "tickers": ["AAPL", "GOOGL", "MSFT"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
        },
        "features": {
            "lookback": 20,
            "scale": True,
            "technical_indicators": ["RSI", "MACD", "BB"],
        },
        "model": {
            "type": "lstm",
            "lstm": {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
            },
            "transformer": {
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
            }
        },
        "training": {
            "batch_size": 32,
            "epochs": 10,
            "lr": 0.001,
        },
        "portfolio": {
            "optimization": "max_sharpe",
            "max_weight": 0.3,
            "min_weight": 0.05,
        }
    }


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_yfinance(monkeypatch):
    """Mock yfinance to avoid API calls."""
    def mock_download(*args, **kwargs):
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        n = len(dates)
        return pd.DataFrame({
            "Open": np.random.uniform(100, 150, n),
            "High": np.random.uniform(105, 155, n),
            "Low": np.random.uniform(95, 145, n),
            "Close": np.random.uniform(100, 150, n),
            "Volume": np.random.randint(1000000, 10000000, n),
            "Adj Close": np.random.uniform(100, 150, n),
        }, index=dates)
    
    import yfinance as yf
    monkeypatch.setattr(yf, "download", mock_download)
    return yf


@pytest.fixture
def mock_finbert_pipeline(monkeypatch):
    """Mock FinBERT pipeline to avoid loading transformers."""
    def mock_pipeline(*args, **kwargs):
        def pipeline_fn(text):
            # Return consistent format regardless of text
            return [
                {"label": "positive", "score": 0.6},
                {"label": "negative", "score": 0.2},
                {"label": "neutral", "score": 0.2},
            ]
        return pipeline_fn
    
    try:
        from transformers import pipeline as hf_pipeline
        monkeypatch.setattr("transformers.pipeline", mock_pipeline)
    except ImportError:
        pass
    
    return mock_pipeline


@pytest.fixture
def mock_cvxpy(monkeypatch):
    """Mock CVXPY for optimization tests."""
    mock_cp = MagicMock()
    
    # Setup Variable, Minimize, Problem classes
    def variable(n, nonneg=False):
        var = MagicMock()
        var.value = np.random.dirichlet(np.ones(n))  # Random valid weights
        return var
    
    mock_cp.Variable = variable
    mock_cp.sum = lambda x: x
    mock_cp.Minimize = lambda x: x
    mock_cp.Problem = MagicMock()
    mock_cp.Problem.return_value.solve.return_value = None
    
    try:
        import cvxpy
        monkeypatch.setattr("cvxpy.Variable", variable)
        monkeypatch.setattr("cvxpy.Minimize", mock_cp.Minimize)
        monkeypatch.setattr("cvxpy.Problem", mock_cp.Problem)
    except ImportError:
        pass
    
    return mock_cp


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    return {
        "root": tmp_path,
        "raw": raw_dir,
        "processed": processed_dir,
    }
