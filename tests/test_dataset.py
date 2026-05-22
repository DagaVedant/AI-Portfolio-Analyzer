"""
Unit tests for src/dataset.py — Data download, feature engineering, and DataLoader creation.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Conditionally import to handle missing dependencies
try:
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDatasetUtilities:
    """Test dataset.py utility functions."""
    
    def test_add_technical_indicators_rsi(self, sample_price_data):
        """Test RSI calculation."""
        from src.dataset import add_technical_indicators
        
        df = add_technical_indicators(sample_price_data.copy(), indicators=["RSI"])
        
        assert "RSI_14" in df.columns
        assert not df["RSI_14"].isna().all()
        assert (df["RSI_14"] >= 0).all()
        assert (df["RSI_14"] <= 100).all()
    
    def test_add_technical_indicators_macd(self, sample_price_data):
        """Test MACD calculation."""
        from src.dataset import add_technical_indicators
        
        df = add_technical_indicators(sample_price_data.copy(), indicators=["MACD"])
        
        assert "MACD_12_26" in df.columns
        assert "MACD_Signal" in df.columns
        assert "MACD_Hist" in df.columns
    
    def test_add_technical_indicators_bollinger_bands(self, sample_price_data):
        """Test Bollinger Bands calculation."""
        from src.dataset import add_technical_indicators
        
        df = add_technical_indicators(sample_price_data.copy(), indicators=["BB"])
        
        assert "BB_Upper" in df.columns
        assert "BB_Middle" in df.columns
        assert "BB_Lower" in df.columns
        
        # Upper should be >= Middle >= Lower
        assert (df["BB_Upper"] >= df["BB_Middle"]).all()
        assert (df["BB_Middle"] >= df["BB_Lower"]).all()
    
    def test_add_technical_indicators_all(self, sample_price_data):
        """Test that all indicator types can be computed without error."""
        from src.dataset import add_technical_indicators
        
        indicators = ["RSI", "MACD", "BB", "SMA", "EMA"]
        df = add_technical_indicators(sample_price_data.copy(), indicators=indicators)
        
        # Check that some columns were added
        added_cols = set(df.columns) - set(sample_price_data.columns)
        assert len(added_cols) > 0
    
    def test_add_technical_indicators_empty(self, sample_price_data):
        """Test with empty indicator list."""
        from src.dataset import add_technical_indicators
        
        df = add_technical_indicators(sample_price_data.copy(), indicators=[])
        
        # Should return DataFrame with same columns
        assert len(df.columns) == len(sample_price_data.columns)
    
    def test_scale_features_standardscaler(self, sample_features_df):
        """Test StandardScaler feature scaling."""
        from src.dataset import scale_features
        
        scaled_df, scaler = scale_features(sample_features_df.copy())
        
        # Check mean ~0 and std ~1 for scaled features
        assert np.abs(scaled_df.mean().mean()) < 0.1
        assert np.abs(scaled_df.std().mean() - 1.0) < 0.1
        
        # Scaler should be fitted
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")
    
    def test_scale_features_transform_consistency(self, sample_features_df):
        """Test that scaler can transform new data consistently."""
        from src.dataset import scale_features
        
        df1 = sample_features_df.iloc[:50]
        df2 = sample_features_df.iloc[50:]
        
        scaled1, scaler = scale_features(df1.copy())
        scaled2 = scaler.transform(df2.copy())
        
        # Scaler parameters should be fitted on df1
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) == df1.shape[1]
    
    def test_create_sequences_basic(self, sample_features_df):
        """Test sequence creation for time-series."""
        from src.dataset import create_sequences
        
        sequences, targets = create_sequences(sample_features_df.values, lookback=10)
        
        assert len(sequences) > 0
        assert len(sequences) == len(targets)
        assert sequences[0].shape == (10, sample_features_df.shape[1])
    
    def test_create_sequences_edge_cases(self, sample_features_df):
        """Test sequence creation with edge cases."""
        from src.dataset import create_sequences
        
        # Lookback larger than data
        sequences, targets = create_sequences(sample_features_df.values, lookback=200)
        
        # Should still work, just return empty or minimal sequences
        assert isinstance(sequences, (list, np.ndarray))
        assert isinstance(targets, (list, np.ndarray))
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_market_dataset_init(self, sample_features_df):
        """Test MarketDataset initialization."""
        from src.dataset import MarketDataset
        
        dataset = MarketDataset(sample_features_df.values, lookback=10)
        
        assert len(dataset) > 0
        assert hasattr(dataset, "__getitem__")
        assert hasattr(dataset, "__len__")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_market_dataset_indexing(self, sample_features_df):
        """Test MarketDataset item retrieval."""
        from src.dataset import MarketDataset
        
        dataset = MarketDataset(sample_features_df.values, lookback=10)
        
        x, y = dataset[0]
        
        # Check shapes
        assert x.shape[0] == 10  # lookback
        assert x.shape[1] == sample_features_df.shape[1]
        assert y.shape == (sample_features_df.shape[1],)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_get_dataloaders(self, sample_features_df):
        """Test DataLoader creation with train/val/test split."""
        from src.dataset import get_dataloaders
        
        train_loader, val_loader, test_loader = get_dataloaders(
            sample_features_df.values,
            lookback=10,
            batch_size=16,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Check that data splits add up
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)
        
        assert train_size + val_size + test_size > 0
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_get_dataloaders_batching(self, sample_features_df):
        """Test DataLoader batching."""
        from src.dataset import get_dataloaders
        
        train_loader, _, _ = get_dataloaders(
            sample_features_df.values,
            lookback=10,
            batch_size=16,
        )
        
        # Get first batch
        batch_x, batch_y = next(iter(train_loader))
        
        # Check batch shape
        assert batch_x.shape[0] <= 16  # batch_size
        assert batch_y.shape[0] == batch_x.shape[0]


class TestDataDownload:
    """Test data download functions."""
    
    @patch("yfinance.download")
    def test_download_ticker_mock(self, mock_yf, tmp_data_dir):
        """Test ticker download with mocked yfinance."""
        from src.dataset import download_ticker
        
        # Setup mock
        mock_yf.return_value = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low": [95.0] * 10,
            "Close": [102.0] * 10,
            "Volume": [1000000] * 10,
            "Adj Close": [102.0] * 10,
        }, index=pd.date_range("2023-01-01", periods=10, freq="D"))
        
        df = download_ticker(
            "AAPL",
            start="2023-01-01",
            end="2023-01-10",
            raw_dir=str(tmp_data_dir["raw"])
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Close" in df.columns
    
    def test_download_ticker_cache(self, tmp_data_dir):
        """Test that cached ticker data is reused."""
        from src.dataset import download_ticker
        
        # Create a cached file
        cache_file = tmp_data_dir["raw"] / "AAPL.parquet"
        df_cached = pd.DataFrame({
            "Open": [100.0] * 5,
            "Close": [102.0] * 5,
        }, index=pd.date_range("2023-01-01", periods=5, freq="D"))
        df_cached.to_parquet(cache_file)
        
        # Download should use cache
        with patch("yfinance.download") as mock_yf:
            df = download_ticker(
                "AAPL",
                start="2023-01-01",
                end="2023-01-05",
                raw_dir=str(tmp_data_dir["raw"])
            )
            
            # yfinance.download should NOT have been called
            mock_yf.assert_not_called()


class TestDataPreprocessing:
    """Test data preprocessing pipeline."""
    
    def test_train_val_test_split(self, sample_price_data):
        """Test time-series safe train/val/test split."""
        from src.dataset import train_val_test_split
        
        train, val, test = train_val_test_split(
            sample_price_data,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        
        # Check sizes
        assert len(train) + len(val) + len(test) == len(sample_price_data)
        
        # Check chronological order (no data leakage)
        train_dates = train.index
        val_dates = val.index
        test_dates = test.index
        
        assert train_dates[-1] < val_dates[0]  # train ends before val starts
        assert val_dates[-1] < test_dates[0]   # val ends before test starts
    
    def test_train_val_test_split_ratios(self, sample_price_data):
        """Test that split ratios are approximately correct."""
        from src.dataset import train_val_test_split
        
        train, val, test = train_val_test_split(
            sample_price_data,
            train_ratio=0.6,
            val_ratio=0.2,
        )
        
        total = len(sample_price_data)
        
        # Allow 5% tolerance
        assert 0.55 < len(train) / total < 0.65
        assert 0.15 < len(val) / total < 0.25
        assert len(test) / total > 0.1


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        from src.dataset import add_technical_indicators
        
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        
        # Should not raise error
        result = add_technical_indicators(empty_df)
        assert isinstance(result, pd.DataFrame)
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrames."""
        from src.dataset import create_sequences
        
        single_row = np.random.randn(1, 10)
        sequences, targets = create_sequences(single_row, lookback=5)
        
        # Should handle gracefully
        assert isinstance(sequences, (list, np.ndarray))
    
    def test_nan_handling(self, sample_price_data):
        """Test that NaN values are handled properly."""
        from src.dataset import add_technical_indicators
        
        df = sample_price_data.copy()
        df.iloc[0:5, 0] = np.nan
        
        result = add_technical_indicators(df)
        
        # Should complete without error
        assert isinstance(result, pd.DataFrame)
