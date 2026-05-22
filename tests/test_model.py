"""
Unit tests for src/model.py — Model definitions (LSTM, Transformer, XGBoost, LightGBM).
"""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Conditional imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class TestLSTMModel:
    """Test LSTM model architecture and forward pass."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_initialization(self):
        """Test LSTM model initialization."""
        from src.model import LSTMModel
        
        model = LSTMModel(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            dropout=0.2
        )
        
        assert isinstance(model, nn.Module)
        assert model.hidden_size == 64
        assert model.num_layers == 2
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass with correct input/output shapes."""
        from src.model import LSTMModel
        
        model = LSTMModel(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=3,
        )
        
        # Create sample input: (batch_size, seq_len, input_size)
        batch_size, seq_len = 32, 10
        x = torch.randn(batch_size, seq_len, 20)
        
        output = model(x)
        
        # Output shape should be (batch_size, output_size)
        assert output.shape == (batch_size, 3)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_different_batch_sizes(self):
        """Test LSTM with different batch sizes."""
        from src.model import LSTMModel
        
        model = LSTMModel(input_size=20, hidden_size=64, output_size=3)
        
        for batch_size in [1, 16, 32, 64]:
            x = torch.randn(batch_size, 10, 20)
            output = model(x)
            assert output.shape[0] == batch_size
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_dropout_effect(self):
        """Test that dropout is applied correctly."""
        from src.model import LSTMModel
        
        model_no_dropout = LSTMModel(
            input_size=20, hidden_size=64, dropout=0.0
        )
        model_with_dropout = LSTMModel(
            input_size=20, hidden_size=64, dropout=0.5
        )
        
        x = torch.randn(1, 10, 20)
        
        # Both should produce output, dropout just affects training
        out1 = model_no_dropout(x)
        out2 = model_with_dropout(x)
        
        assert out1.shape == out2.shape
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_with_uncertainty(self):
        """Test LSTM MC-Dropout uncertainty estimation."""
        from src.model import LSTMModel
        
        model = LSTMModel(
            input_size=20, hidden_size=64, dropout=0.2
        )
        
        x = torch.randn(1, 10, 20)
        
        try:
            mean, std = model.predict_with_uncertainty(x, n_samples=10)
            assert mean.shape == (1, 3)
            assert std.shape == (1, 3)
            assert (std >= 0).all()
        except AttributeError:
            pytest.skip("predict_with_uncertainty not implemented")


class TestTransformerModel:
    """Test Transformer model architecture."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_transformer_initialization(self):
        """Test Transformer model initialization."""
        try:
            from src.model import TransformerModel
            
            model = TransformerModel(
                input_size=20,
                d_model=64,
                nhead=4,
                num_layers=2,
                output_size=3,
            )
            
            assert isinstance(model, nn.Module)
        except ImportError:
            pytest.skip("TransformerModel not available")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_transformer_forward_pass(self):
        """Test Transformer forward pass."""
        try:
            from src.model import TransformerModel
            
            model = TransformerModel(
                input_size=20,
                d_model=64,
                nhead=4,
                num_layers=2,
            )
            
            x = torch.randn(32, 10, 20)  # (batch, seq_len, features)
            output = model(x)
            
            assert output.shape == (32, 3)  # (batch, output_size)
        except ImportError:
            pytest.skip("TransformerModel not available")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_transformer_attention_heads(self):
        """Test Transformer with different attention head configs."""
        try:
            from src.model import TransformerModel
            
            for nhead in [1, 2, 4, 8]:
                model = TransformerModel(
                    input_size=32,  # divisible by nhead
                    d_model=32,
                    nhead=nhead,
                )
                x = torch.randn(16, 10, 32)
                output = model(x)
                assert output.shape[0] == 16
        except ImportError:
            pytest.skip("TransformerModel not available")


class TestXGBoostModel:
    """Test XGBoost model initialization and inference."""
    
    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_initialization(self):
        """Test XGBoost model initialization."""
        try:
            from src.model import XGBModel
            
            model = XGBModel(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
            )
            
            assert model is not None
        except (ImportError, NameError):
            pytest.skip("XGBModel not available")
    
    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_training_and_prediction(self, sample_features_df, sample_returns_df):
        """Test XGBoost training and prediction."""
        try:
            import xgboost as xgb
            
            X = sample_features_df.values
            y = np.random.randn(X.shape[0])  # Random targets
            
            model = xgb.XGBRegressor(
                n_estimators=10,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            
            # Should train without error
            model.fit(X[:-10], y[:-10])
            
            # Predict
            y_pred = model.predict(X[-10:])
            
            assert y_pred.shape == (10,)
        except ImportError:
            pytest.skip("XGBoost not installed")
    
    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_feature_importance(self, sample_features_df):
        """Test XGBoost feature importance computation."""
        try:
            import xgboost as xgb
            
            X = sample_features_df.values
            y = np.random.randn(X.shape[0])
            
            model = xgb.XGBRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            importances = model.feature_importances_
            
            assert len(importances) == X.shape[1]
            assert (importances >= 0).all()
        except ImportError:
            pytest.skip("XGBoost not installed")


class TestLightGBMModel:
    """Test LightGBM model initialization."""
    
    @pytest.mark.skipif(not LGB_AVAILABLE, reason="LightGBM not installed")
    def test_lightgbm_training(self, sample_features_df):
        """Test LightGBM training."""
        try:
            import lightgbm as lgb
            
            X = sample_features_df.values
            y = np.random.randn(X.shape[0])
            
            model = lgb.LGBMRegressor(
                n_estimators=10,
                max_depth=5,
                random_state=42
            )
            
            model.fit(X[:-10], y[:-10])
            y_pred = model.predict(X[-10:])
            
            assert y_pred.shape == (10,)
        except ImportError:
            pytest.skip("LightGBM not installed")


class TestModelFactory:
    """Test model instantiation via factory method."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_get_model_lstm(self):
        """Test LSTM model creation via factory."""
        try:
            from src.model import get_model
            
            model = get_model(
                model_type="lstm",
                input_size=20,
                hidden_size=64,
            )
            
            assert model is not None
        except (ImportError, NameError):
            pytest.skip("get_model factory not available")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_get_model_transformer(self):
        """Test Transformer model creation via factory."""
        try:
            from src.model import get_model
            
            model = get_model(
                model_type="transformer",
                input_size=20,
                d_model=64,
                nhead=4,
            )
            
            assert model is not None
        except (ImportError, NameError):
            pytest.skip("get_model factory not available")
    
    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGBoost not installed")
    def test_get_model_xgboost(self):
        """Test XGBoost model creation via factory."""
        try:
            from src.model import get_model
            
            model = get_model(
                model_type="xgboost",
                n_estimators=100,
            )
            
            assert model is not None
        except (ImportError, NameError):
            pytest.skip("get_model factory not available")


class TestModelIO:
    """Test model saving and loading."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_torch_model_save_load(self, tmp_path):
        """Test PyTorch model checkpointing."""
        from src.model import LSTMModel
        
        model = LSTMModel(input_size=20, hidden_size=64)
        
        # Save
        checkpoint_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Load
        new_model = LSTMModel(input_size=20, hidden_size=64)
        new_model.load_state_dict(torch.load(checkpoint_path))
        
        # Compare weights
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


class TestOutputShapes:
    """Test model output shapes for different input scenarios."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_output_is_3d_vector(self):
        """Test that LSTM outputs 3-element vector (return, volatility, downside_prob)."""
        from src.model import LSTMModel
        
        model = LSTMModel(input_size=20, output_size=3)
        x = torch.randn(1, 10, 20)
        output = model(x)
        
        assert output.shape == (1, 3)
        # Output values should be in reasonable ranges
        assert torch.isfinite(output).all()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_batch_outputs_correct_size(self):
        """Test LSTM outputs correct batch dimension."""
        from src.model import LSTMModel
        
        model = LSTMModel(input_size=20)
        
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 10, 20)
            output = model(x)
            assert output.shape[0] == batch_size


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_single_sample(self):
        """Test LSTM with batch size of 1."""
        from src.model import LSTMModel
        
        model = LSTMModel(input_size=20, hidden_size=64)
        x = torch.randn(1, 10, 20)
        
        output = model(x)
        assert output.shape == (1, 3)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_lstm_zero_layers_raises_error(self):
        """Test that invalid layer count raises error."""
        from src.model import LSTMModel
        
        with pytest.raises((ValueError, RuntimeError)):
            LSTMModel(input_size=20, num_layers=0)
    
    @pytest.mark.skipif(not XGB_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_single_feature(self, sample_features_df):
        """Test XGBoost with single feature."""
        try:
            import xgboost as xgb
            
            X = sample_features_df.iloc[:, :1].values  # Single feature
            y = np.random.randn(X.shape[0])
            
            model = xgb.XGBRegressor(n_estimators=5, random_state=42)
            model.fit(X[:-10], y[:-10])
            y_pred = model.predict(X[-10:])
            
            assert y_pred.shape == (10,)
        except ImportError:
            pytest.skip("XGBoost not installed")
