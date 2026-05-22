# AI Portfolio Analyzer — Implementation Prompts

Copy and paste any of these prompts to get started on specific improvements.

---

## 🏗️ CODE ARCHITECTURE & QUALITY

### Prompt 1: Set Up Testing Suite with pytest
```
I want to add comprehensive testing to the AI Portfolio Analyzer project. 
Please:
1. Create a tests/ directory structure with separate test files for each module
2. Create test fixtures for sample data (small datasets for quick tests)
3. Add unit tests for these modules (target 70%+ coverage):
   - src/dataset.py: test feature engineering, scaling, DataLoader creation
   - src/sentiment.py: test FinBERT scoring and fallback to lexicon scorer
   - src/model.py: test model initialization (LSTM, Transformer, XGBoost)
   - src/optimize.py: test portfolio optimization and weight constraints
4. Set up pytest configuration and add test commands to README
5. Add coverage reporting with pytest-cov
6. Ensure all tests pass locally

Keep tests fast (< 10 seconds total), use mocked API calls where needed, and include edge cases.
```

### Prompt 2: Add Type Hints & Static Type Checking
```
I want to add type hints and static type checking to improve code quality in the AI Portfolio Analyzer.
Please:
1. Add comprehensive type hints to all public functions and key private functions in:
   - src/train.py
   - src/dataset.py
   - src/model.py
   - src/inference.py
   - src/sentiment.py
   - src/optimize.py
2. Use proper return type annotations and argument type annotations
3. Use Union, Optional, List, Dict from typing module where appropriate
4. Add type hints for numpy arrays (use np.ndarray) and pandas DataFrames
5. Set up pyright or mypy configuration in pyproject.toml
6. Run type checking and fix any errors
7. Add a pre-commit hook or GitHub Action to check types on future commits

After completion, verify that `mypy src/ --strict` passes (or with reasonable flags).
```

### Prompt 3: Improve Error Handling & Input Validation
```
I want to add robust error handling and input validation to the AI Portfolio Analyzer.
Please:
1. Create a validation module (src/validation.py) with functions to:
   - Validate ticker symbols (check if valid, has data)
   - Validate date ranges (start < end, not in future)
   - Validate hyperparameters (learning_rate > 0, hidden_size > 0, etc.)
   - Validate config.yaml structure (use Pydantic models)
2. Add error handling to:
   - News fetching (graceful fallback when API fails)
   - Data download (handle missing tickers, timeouts)
   - Model inference (handle edge cases, NaN values)
3. Improve error messages to be user-friendly and actionable
4. Add custom exception classes (e.g., DataValidationError, APIError, ModelError)
5. Update backend_api.py to return proper HTTP error responses with details
6. Add docstring examples showing expected exceptions

Test with invalid inputs to ensure errors are caught gracefully.
```

### Prompt 4: Implement Structured Logging
```
I want to add comprehensive structured logging to the AI Portfolio Analyzer.
Please:
1. Replace basic logging with JSON structured logging (python-json-logger)
2. Add INFO-level logs at key checkpoints in:
   - src/train.py: start training, each epoch, validation, save checkpoint
   - src/dataset.py: start download, data shapes, feature engineering complete
   - src/sentiment.py: sentiment fetch start/end, article counts
   - src/inference.py: each stage of inference pipeline
3. Add DEBUG-level logs for:
   - Hyperparameters on startup
   - Tensor/array shapes
   - Feature names and statistics
4. Add timing information (how long each step takes)
5. Configure logging to output JSON to stdout and optionally to file
6. Add log levels configurable via config.yaml or environment variable
7. Create a logging utility module if needed

Test by running train.py and checking that log output is structured JSON with timestamps, levels, and messages.
```

### Prompt 5: Config Validation with Pydantic
```
I want to add automatic config validation to the AI Portfolio Analyzer using Pydantic.
Please:
1. Create a config schema module (src/config_schema.py) with Pydantic models for:
   - DataConfig (tickers, dates, sequence_length, forecast_horizon)
   - ModelConfig (type, hidden_size, num_layers, dropout)
   - TrainingConfig (learning_rate, batch_size, epochs, early_stopping)
   - NewsConfig (providers, lookback_hours, max_articles)
   - SentimentConfig (model_name, batch_size, decay_factor)
   - PortfolioConfig (method, weight_bounds, risk_free_rate)
   - BacktestConfig (period, rebalance_freq, transaction_costs)
   - FullConfig (combines all above)
2. Update src/utils.py to validate config.yaml on load using Pydantic
3. Provide helpful error messages if config is invalid
4. Generate sample config with all required fields and defaults
5. Add validators for cross-field constraints (e.g., forecast_horizon < sequence_length)

Test by running train.py with valid and invalid configs to ensure proper validation and error messages.
```

---

## 🚀 NEXT STEPS

If you want prompts for model improvements, data science enhancements, frontend development, or deployment:

Just ask! You can say:
- "Give me the model training prompts"
- "Show me frontend prompts"
- "Give me the full list of prompts"

Or copy any prompt above and paste it directly in your next message to start implementing!
