# AI Portfolio Analyzer

> **Educational disclaimer:** This project is for learning purposes only.
> Nothing here constitutes financial advice. Always consult a qualified
> financial advisor before making investment decisions.

---

A complete, production-style machine-learning pipeline that:

- Fetches historical stock/ETF data with **yfinance**
- Engineers 60+ **technical indicators** and time-series features
- Pulls live **financial news** from multiple providers (NewsAPI, Finnhub, Alpha Vantage, Yahoo RSS, GDELT)
- Scores each article with **FinBERT** financial sentiment (falls back to lexicon scorer)
- Trains **LSTM / Transformer** (PyTorch) and **XGBoost / LightGBM** models to predict return, volatility, and downside risk
- Optimises portfolios via **Max Sharpe / Min Volatility / Risk Parity** (cvxpy + scipy)
- Backtests walk-forward with **no look-ahead bias**
- Tracks experiments with **Weights & Biases**
- Serves everything through a polished **Streamlit dark-mode dashboard**

---

## Project Structure

```
ai-portfolio-analyzer/
│
├── configs/
│   └── config.yaml          # single source of truth for all settings
│
├── src/
│   ├── utils.py             # config loader, device detection, W&B helpers
│   ├── dataset.py           # data download, feature engineering, DataLoaders
│   ├── model.py             # LSTM, Transformer, XGBoost/LightGBM
│   ├── train.py             # training loop, checkpointing, early stopping
│   ├── evaluate.py          # test-set evaluation, prediction plots
│   ├── news.py              # multi-provider news fetcher
│   ├── sentiment.py         # FinBERT scoring + ticker-level aggregation
│   ├── optimize.py          # portfolio optimisation (cvxpy / scipy)
│   ├── backtest.py          # walk-forward backtesting engine
│   └── inference.py         # live inference pipeline → dashboard
│
├── dashboard/
│   ├── app.py               # main Streamlit app
│   ├── components.py        # reusable UI sections
│   └── dashboard_utils.py   # Plotly chart generators + formatters
│
├── notebooks/
│   ├── exploration.ipynb    # EDA walkthrough
│   └── train_colab.ipynb    # Google Colab training notebook
│
├── data/
│   ├── raw/                 # cached yfinance parquet files
│   └── processed/           # feature-engineered parquet files
│
├── checkpoints/             # saved model weights
├── reports/figures/         # generated HTML charts
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick Start (Local)

### 1 — Clone and install

```bash
git clone https://github.com/yourname/ai-portfolio-analyzer.git
cd ai-portfolio-analyzer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure API keys (optional)

```bash
cp .env.example .env
# Edit .env and add any of: NEWSAPI_KEY, FINNHUB_KEY, AV_KEY, WANDB_API_KEY
```

Without API keys the project automatically falls back to **Yahoo Finance RSS** for news
and the **lexicon scorer** for sentiment — no keys required at all.

### 3 — Download data and train

```bash
# Train the default LSTM model on AAPL
python src/train.py --config configs/config.yaml --ticker AAPL

# Train the XGBoost baseline instead
# Edit configs/config.yaml → model.type: xgboost, then re-run
```

### 4 — Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501**, type a ticker (e.g. `NVDA`), click **Analyse**.

---

## Configuration

All settings live in `configs/config.yaml`. Key sections:

| Section | What it controls |
|---|---|
| `data` | Tickers, date range, sequence length, forecast horizon |
| `model` | Architecture (lstm/transformer/xgboost/lightgbm), hidden size, layers |
| `training` | Learning rate, batch size, epochs, early stopping |
| `news` | Provider preference, lookback hours, max articles |
| `sentiment` | FinBERT model name, recency decay, batch size |
| `portfolio` | Optimisation method, weight bounds, risk-free rate |
| `backtest` | Period, rebalance frequency, transaction costs |
| `wandb` | Enable/disable, project name, entity, tags |
| `dashboard` | Default ticker, forecast horizon, port |

---

## Model Architectures

### LSTM (`model.type: lstm`)
Multi-layer LSTM with LayerNorm, GELU head, and **MC-Dropout** for uncertainty estimation.
Input: sliding window of `sequence_length` days × N features → output: `[return, volatility, downside_prob]`.

### Transformer (`model.type: transformer`)
Encoder-only Transformer with sinusoidal positional encoding, pre-LN layers, and mean pooling.
Same input/output shape as LSTM.

### XGBoost / LightGBM (`model.type: xgboost` or `lightgbm`)
Three separate regressors (one per output), trained on temporally-averaged feature windows.
Fast to train, strong baseline, interpretable via feature importances.

---

## News Sentiment Pipeline

```
fetch_news(ticker)
    → articles [title, summary, source, published, url, age_hours]

score_articles(articles)            ← FinBERT or lexicon fallback
    → each article gets [positive, negative, neutral, score, label]

aggregate_sentiment(articles)
    → ticker-level features with recency-weighted scores

sentiment_risk_adjustment(features)
    → return_adj, vol_adj, downside_adj, uncertainty_mult
```

Sentiment is incorporated into model predictions **as additional input features**, not
as a magic price-change multiplier. The sentiment risk adjustment is a transparent,
additive post-processing step applied on top of the model output.

---

## Portfolio Optimisation

```
optimize_portfolio(price_df, cfg, model_preds)
    → weights, expected_return, expected_volatility, sharpe_ratio, risk_contributions
```

Supported methods (set in `config.yaml → portfolio.method`):

| Method | Description |
|---|---|
| `max_sharpe` | Maximise Sharpe ratio (cvxpy CLARABEL solver, scipy SLSQP fallback) |
| `min_volatility` | Minimise portfolio variance |
| `risk_parity` | Equal risk contribution per asset |
| `mean_variance` | Alias for max_sharpe with blended return estimates |

---

## Backtesting

Walk-forward with **strict no-look-ahead**:
- At each rebalance date, only data up to (not including) that date is used.
- Rebalances monthly (default) or weekly.
- Transaction costs deducted on every trade based on turnover.
- Full equity curve, drawdown series, and benchmark comparison (SPY).

Run standalone:

```bash
python src/backtest.py --config configs/config.yaml
```

---

## W&B Experiment Tracking

1. Set `WANDB_API_KEY` in `.env`
2. Set `wandb.enabled: true` in `config.yaml`
3. Add your username to `wandb.entity`

Training will automatically log:
- All hyperparameters
- Train / validation loss per epoch
- Test metrics (RMSE, MAE, R², IC, directional accuracy)
- Model checkpoint as W&B artifact
- Prediction plots

---

## Google Colab

Open `notebooks/train_colab.ipynb` in Colab:

1. **Runtime → Change runtime type → GPU** (T4 is free)
2. Run cells top-to-bottom
3. Checkpoints are saved back to your Google Drive automatically

---

## Dashboard Sections

| Tab | Contents |
|---|---|
| **Overview** | Price + SMA chart, sentiment gauge, risk score |
| **Forecast** | Predicted return, price cone (95% CI), sentiment adjustments |
| **Risk** | Risk score, VaR, CVaR, drawdown, beta, downside probability |
| **Sentiment** | Weighted sentiment gauge, bullish/bearish article cards |
| **Portfolio** | Allocation bars, pie chart, risk contributions |
| **Charts** | Rolling volatility, drawdown, full price history |

---

## Learning Outcomes

By working through this project you will learn:

- Time-series feature engineering for financial data
- LSTM and Transformer architectures for sequence regression
- MC-Dropout for predictive uncertainty
- Gradient-boosted tree models for tabular financial data
- NLP-based sentiment analysis with FinBERT
- Modern Portfolio Theory and convex optimisation
- Walk-forward backtesting without look-ahead bias
- Building production-style ML pipelines with config-driven development
- Experiment tracking with Weights & Biases
- Streamlit dashboard development with custom CSS and Plotly

---

## License

MIT — free for personal and educational use.
