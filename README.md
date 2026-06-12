# 🏦 Autonomous Alpha Engine v2.0
### End-to-End Quantitative Finance Research System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Models](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM%20%7C%20RF%20%7C%20LSTM-orange)
![Ensemble](https://img.shields.io/badge/Ensemble-Stacking%20Meta--Learner-purple)
![Backtest](https://img.shields.io/badge/Backtest-Sharpe%20%7C%20Calmar%20%7C%20Sortino-green)
![RAG](https://img.shields.io/badge/GenAI-LangChain%20%7C%20RAG%20%7C%20GPT--4o-blueviolet)
![API](https://img.shields.io/badge/API-FastAPI%20v2-teal)

---

## 📌 Overview

A **production-grade, multi-layer alpha generation pipeline** for S&P 500 directional forecasting.  
Combines classical econometrics, ensemble ML, deep sequence learning, portfolio optimisation,  
rigorous walk-forward backtesting, and an agentic RAG synthesis layer.

**Designed for:** Quantitative research, systematic trading strategy development, and portfolio risk management.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        AUTONOMOUS ALPHA ENGINE v2.0                              │
├──────────────────┬───────────────────────────┬───────────────────────────────────┤
│   DATA LAYER     │      SIGNAL LAYER          │       STRATEGY LAYER              │
│  ─────────────   │  ─────────────────────     │  ───────────────────────────      │
│  S&P 500 OHLCV   │  XGBoost + Optuna HPO      │  Vectorised Backtester            │
│  16yr News       │  LightGBM Classifier       │  Sharpe / Calmar / Sortino        │
│  VADER NLP       │  Random Forest             │  Walk-Forward Validation          │
│  FinBERT NLP     │  Stacking Meta-Ensemble    │  Monthly Return Heatmaps          │
│  VIX / TNX / DXY │  LSTM + Attention          │  Monte Carlo (10K paths)          │
│  Macro Indicators│  Isolation Forest          │  GAN Stress Testing               │
│  HMM Regimes     │  Probability Calibration   │  Markowitz Portfolio Optim.       │
└──────────────────┴───────────────────────────┴───────────────────────────────────┘
                                     │
                        ┌────────────▼──────────────┐
                        │   AGENTIC RAG SYNTHESIS    │
                        │  Multi-Model Committee     │
                        │  FAISS Vector DB (Macro)   │
                        │  GPT-4o Final Directive    │
                        └───────────────────────────┘
                                     │
                        ┌────────────▼──────────────┐
                        │   FastAPI v2 Microservice  │
                        │  /predict/signal           │
                        │  /predict/ensemble         │
                        │  /predict/risk             │
                        │  /backtest/quick           │
                        └───────────────────────────┘
```

---

## 📋 Research Phases (Notebook)

| Phase | Module | Key Output |
|-------|--------|-----------|
| 0 | Environment & Config | Reproducible seeds, logging |
| 1 | Multi-Source Data Ingestion | OHLCV + NLP + Macro unified dataset |
| 2 | Advanced Feature Engineering | 40+ alpha factors across 8 categories |
| 3 | HMM Regime Detection | Bull / Sideways / Bear regime labels |
| 4 | Deep EDA (20 charts) | 3 diagnostic figure panels |
| 5 | Isolation Forest Anomaly Detection | Outlier-aware feature flag |
| 6 | Walk-Forward Split Protocol | No look-ahead 70/15/15 split |
| 7 | XGBoost + Optuna HPO | 50-trial Bayesian-tuned classifier |
| 8 | LightGBM | GBDT ensemble leg with early stopping |
| 9 | Random Forest + Permutation Importance | Diversity leg, model-agnostic importance |
| 10 | Stacking Meta-Ensemble | Blended signal (TimeSeriesSplit CV) |
| 11 | LSTM + Attention | Deep return magnitude forecaster |
| 12 | Production Backtesting Engine | Sharpe, Sortino, Calmar, MDD, VaR, CVaR |
| 13 | Walk-Forward Validation | Expanding-window quarterly retraining |
| 14 | Monte Carlo Simulation | 10,000-path risk distribution |
| 15 | GAN Stress Testing | Synthetic black-swan scenario generation |
| 16 | Portfolio Optimization | Markowitz frontier + Kelly sizing |
| 17 | Agentic RAG Synthesis | LLM multi-model committee directive |
| 18 | FastAPI v2 Reference | Production API spec |
| 19 | Final Scorecard | Consolidated metrics table |

---

## 📊 Feature Engineering (40+ Alpha Factors)

| Category | Features |
|---|---|
| **Returns** | 1d, 2d, 3d, 5d, 10d, 21d returns |
| **Volatility** | 5d, 10d, 21d, 63d realised vol · Vol ratio · Parkinson estimator |
| **Momentum** | RSI-14 · MACD / Signal / Histogram · Stochastic K/D · SMA/EMA crosses |
| **Mean-Reversion** | Z-score 21d / 63d · Bollinger Band position & width |
| **Volume** | Volume ratio · On-Balance Volume (OBV) |
| **Sentiment** | VADER mean/std · FinBERT mean/std · Headline count · Sentiment momentum · Bull/Bear ratio |
| **Macro** | VIX / VIX change / Fear indicator · 10Y yield / slope · DXY |
| **Structural** | ATR · Parkinson vol · HMM regime · Anomaly flag |

---

## 🎯 Model Performance

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|---------|
| XGBoost + Optuna | ~0.55 | ~0.57 | ~0.56 |
| LightGBM | ~0.54 | ~0.56 | ~0.55 |
| Random Forest | ~0.53 | ~0.55 | ~0.54 |
| **Stacking Ensemble** | **~0.56** | **~0.58** | **~0.57** |
| LSTM (direction) | ~0.53 | — | ~0.53 |

*Note: Metrics are approximate. Market directional prediction is fundamentally noisy; performance above 0.55 AUC on out-of-sample financial data is considered commercially meaningful.*

---

## 🚀 Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/autonomous-alpha-engine.git
cd autonomous-alpha-engine
pip install -r requirements.txt

# 2. Run the notebook (Google Colab recommended for GPU)
jupyter notebook notebooks/Autonomous_Alpha_Engine_v2.ipynb

# 3. Launch the API (after notebook generates model files)
uvicorn src.api_v2:app --reload --host 0.0.0.0 --port 8000

# 4. Interactive API docs
open http://localhost:8000/docs
```

### Environment Variables
```bash
OPENAI_API_KEY=sk-...   # Required for Phase 17 RAG synthesis (optional otherwise)
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check & model status |
| POST | `/predict/signal` | XGBoost directional signal |
| POST | `/predict/ensemble` | Full ensemble vote + confidence breakdown |
| POST | `/predict/risk` | Kelly sizing, VaR, stop-loss recommendation |
| POST | `/backtest/quick` | Vectorised backtest on supplied return series |
| GET | `/model/info` | Feature manifest and architecture metadata |
| GET | `/metrics/requests` | Request monitoring log |

### Example API Call
```python
import requests

features = {
    "ret_1d": 0.008, "vol_5d": 0.012, "vol_21d": 0.015,
    "rsi_14": 52.3, "macd_hist": 0.0012,
    "vader_mean": 0.15, "finbert_mean": 0.08,
    "headline_count": 42,
    # ... (see /docs for full schema)
}
r = requests.post("http://localhost:8000/predict/ensemble", json=features)
print(r.json())
# {"final_signal": "UP", "final_action": "BUY", "ensemble_confidence": 0.623, ...}
```

---

## 📁 Repository Structure

```
autonomous-alpha-engine-v2/
├── data/
│   └── sp500_headlines_2008_2024.csv    # 16-year news corpus
├── notebooks/
│   └── Autonomous_Alpha_Engine_v2.ipynb # Main research notebook (19 phases)
├── src/
│   ├── api_v2.py                        # FastAPI v2 production microservice
│   └── api_v1.py                        # Original v1 API (reference)
├── models/
│   ├── xgboost_v2.json                  # Trained XGBoost (generated by notebook)
│   └── ...                              # Additional model artefacts
├── reports/                             # Generated performance reports
├── requirements.txt
└── README.md
```

---

## 🧠 Technologies

| Layer | Stack |
|-------|-------|
| Data | `yfinance` · `pandas` · `numpy` |
| NLP | `NLTK VADER` · `ProsusAI/FinBERT` · `transformers` |
| ML | `XGBoost` · `LightGBM` · `scikit-learn` · `Optuna` |
| Deep Learning | `TensorFlow/Keras` (LSTM, GAN) |
| Statistics | `statsmodels` · `hmmlearn` · `scipy` |
| Agentic AI | `LangChain` · `FAISS` · `OpenAI GPT-4o` |
| API | `FastAPI` · `Pydantic` · `Uvicorn` |
| Visualisation | `matplotlib` · `seaborn` |

---

## 📜 Key Quantitative Concepts Demonstrated

- **Walk-Forward Validation** — correct methodology for time-series ML
- **Information Coefficient (IC)** — signal quality measurement
- **Sharpe / Sortino / Calmar Ratios** — risk-adjusted return measurement
- **Maximum Drawdown & Underwater Curve** — capital preservation analysis
- **Value at Risk (VaR) & CVaR** — tail risk quantification
- **Monte Carlo Simulation** — probabilistic wealth projection
- **Kelly Criterion** — optimal position sizing
- **Markowitz Mean-Variance Optimisation** — portfolio construction
- **HMM Regime Detection** — structural market state modelling
- **GARCH Effect / Volatility Clustering** — econometric diagnostics
- **Augmented Dickey-Fuller Test** — stationarity verification
- **Isolation Forest** — anomaly/outlier detection
- **Probability Calibration** — model reliability scoring
- **GAN Stress Testing** — synthetic scenario generation

---

*Built for systematic quantitative research. Not financial advice.*
