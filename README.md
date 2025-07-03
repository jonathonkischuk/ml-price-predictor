# 🧠 Price Movement Predictor: ML-Powered Forecasting for Stocks & Crypto

Welcome to the **Price Movement Predictor** — an end-to-end machine learning project that predicts **next-day price direction (Up/Down)** for select stocks and cryptocurrencies using technical indicators, engineered features, and ensemble learning models.

> 📈 Assets Covered:  
> • Stocks: `Microsoft (MSFT)`, `Meta (META)`, `Rithm Capital (RITM)`  
> • Cryptocurrencies: `JASMY`, `Render Token (RNDR)`

---

## 🚀 Project Overview

This project demonstrates how to:
- Collect and preprocess historical market data (stocks + crypto)
- Engineer useful features (returns, momentum, RSI, volatility, etc.)
- Train machine learning models (XGBoost Classifier)
- Evaluate predictions and performance
- Containerize the pipeline using Docker

<div align="center">
  <img src="https://github.com/yourusername/price-movement-predictor/blob/main/assets/demo-flowchart.png" alt="Pipeline Overview" width="700">
</div>

---

## 📂 Project Structure

```bash
price-movement-predictor/
├── data/                  # Raw downloaded CSVs
├── models/                # Saved model files (joblib)
├── notebooks/             # Optional: Jupyter explorations
├── predictor/             # Core logic
│   ├── data_loader.py         # Data collection (Yahoo + CoinGecko)
│   ├── feature_engineering.py # Technical indicator generation
│   ├── model.py               # ML training logic
│   ├── evaluate.py            # (Optional) Visualization / metrics
│   └── utils.py               # Utility functions (if needed)
├── main.py               # End-to-end orchestrator
├── requirements.txt      # Python dependencies
├── Dockerfile            # Containerized environment
└── README.md             # Project documentation
