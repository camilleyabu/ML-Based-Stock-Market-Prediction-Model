# ML Stock Market Prediction Model 

## Overview

This project compares two machine learning approaches for predicting next-day stock price movement:

- Random Forest Regressor (tree-based ensemble model)
- LSTM (Long Short-Term Memory neural network)

The goal is to evaluate which model performs better on short-term stock prediction using real historical market data.

This project uses daily stock data from Alpaca Markets for:

- Apple (AAPL)
- NVIDIA (NVDA)

## Models Implemented

1. Random Forest Regressor 

Predicts next-day return or price using engineered technical indicators.

Features include:
- returns
- moving averages
- volatility
- volume changes
- market indicators (SPY, QQQ)

Advantages:
- Handles nonlinear relationships
- Stable performance
- Strong baseline model

2. LSTM Neural Network

Sequential deep learning model designed for time-series prediction.

Advantages:
- Learns temporal dependencies
- Models sequence patterns

Limitations observed:
- Financial returns contain high noise
- Sequential signal is weak at daily frequency

## Data Source

Data obtained using Alpaca Markets API:

- Daily bars
- 2022–2026
- Symbols:
    - AAPL
    - NVDA
    - SPY
    - QQQ

## Results Summary

Stock	Model	Direction Accuracy
AAPL	Random Forest	54.4%
AAPL	LSTM	47.3%
NVDA	Random Forest	53.2%
NVDA	LSTM	45.3%

Random Forest outperformed LSTM in directional accuracy for both stocks.

This reflects the noisy and difficult nature of short-term financial prediction.

## Setup

Create virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies: 

```bash
pip install -r requirements.txt
```

Setup Alpaca API
Copy:
```bash
cp .env.example .env
```

Add your keys:
```bash
API_KEY=your_key
API_SECRET=your_secret
```

Running Models:

Run Random Forest:
```bash
python src/aapl_random_forest_reg.py
python src/nvda_random_forest_reg.py
```

Run LSTM:
```bash
python src/aapl_lstm.py
python src/nvda_lstm.py
```
Outputs will be saved to:
```bash
outputs/
```

## Outputs Generated

- prediction plots
- prediction CSV lines
- feature importance analysis

## Technologies Used
- Python
- scikit-learn
- TensorFlow / Keras
- pandas
- numpy
- matplotlib
- Alpaca Markets API

## Author

Camille Yabu
Computer Engineering @UCSC