# ML Stock Market Prediction Model 

Authors: Camille Yabu, Antonio Wydler

Machine learning project exploring next-day stock price prediction using historical market data and technical indicators.

## Features
- Pulls stock data from Alpaca Markets API  
- Uses technical indicators and market context features (SPY, QQQ)  
- Random Forest models for price and directional prediction  
- Includes walk-forward validation and baseline comparisons  

## Current Coverage
- AAPL  
- NVDA  

## Example Results
- NVDA MAE ≈ $3.5  
- Direction accuracy ≈ 53%  

## Setup
```bash
pip install -r requirements.txt
