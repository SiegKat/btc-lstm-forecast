# Cryptocurrency Price Prediction with Deep Learning & Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning project that predicts **Bitcoin (BTC/USDT)** prices using deep learning and trains a reinforcement learning agent to trade autonomously.

## Overview

This project explores three complementary approaches to cryptocurrency trading:

| Model | Approach | Purpose |
|-------|----------|---------|
| **Bidirectional LSTM** | Supervised Learning | Next-step price prediction |
| **MC-Dropout LSTM** | Bayesian Deep Learning | Price prediction with uncertainty quantification |
| **PPO Trading Agent** | Reinforcement Learning | Learn a Buy / Hold / Sell policy from market interaction |

## Data Pipeline

Data is sourced from the [Binance Vision](https://data.binance.vision/) public repository:

- **Trading pairs:** ETHUSDT, ETHBTC, ETHUSDC, BTCUSDT
- **Intervals:** 5-minute and 15-minute candles
- **Period:** August 2017 &ndash; September 2024
- **Primary modelling target:** BTCUSDT 15-minute (~250K candles)

The data loading logic uses pair-specific URL slicing to handle different listing dates and gaps in data availability (e.g. ETHUSDC's missing months). These offsets are hardcoded and must not be changed without verifying against Binance data availability.

## Architecture

```
                    ┌─────────────────────┐
                    │   Binance Vision    │
                    │   (4 pairs, 2 TFs)  │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │  URL Build & Trim   │
                    │  (pair-specific     │
                    │   month slicing)    │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────┐
                    │ Feature Engineering │
                    │  RSI, MACD, BB, ATR │
                    │  OBV, Stochastic    │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌────▼────────┐ ┌──▼────────────┐
     │ Bidirectional │ │ MC-Dropout  │ │  PPO Trading  │
     │     LSTM      │ │    LSTM     │ │    Agent      │
     └────────┬──────┘ └────┬────────┘ └──┬────────────┘
              │              │              │
     ┌────────▼──────┐ ┌────▼────────┐ ┌──▼────────────┐
     │    Price      │ │  Price +    │ │   Portfolio   │
     │  Prediction   │ │ Uncertainty │ │   Backtest    │
     └───────────────┘ └─────────────┘ └───────────────┘
```

## Project Structure

```
btc-lstm-forecast/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Binance URL building, trimming & loading
│   ├── features.py            # Technical indicators & sequence creation
│   ├── lstm_model.py          # Bidirectional LSTM & MC-Dropout builders
│   ├── rl_env.py              # Gymnasium trading env + PPO training
│   └── utils.py               # Plotting, metrics, evaluation helpers
├── notebooks/
│   └── crypto_prediction.ipynb  # Main analysis & unseen-data validation notebook
├── models/                    # Saved model weights (gitignored)
└── data/                      # Downloaded Binance CSVs (gitignored, auto-downloaded)
```

## Technical Indicators

The feature engineering step adds 11 indicators computed with [pandas-ta](https://github.com/twopirllc/pandas-ta):

| Indicator | Parameters | Description |
|-----------|-----------|-------------|
| RSI | 14 | Relative Strength Index |
| MACD | 12/26/9 | Moving Average Convergence Divergence + Signal Line |
| SMA | 10, 50 | Simple Moving Averages (short & medium-term) |
| Bollinger Bands | 20, 2&sigma; | Upper and Lower volatility bands |
| ATR | 14 | Average True Range |
| OBV | - | On-Balance Volume |
| Stochastic | %K(14), %D(3) | Overbought / oversold oscillator |

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/SiegKat/btc-lstm-forecast.git
cd btc-lstm-forecast
pip install -r requirements.txt
```

### Running the Notebook

```bash
cd notebooks
jupyter notebook crypto_prediction.ipynb
```

The notebook will:
1. Download ~250K&ndash;750K candles per pair from Binance (requires internet)
2. Preprocess and compute technical indicators
3. Train a Bidirectional LSTM (~10 epochs)
4. Train an MC-Dropout LSTM with uncertainty estimation (~10 epochs)
5. Train a PPO trading agent (~50K timesteps)
6. Generate predictions, uncertainty intervals, and a 500-step rolling forecast
7. Download 3 months of unseen data and evaluate model drift at 1-week, 1-month, and 3-month horizons

## Technologies

- **Deep Learning:** TensorFlow / Keras (Bidirectional LSTM, MC-Dropout)
- **Reinforcement Learning:** Stable-Baselines3 (PPO), Gymnasium
- **Data:** pandas, NumPy, Binance Vision public CSV API
- **Feature Engineering:** pandas-ta
- **Visualization:** Matplotlib

## Limitations & Disclaimer

- Models use only price/volume-derived features.
- Auto-regressive forecasts accumulate error rapidly.
- **This project is for educational purposes only. It does not constitute financial advice.**

## Future Improvements

- Transformer-based architectures (Temporal Fusion Transformer)
- Multi-asset correlation features (ETH/BTC spread)
- Walk-forward cross-validation for realistic backtesting
- Paper-trading deployment for the RL agent
- Hyperparameter optimization via Optuna

## License

MIT
