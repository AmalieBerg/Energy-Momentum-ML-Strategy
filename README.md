# Energy Momentum ML Trading Strategy

Machine learning-based momentum trading strategy for energy futures markets (WTI Crude, Brent Crude, Natural Gas, Heating Oil, Gasoline).

## 🎯 Project Overview

This project implements a quantitative trading strategy that uses machine learning to predict regime changes in energy commodity markets. The strategy combines technical momentum indicators with market regime classification to generate long/short/neutral signals.

### Key Features
- **Multi-asset coverage**: 5 major energy commodities
- **ML-based regime detection**: Random Forest, Gradient Boosting, LightGBM
- **Comprehensive feature engineering**: 147+ technical and market features
- **Realistic backtesting**: Transaction costs, proper train/test splits, realistic performance metrics

## 📊 Performance Summary (2023-2024 Test Period)

| Asset | Annual Return | Sharpe Ratio | Max Drawdown | Outperformance vs B&H |
|-------|--------------|--------------|--------------|----------------------|
| **WTI Crude** | 12.85% | 0.54 | -20.64% | +34.96% |
| **Brent Crude** | 0.64% | 0.17 | -21.84% | +10.66% |
| **Heating Oil** | 16.86% | 0.65 | -28.39% | +61.90% |
| **Gasoline** | -17.95% | -0.41 | -44.53% | -16.27% |
| **Portfolio Avg** | **3.10%** | **0.24** | **-28.85%** | **+22.81%** |

### Key Insights
- ✅ Strategy significantly outperforms buy & hold (+22.81% on average)
- ✅ Best performance on **Heating Oil** (16.86% annual return, 0.65 Sharpe)
- ✅ Consistent positive returns on WTI Crude and Heating Oil
- ⚠️ Underperformance on Gasoline in 2023-2024 test period

## 🏗️ Project Structure

```
energy-momentum-ml-strategy/
│
├── data/
│   ├── raw/                    # Raw price data from exchanges
│   │   ├── CL_daily.csv       # WTI Crude
│   │   ├── BZ_daily.csv       # Brent Crude
│   │   ├── HO_daily.csv       # Heating Oil
│   │   └── RB_daily.csv       # Gasoline
│   │
│   └── processed/              # Processed features and labels
│       ├── ml_features.csv    # 147 engineered features
│       ├── regime_labels.csv  # Market regime classifications
│       └── cached_prices_and_features.csv
│
├── models/
│   └── saved/                  # Trained ML models
│       ├── {asset}_Random_Forest_model.pkl
│       ├── {asset}_Random_Forest_scaler.pkl
│       └── {asset}_Random_Forest_encoder.pkl
│
├── results/                    # Backtest results and visualizations
│   ├── backtest_results_*.csv
│   └── backtest_*.png
│
├── backtest_engine_FINAL.py   # Main backtesting script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- yfinance (optional, for fetching fresh price data)

### Running the Backtest

```bash
python backtest_engine_FINAL.py
```

This will:
1. Load pre-calculated ML features from `data/processed/ml_features.csv`
2. Load actual price data for return calculations
3. Generate trading signals using trained Random Forest models
4. Calculate realistic performance metrics
5. Save results and plots to `results/` directory

### Configuration

Edit the configuration section in `backtest_engine_FINAL.py`:

```python
INITIAL_CAPITAL = 100000       # Starting capital
TRAIN_END_DATE = '2022-12-31'  # End of training period
TEST_START_DATE = '2023-01-01' # Start of test period
POSITION_SIZE = 0.2            # 20% of capital per position
TRANSACTION_COST = 0.001       # 0.1% transaction costs
```

## 🔬 Methodology

### Feature Engineering (147 features per asset)

**Momentum Features:**
- Multi-period momentum (5d, 10d, 21d, 60d, 126d)
- Momentum strength indicators
- Rate of change across different timeframes

**Volatility Features:**
- Rolling standard deviation (10d, 20d, 60d)
- Parkinson volatility (high-low range)
- Volatility regimes

**Technical Indicators:**
- RSI (Relative Strength Index)
- Moving averages (20, 50, 200 day)
- Price vs MA ratios
- Bollinger Bands

**Market Context:**
- VIX (volatility index) levels and changes
- US Dollar Index (DXY) momentum
- Treasury yield curve slope
- Cross-asset correlations

### ML Models

**Primary Model: Random Forest Classifier**
- Predicts market regime: Bearish (0) / Neutral (1) / Bullish (2)
- Features scaled using StandardScaler
- Trained on 2010-2022 data, tested on 2023-2024

**Regime to Signal Mapping:**
- Regime 0 (Bearish) → Short position (-1)
- Regime 1 (Neutral) → Flat position (0)
- Regime 2 (Bullish) → Long position (+1)

### Backtesting Framework

**Risk Management:**
- Position sizing: 20% of capital per asset
- Transaction costs: 0.1% per trade
- No leverage

**Validation:**
- Walk-forward testing (no look-ahead bias)
- Signals shifted by 1 day (trade on next day's open)
- Realistic slippage and transaction costs included

## 📈 Results & Analysis

### Strategy Strengths
1. **Regime Adaptability**: Successfully identifies and trades different market regimes
2. **Downside Protection**: Outperforms buy & hold during declining markets
3. **Heating Oil Expertise**: Particularly strong on Heating Oil (16.86% annual return)

### Areas for Improvement
1. **Gasoline Performance**: Strategy underperformed on Gasoline (-17.95%)
2. **Low Sharpe Ratios**: Average Sharpe of 0.24 suggests high volatility relative to returns
3. **Natural Gas**: Missing price data - need to add for complete portfolio

### Risk Metrics
- **Maximum Drawdown**: -44.53% (Gasoline) to -20.64% (WTI Crude)
- **Win Rate**: ~50% (typical for momentum strategies)
- **Volatility**: Higher than buy & hold due to active trading

## 🔧 Technical Implementation

### Critical Design Decisions

**1. Separate Features and Prices**
- ML features (momentum, RSI, etc.) → Used for predictions
- Actual prices → Used for return calculations
- Never mix the two (causes nonsensical results)

**2. Proper Return Calculation**
```python
# ✓ CORRECT
asset_returns = df[f'{asset}_Close'].pct_change()
strategy_returns = signals.shift(1) * asset_returns

# ❌ WRONG (produces fake results)
# asset_returns = df[f'{asset}_momentum_5d']
```

**3. Feature Filtering**
Price columns (`_Close`, `_Open`, `_High`, `_Low`, `_Volume`) are excluded from ML features to avoid data leakage.

## 📝 Future Enhancements

### Short-term
- [ ] Add Natural Gas price data
- [ ] Test other ML models (Gradient Boosting, LightGBM, Neural Networks)
- [ ] Implement position sizing optimization
- [ ] Add stop-loss mechanisms

### Medium-term
- [ ] Portfolio optimization across assets
- [ ] Dynamic position sizing based on volatility
- [ ] Include more market features (crude oil inventories, refinery utilization)
- [ ] Walk-forward optimization

### Long-term
- [ ] Live trading integration
- [ ] Real-time data feeds
- [ ] Automated execution
- [ ] Risk management dashboard

## 📚 Academic Background

This project combines knowledge from:
- **Finance**: Portfolio theory, derivatives pricing, risk management
- **Physics**: Time series analysis, statistical mechanics analogies
- **Machine Learning**: Classification, feature engineering, model validation

Author: Amalie Berg
- Master's in Economics & Finance (NHH/CEMS)
- Master's in Materials, Energy & Nanotechnology (University of Oslo)
- Currently pursuing: M.S. Software Engineering (Quantic)

## 🔗 Related Work

- Published research: Solar cell efficiency optimization (Solar Energy Materials, 2019)
- Thesis: Heston-Nandi GARCH option valuation models (NHH, 2025)
- Experience: Risk analysis at Storebrand Asset Management

## ⚠️ Disclaimer

This project is for educational and research purposes only. Past performance does not guarantee future results. Trading energy futures involves substantial risk and may not be suitable for all investors.

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a personal research project. For questions or collaboration inquiries, please open an issue.

---

**Last Updated:** January 2025  
**Status:** Active Development
