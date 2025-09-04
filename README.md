 # Energy Momentum ML Strategy

**A machine learning enhanced momentum trading strategy for energy futures using GARCH volatility forecasting.**

## 🎯 Project Status: Week 1 Complete ✅

### ✅ Completed Features:
- **Data Pipeline**: Automated download of 5 energy futures (15+ years of data)
- **GARCH Model**: Professional volatility forecasting with diagnostic plots
- **Data Processing**: Clean price and return series ready for strategy development

### 📊 Key Results:
- **WTI Crude GARCH(1,1)**: 98.4% persistence, 23.2% average volatility
- **Data Coverage**: 3,769 daily observations (2010-2024)
- **Assets**: WTI Crude, Brent Crude, Natural Gas, Heating Oil, Gasoline

### 🔄 Next Week: Machine Learning Regime Detection

## 🛠️ Technical Implementation

### Data Sources:
- Yahoo Finance API for energy futures
- Daily frequency with robust error handling
- Automatic forward-filling for missing values

### GARCH Model Features:
- Based on Heston-Nandi theoretical framework
- Professional diagnostic plots
- Multi-asset volatility forecasting
- Realistic volatility levels (15-40% annually for energy)

## 📁 Project Structure:
├── config.py              # Project configuration
├── data/
│   ├── data_pipeline.py    # Data collection & processing
│   └── raw/               # Downloaded datasets (gitignored)
├── models/
│   └── garch_model.py     # GARCH volatility forecasting
└── results/               # Diagnostic plots & analysis

├── config.py              # Project configuration
├── data/
│   ├── data_pipeline.py    # Data collection & processing
│   └── raw/               # Downloaded datasets (gitignored)
├── models/
│   └── garch_model.py     # GARCH volatility forecasting
└── results/               # Diagnostic plots & analysis


## Week 2 Day 1 Complete: Feature Engineering Pipeline

### New Features Added:
- **Feature Engineering**: 146 ML features across multiple categories
- **GARCH Integration**: Volatility features using thesis expertise
- **External Data**: VIX, Treasury rates, Dollar Index integration
- **Multi-Asset Support**: Features for all 5 energy futures

### Feature Categories:
- **Momentum Features**: 60 features across multiple time horizons
- **Volatility Features**: 30 GARCH-based volatility indicators
- **Market Structure**: 50 trend and microstructure features
- **External Factors**: 6 macro-economic indicators

### Technical Implementation:
- Robust data cleaning and feature selection
- Automatic handling of missing values
- Memory-efficient feature storage (4.2 MB dataset)
- Professional error handling and validation