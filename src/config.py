"""
Configuration file for Energy Momentum ML Strategy
All paths are dynamic and relative to project root
"""
import os
from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data settings
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'

# Energy tickers (Yahoo Finance)
ENERGY_TICKERS = {
    'CL=F': 'WTI_Crude',      # WTI Crude Oil
    'BZ=F': 'Brent_Crude',    # Brent Crude
    'NG=F': 'Natural_Gas',    # Natural Gas
    'HO=F': 'Heating_Oil',    # Heating Oil
    'RB=F': 'Gasoline'        # Gasoline
}

# Assets list
ASSETS = [
    'WTI_Crude',
    'Brent_Crude',
    'Natural_Gas',
    'Heating_Oil',
    'Gasoline'
]

# GARCH settings
GARCH_LAGS = (1, 1)  # GARCH(1,1)
VOL_TARGET = 0.15    # 15% annual volatility target

# Paths (all relative to PROJECT_ROOT)
DATA_PATH = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed'
MODEL_PATH = PROJECT_ROOT / 'models' / 'saved'
RESULTS_PATH = PROJECT_ROOT / 'results'

# Ensure directories exist
DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Backtesting parameters
BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'train_end_date': '2022-12-31',
    'test_start_date': '2023-01-01',
    'position_size': 0.2,
    'transaction_cost': 0.001
}

# Model training parameters
ML_CONFIG = {
    'n_regimes': 3,
    'regime_method': 'gmm',
    'test_size': 0.2,
    'random_state': 42,
    'cv_splits': 5
}

print(f"✅ Configuration loaded successfully!")
print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📁 Data path: {DATA_PATH}")
print(f"📁 Results path: {RESULTS_PATH}")
