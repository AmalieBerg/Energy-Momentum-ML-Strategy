import os

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

# GARCH settings
GARCH_LAGS = (1, 1)  # GARCH(1,1) like my thesis
VOL_TARGET = 0.15    # 15% annual volatility target

# Paths
DATA_PATH = 'data/raw/'
PROCESSED_PATH = 'data/processed/'
RESULTS_PATH = 'results/'

print("✅ Energy trading config loaded successfully!")