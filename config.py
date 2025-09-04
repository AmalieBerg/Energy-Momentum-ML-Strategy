 # config.py
import pandas as pd
import numpy as np
import yfinance as yf

print("✅ All imports successful!")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")

# Test data download
test_data = yf.download("CL=F", start="2024-01-01", end="2024-01-31")
print(f"✅ Downloaded {len(test_data)} rows of WTI data")
