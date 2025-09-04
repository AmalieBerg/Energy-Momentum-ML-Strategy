# test_setup.py
from config import *
import yfinance as yf
import pandas as pd

print("🧪 Testing final setup...")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Energy assets: {list(ENERGY_TICKERS.keys())}")

# Test one download
print("🛢️ Testing WTI download...")
test_data = yf.download("CL=F", start="2024-01-01", end="2024-01-31")
print(f"✅ Downloaded {len(test_data)} rows of WTI data")
print("Setup complete! Ready for data pipeline.")