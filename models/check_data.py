"""
Simple data inspection script
"""

import pandas as pd
import os

BASE_DIR = r"C:\Users\Amalie Berg\Desktop\Energy Momentum ML Strategy\energy-momentum-ml-strategy"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
file_path = os.path.join(DATA_DIR, 'ml_features.csv')

print("\n" + "="*70)
print("DATA INSPECTION")
print("="*70)

# Load data
print(f"\nLoading: {file_path}")
df = pd.read_csv(file_path)

print(f"\nData loaded successfully!")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "-"*70)
print("COLUMN NAMES:")
print("-"*70)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "-"*70)
print("FIRST 5 ROWS:")
print("-"*70)
print(df.head())

print("\n" + "-"*70)
print("DATA INFO:")
print("-"*70)
print(df.info())

print("\n" + "-"*70)
print("CHECK FOR ASSET-RELATED COLUMNS:")
print("-"*70)
asset_cols = [col for col in df.columns if 'asset' in col.lower()]
if asset_cols:
    print(f"Found: {asset_cols}")
    for col in asset_cols:
        print(f"\n{col} unique values:")
        print(df[col].unique())
else:
    print("No 'asset' column found")
    print("\nChecking for other identifier columns...")
    possible_id_cols = [col for col in df.columns if any(x in col.lower() for x in ['ticker', 'symbol', 'name', 'instrument'])]
    if possible_id_cols:
        print(f"Found possible identifier columns: {possible_id_cols}")
    else:
        print("No obvious identifier column found")