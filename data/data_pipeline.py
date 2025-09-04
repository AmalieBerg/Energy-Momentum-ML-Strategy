# data/data_pipeline.py
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class EnergyDataPipeline:
    def __init__(self):
        self.raw_data = {}
        # Make sure directories exist
        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        
    def download_energy_data(self):
        """Download all energy futures from your config"""
        print("📈 Downloading energy futures data...")
        
        for ticker, name in ENERGY_TICKERS.items():
            print(f"  Downloading {name} ({ticker})")
            try:
                # Download with auto_adjust=True to avoid the warning
                data = yf.download(ticker, start=START_DATE, end=END_DATE, 
                                 progress=False, auto_adjust=True)
                
                if len(data) > 0:
                    print(f"    Raw data shape: {data.shape}")
                    print(f"    Columns: {list(data.columns)}")
                    
                    # Handle MultiIndex columns (tuples) vs regular columns (strings)
                    if isinstance(data.columns, pd.MultiIndex) or (len(data.columns) > 0 and isinstance(data.columns[0], tuple)):
                        # Columns are tuples like ('Close', 'CL=F')
                        print("    Detected tuple columns")
                        close_col = None
                        for col in data.columns:
                            if col[0] == 'Close':  # First element of tuple
                                close_col = col
                                break
                        
                        if close_col:
                            price_series = data[close_col].dropna()
                            price_series.name = name
                            self.raw_data[name] = price_series
                            print(f"  ✅ {name}: {len(price_series)} observations using {close_col}")
                        else:
                            print(f"  ❌ {name}: No Close column found in {list(data.columns)}")
                            
                    else:
                        # Regular string columns
                        print("    Detected string columns")
                        if 'Close' in data.columns:
                            price_series = data['Close'].dropna()
                            price_series.name = name
                            self.raw_data[name] = price_series
                            print(f"  ✅ {name}: {len(price_series)} observations using 'Close'")
                        elif 'Adj Close' in data.columns:
                            price_series = data['Adj Close'].dropna()
                            price_series.name = name
                            self.raw_data[name] = price_series
                            print(f"  ✅ {name}: {len(price_series)} observations using 'Adj Close'")
                        else:
                            print(f"  ❌ {name}: No suitable price column found in {list(data.columns)}")
                else:
                    print(f"  ❌ {name}: No data returned")
                    
            except Exception as e:
                print(f"  ❌ {name}: Error - {str(e)}")
                import traceback
                traceback.print_exc()
                
        return self.raw_data
    
    def save_data(self):
        """Save data to the raw folder"""
        if self.raw_data:
            # Debug: Check what we have
            print(f"\n🔍 Debug - Data to save:")
            for name, series in self.raw_data.items():
                print(f"  {name}: {len(series)} obs, {type(series)}")
                if hasattr(series, 'index'):
                    print(f"    Date range: {series.index[0]} to {series.index[-1]}")
            
            # Create DataFrame - should work now with proper Series objects
            df = pd.DataFrame(self.raw_data)
            
            # Fill any missing values forward, then backward
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Save raw prices
            prices_file = f"{DATA_PATH}energy_prices.csv"
            df.to_csv(prices_file)
            print(f"💾 Prices saved to {prices_file}")
            
            # Calculate and save returns
            returns = df.pct_change().dropna()  # Simple returns
            log_returns = np.log(df / df.shift(1)).dropna()  # Log returns (better for GARCH)
            
            returns_file = f"{DATA_PATH}energy_returns.csv"
            log_returns_file = f"{DATA_PATH}energy_log_returns.csv"
            
            returns.to_csv(returns_file)
            log_returns.to_csv(log_returns_file)
            
            print(f"💾 Simple returns saved to {returns_file}")
            print(f"💾 Log returns saved to {log_returns_file}")
            print(f"📊 Final data shape: {df.shape}")
            
            return df, returns, log_returns
        else:
            print("❌ No data to save!")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Test it
if __name__ == "__main__":
    pipeline = EnergyDataPipeline()
    data = pipeline.download_energy_data()
    df, returns, log_returns = pipeline.save_data()
    
    if not df.empty:
        print(f"\n📊 Final Summary:")
        print(f"  📅 Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  📈 Assets and data points:")
        for name in df.columns:
            series = df[name].dropna()
            if len(series) > 0:
                print(f"    {name}: {len(series)} observations")
        
        print(f"\n📊 Sample prices (last 5 days):")
        print(df.tail())
        
        print(f"\n📊 Sample log returns (last 5 days):")
        print(log_returns.tail().round(4))
        
    else:
        print("\n❌ No data was successfully downloaded")