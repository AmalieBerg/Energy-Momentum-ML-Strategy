"""
Data Pipeline for Energy Futures
Downloads and processes energy market data
"""
import sys
from pathlib import Path

# Add project root to path (3 levels up: data -> src -> project_root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import *
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnergyDataPipeline:
    """Pipeline for downloading and processing energy futures data"""
    
    def __init__(self):
        self.raw_data = {}
        
    def download_energy_data(self):
        """Download all energy futures from config"""
        logger.info("📈 Downloading energy futures data...")
        
        for ticker, name in ENERGY_TICKERS.items():
            logger.info(f"  Downloading {name} ({ticker})")
            try:
                # Download with auto_adjust=True to avoid warnings
                data = yf.download(
                    ticker, 
                    start=START_DATE, 
                    end=END_DATE, 
                    progress=False, 
                    auto_adjust=True
                )
                
                if len(data) > 0:
                    logger.debug(f"    Raw data shape: {data.shape}")
                    logger.debug(f"    Columns: {list(data.columns)}")
                    
                    # Handle MultiIndex columns vs regular columns
                    if isinstance(data.columns, pd.MultiIndex) or \
                       (len(data.columns) > 0 and isinstance(data.columns[0], tuple)):
                        # Columns are tuples like ('Close', 'CL=F')
                        close_col = None
                        for col in data.columns:
                            if col[0] == 'Close':
                                close_col = col
                                break
                        
                        if close_col:
                            price_series = data[close_col].dropna()
                            price_series.name = name
                            self.raw_data[name] = price_series
                            logger.info(f"  ✅ {name}: {len(price_series)} observations")
                        else:
                            logger.warning(f"  ❌ {name}: No Close column found")
                            
                    else:
                        # Regular string columns
                        if 'Close' in data.columns:
                            price_series = data['Close'].dropna()
                            price_series.name = name
                            self.raw_data[name] = price_series
                            logger.info(f"  ✅ {name}: {len(price_series)} observations")
                        elif 'Adj Close' in data.columns:
                            price_series = data['Adj Close'].dropna()
                            price_series.name = name
                            self.raw_data[name] = price_series
                            logger.info(f"  ✅ {name}: {len(price_series)} observations")
                        else:
                            logger.warning(f"  ❌ {name}: No suitable price column found")
                else:
                    logger.warning(f"  ❌ {name}: No data returned")
                    
            except Exception as e:
                logger.error(f"  ❌ {name}: Error - {str(e)}")
                
        return self.raw_data
    
    def save_data(self):
        """Save data to the raw folder"""
        if not self.raw_data:
            logger.error("❌ No data to save!")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"\n🔍 Debug - Data to save:")
        for name, series in self.raw_data.items():
            logger.info(f"  {name}: {len(series)} obs, {type(series)}")
            if hasattr(series, 'index'):
                logger.info(f"    Date range: {series.index[0]} to {series.index[-1]}")
        
        # Create DataFrame
        df = pd.DataFrame(self.raw_data)
        
        # Fill any missing values (fixed deprecated method)
        df = df.ffill().bfill()
        
        # Save raw prices
        prices_file = DATA_PATH / 'energy_prices.csv'
        df.to_csv(prices_file)
        logger.info(f"💾 Prices saved to {prices_file}")
        
        # Calculate and save returns
        returns = df.pct_change().dropna()  # Simple returns
        log_returns = np.log(df / df.shift(1)).dropna()  # Log returns
        
        returns_file = DATA_PATH / 'energy_returns.csv'
        log_returns_file = DATA_PATH / 'energy_log_returns.csv'
        
        returns.to_csv(returns_file)
        log_returns.to_csv(log_returns_file)
        
        logger.info(f"💾 Simple returns saved to {returns_file}")
        logger.info(f"💾 Log returns saved to {log_returns_file}")
        logger.info(f"📊 Final data shape: {df.shape}")
        
        return df, returns, log_returns


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("Energy Data Pipeline - Downloading Data")
    logger.info("="*70 + "\n")
    
    pipeline = EnergyDataPipeline()
    data = pipeline.download_energy_data()
    df, returns, log_returns = pipeline.save_data()
    
    if not df.empty:
        logger.info(f"\n📊 Final Summary:")
        logger.info(f"  📅 Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"  📈 Assets and data points:")
        for name in df.columns:
            series = df[name].dropna()
            if len(series) > 0:
                logger.info(f"    {name}: {len(series)} observations")
        
        logger.info(f"\n📊 Sample prices (last 5 days):")
        logger.info(f"\n{df.tail()}")
        
        logger.info(f"\n📊 Sample log returns (last 5 days):")
        logger.info(f"\n{log_returns.tail().round(4)}")
    else:
        logger.error("\n❌ No data was successfully downloaded")