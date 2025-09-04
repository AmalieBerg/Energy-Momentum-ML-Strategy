# models/feature_engineering.py
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from models.garch_model import EnergyGARCHModel
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MarketFeatureEngineer:
    """
    Create features for market regime detection
    Combines traditional indicators with your GARCH expertise!
    """
    
    def __init__(self):
        self.features = None
        self.feature_names = []
        
    def create_momentum_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum indicators across different time horizons
        """
        print("📈 Creating momentum features...")
        
        momentum_features = pd.DataFrame(index=prices.index)
        
        # Multiple momentum horizons (days)
        horizons = [5, 10, 21, 63, 126, 252]  # 1 week to 1 year
        
        for asset in prices.columns:
            for horizon in horizons:
                # Simple momentum (returns)
                mom_col = f"{asset}_momentum_{horizon}d"
                momentum_features[mom_col] = prices[asset].pct_change(horizon)
                
                # Momentum strength (consistency of direction)
                strength_col = f"{asset}_momentum_strength_{horizon}d"
                rolling_returns = prices[asset].pct_change().rolling(horizon)
                momentum_features[strength_col] = rolling_returns.apply(
                    lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
                )
        
        print(f"   ✅ Created {len([c for c in momentum_features.columns if 'momentum' in c])} momentum features")
        return momentum_features
    
    def create_volatility_features(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features using your GARCH expertise!
        """
        print("🔥 Creating volatility features with GARCH models...")
        
        vol_features = pd.DataFrame(index=returns.index)
        
        for asset in returns.columns:
            print(f"   Processing {asset}...")
            
            # Fit GARCH model (your expertise!)
            garch = EnergyGARCHModel(p=1, q=1)
            returns_clean = returns[asset].dropna()
            
            try:
                garch.fit_garch(returns_clean, verbose=False)
                
                # GARCH conditional volatility (your core strength!)
                garch_vol = garch.get_conditional_volatility()
                garch_vol = garch_vol.reindex(returns.index, method='ffill')
                vol_features[f"{asset}_garch_vol"] = garch_vol
                
                # Realized volatility (different windows)
                for window in [5, 21, 63]:
                    realized_vol = returns[asset].rolling(window).std() * np.sqrt(252)
                    vol_features[f"{asset}_realized_vol_{window}d"] = realized_vol
                
                # Volatility term structure (GARCH vs realized)
                vol_features[f"{asset}_vol_term_structure"] = (
                    garch_vol * np.sqrt(252) / vol_features[f"{asset}_realized_vol_21d"]
                ).replace([np.inf, -np.inf], 1.0)
                
                # Volatility persistence (autocorrelation)
                vol_squared = returns[asset] ** 2
                vol_features[f"{asset}_vol_persistence"] = (
                    vol_squared.rolling(63).apply(
                        lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
                    )
                )
                
            except Exception as e:
                print(f"     ⚠️ GARCH failed for {asset}: {e}")
                # Fallback to simple realized volatility
                vol_features[f"{asset}_garch_vol"] = returns[asset].rolling(21).std()
        
        print(f"   ✅ Created {len([c for c in vol_features.columns if 'vol' in c])} volatility features")
        return vol_features
    
    def create_market_structure_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create market microstructure and trend features
        """
        print("📊 Creating market structure features...")
        
        structure_features = pd.DataFrame(index=prices.index)
        
        for asset in prices.columns:
            # Trend strength (R² of linear regression)
            for window in [21, 63, 126]:
                def trend_strength(price_series):
                    if len(price_series) < 10:
                        return 0
                    x = np.arange(len(price_series))
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, price_series.values)
                        return r_value ** 2  # R-squared
                    except:
                        return 0
                
                structure_features[f"{asset}_trend_strength_{window}d"] = (
                    prices[asset].rolling(window).apply(trend_strength)
                )
            
            # Price level relative to moving averages
            for ma_window in [21, 63, 252]:
                ma = prices[asset].rolling(ma_window).mean()
                structure_features[f"{asset}_price_vs_ma{ma_window}"] = (
                    prices[asset] / ma - 1
                )
            
            # Return skewness and kurtosis (regime indicators)
            for window in [21, 63]:
                structure_features[f"{asset}_skewness_{window}d"] = (
                    returns[asset].rolling(window).skew()
                )
                structure_features[f"{asset}_kurtosis_{window}d"] = (
                    returns[asset].rolling(window).kurt()
                )
        
        print(f"   ✅ Created {len(structure_features.columns)} market structure features")
        return structure_features
    
    def get_external_market_data(self) -> pd.DataFrame:
        """
        Download external market indicators (VIX, Treasury rates, etc.)
        """
        print("🌍 Downloading external market indicators...")
        
        external_tickers = {
            '^VIX': 'VIX',           # Market fear gauge
            '^TNX': 'Treasury_10Y',   # 10-year Treasury
            '^IRX': 'Treasury_3M',    # 3-month Treasury
            'DX-Y.NYB': 'DXY',       # Dollar Index
        }
        
        external_data = {}
        
        for ticker, name in external_tickers.items():
            try:
                data = yf.download(ticker, start=START_DATE, end=END_DATE, 
                                 progress=False, auto_adjust=True)
                if len(data) > 0 and len(data.columns) > 0:
                    # Handle different column structures
                    if isinstance(data.columns, pd.MultiIndex) or isinstance(data.columns[0], tuple):
                        close_col = next((col for col in data.columns if col[0] == 'Close'), None)
                        if close_col:
                            external_data[name] = data[close_col].dropna()
                    else:
                        if 'Close' in data.columns:
                            external_data[name] = data['Close'].dropna()
                        elif len(data.columns) == 1:
                            external_data[name] = data.iloc[:, 0].dropna()
                    
                    print(f"   ✅ {name}: {len(external_data.get(name, []))} observations")
                else:
                    print(f"   ❌ {name}: No data")
            except Exception as e:
                print(f"   ⚠️ {name}: {e}")
        
        if external_data:
            external_df = pd.DataFrame(external_data)
            
            # Create external features
            features = pd.DataFrame(index=external_df.index)
            
            # VIX level and changes
            if 'VIX' in external_df.columns:
                features['VIX_level'] = external_df['VIX']
                features['VIX_change'] = external_df['VIX'].pct_change()
                features['VIX_vs_20'] = (external_df['VIX'] > 20).astype(int)  # Fear threshold
            
            # Yield curve slope
            if 'Treasury_10Y' in external_df.columns and 'Treasury_3M' in external_df.columns:
                features['yield_curve_slope'] = external_df['Treasury_10Y'] - external_df['Treasury_3M']
            
            # Dollar strength
            if 'DXY' in external_df.columns:
                features['DXY_level'] = external_df['DXY']
                features['DXY_momentum'] = external_df['DXY'].pct_change(21)
            
            print(f"   ✅ Created {len(features.columns)} external market features")
            return features
        else:
            print("   ❌ No external data available")
            return pd.DataFrame()
    
    def create_all_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature set for regime detection
        """
        print("🧠 Creating complete feature set for ML regime detection...")
        
        feature_sets = []
        
        # 1. Momentum features
        momentum_features = self.create_momentum_features(prices)
        feature_sets.append(momentum_features)
        
        # 2. Volatility features (your GARCH expertise!)
        vol_features = self.create_volatility_features(returns)
        feature_sets.append(vol_features)
        
        # 3. Market structure features
        structure_features = self.create_market_structure_features(prices, returns)
        feature_sets.append(structure_features)
        
        # 4. External market features
        external_features = self.get_external_market_data()
        if not external_features.empty:
            feature_sets.append(external_features)
        
        # Combine all features
        all_features = pd.concat(feature_sets, axis=1)
        
        # Clean the data
        # Forward fill then backward fill missing values
        all_features = all_features.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining infinite values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(0)
        
        self.features = all_features
        self.feature_names = list(all_features.columns)
        
        print(f"\n📋 Feature Engineering Complete:")
        print(f"   📊 Total features: {len(all_features.columns)}")
        print(f"   📅 Date range: {all_features.index[0]} to {all_features.index[-1]}")
        print(f"   📈 Sample size: {len(all_features)} observations")
        
        # Save features for later use
        features_file = f"{PROCESSED_PATH}ml_features.csv"
        all_features.to_csv(features_file)
        print(f"   💾 Features saved to: {features_file}")
        
        return all_features

# Test the feature engineering
if __name__ == "__main__":
    print("🧪 Testing Feature Engineering Pipeline...")
    
    # Load your data
    prices = pd.read_csv(f"{DATA_PATH}energy_prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(f"{DATA_PATH}energy_log_returns.csv", index_col=0, parse_dates=True)
    
    print(f"📊 Input data:")
    print(f"   Prices: {prices.shape}")
    print(f"   Returns: {returns.shape}")
    
    # Create features
    engineer = MarketFeatureEngineer()
    features = engineer.create_all_features(prices, returns)
    
    print(f"\n📈 Feature summary:")
    print(f"   Shape: {features.shape}")
    print(f"   Memory usage: {features.memory_usage().sum() / 1024**2:.1f} MB")
    
    # Show sample features
    print(f"\n🔍 Sample features (last 5 days):")
    print(features.tail())
    
    print(f"\n✅ Feature engineering test complete!")