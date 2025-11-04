"""
Quick Start: Integrating Regime Detection with Your Existing Code
==================================================================

This script shows exactly how to integrate the new regime detection
module with your existing Week 1 GARCH models and data pipeline.

Run this after you've completed Week 1.

Author: Amalie Berg
"""

# ============================================================================
# STEP 1: Import Your Existing Modules + New Regime Detection
# ============================================================================

# Your existing imports from Week 1
from data.data_pipeline import EnergyDataPipeline
from models.garch_model import fit_garch, forecast_volatility
from models.regime_detection import MarketRegimeDetector
from models.regime_integration import RegimeIntegration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 2: Load Your Existing Data (from Week 1)
# ============================================================================

def load_week1_data():
    """
    Load the data and GARCH results you created in Week 1.
    Adjust paths to match your actual file structure.
    """
    # Option A: If you saved processed data
    data = {
        'WTI_Crude': pd.read_csv('data/processed/WTI_prices.csv', index_col=0, parse_dates=True),
        'Brent_Crude': pd.read_csv('data/processed/Brent_prices.csv', index_col=0, parse_dates=True),
        'Natural_Gas': pd.read_csv('data/processed/NG_prices.csv', index_col=0, parse_dates=True),
        'Heating_Oil': pd.read_csv('data/processed/HO_prices.csv', index_col=0, parse_dates=True),
        'Gasoline': pd.read_csv('data/processed/Gasoline_prices.csv', index_col=0, parse_dates=True),
    }
    
    # Load GARCH volatility forecasts (from Week 1)
    volatility = {
        'WTI_Crude': pd.read_csv('results/WTI_volatility.csv', index_col=0, parse_dates=True),
        'Brent_Crude': pd.read_csv('results/Brent_volatility.csv', index_col=0, parse_dates=True),
        'Natural_Gas': pd.read_csv('results/NG_volatility.csv', index_col=0, parse_dates=True),
        'Heating_Oil': pd.read_csv('results/HO_volatility.csv', index_col=0, parse_dates=True),
        'Gasoline': pd.read_csv('results/Gasoline_volatility.csv', index_col=0, parse_dates=True),
    }
    
    # Calculate returns
    returns = {}
    for asset, prices in data.items():
        returns[asset] = np.log(prices / prices.shift(1)).dropna()
    
    return data, returns, volatility


# Option B: If you have your data in a different format
def load_from_data_pipeline():
    """
    Alternative: Load directly from your data pipeline.
    """
    from data.data_pipeline import download_energy_futures
    
    # Download data (your existing function)
    data = download_energy_futures(
        tickers=['CL=F', 'BZ=F', 'NG=F', 'HO=F', 'RB=F'],
        start_date='2010-01-01',
        end_date='2024-12-31'
    )
    
    # Calculate returns
    returns = {}
    for ticker, prices in data.items():
        returns[ticker] = np.log(prices['Adj Close'] / prices['Adj Close'].shift(1)).dropna()
    
    return data, returns


# ============================================================================
# STEP 3: Run Regime Detection on Your Actual Data
# ============================================================================

def run_regime_detection_on_real_data():
    """
    Complete workflow for adding regime detection to your project.
    """
    print("\n" + "="*80)
    print("INTEGRATING REGIME DETECTION WITH YOUR PROJECT")
    print("="*80 + "\n")
    
    # Load your Week 1 data
    print("Loading Week 1 data...")
    try:
        data, returns, volatility = load_week1_data()
        print(f"✓ Loaded data for {len(data)} assets")
    except FileNotFoundError:
        print("⚠ Week 1 files not found. Make sure you've completed Week 1 first.")
        print("  Run your data_pipeline.py and garch_model.py scripts first.\n")
        return
    
    # Initialize regime detection pipeline
    print("\nInitializing regime detection...")
    pipeline = RegimeIntegration(config={
        'n_regimes': 3,
        'regime_method': 'gmm',
        'output_dir': 'results/regimes'
    })
    
    # Load data into pipeline
    print("Loading data into regime detection pipeline...")
    pipeline.load_data(data, returns)
    pipeline.load_garch_volatility(volatility)
    
    # Detect regimes
    print("\nDetecting market regimes...")
    regime_predictions = pipeline.detect_regimes()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    pipeline.plot_all_regimes(save=True)
    
    # Export data with regime labels
    print("\nExporting regime-labeled data...")
    combined_data = pipeline.export_regime_data()
    
    # Generate report
    print("\nGenerating regime detection report...")
    pipeline.generate_regime_report()
    
    print("\n" + "="*80)
    print("✓ REGIME DETECTION COMPLETE!")
    print("="*80)
    print(f"\nCheck results in: results/regimes/")
    print("\nGenerated files:")
    print("  - regime_detection_report.txt")
    print("  - all_assets_with_regimes.csv  (← Use this for ML training)")
    print("  - {asset}_with_regimes.csv (one per asset)")
    print("  - {asset}_regime_timeline.png (visualizations)")
    print("  - {asset}_regime_characteristics.png")
    
    return pipeline


# ============================================================================
# STEP 4: Quick Check - Inspect Regime Results
# ============================================================================

def inspect_regime_results(pipeline):
    """
    Quick inspection of regime detection results.
    """
    print("\n" + "="*80)
    print("REGIME DETECTION SUMMARY")
    print("="*80 + "\n")
    
    for asset_name, model in pipeline.regime_models.items():
        print(f"\n{asset_name}:")
        print("-" * len(asset_name))
        stats = model.get_regime_statistics()
        
        for idx, row in stats.iterrows():
            print(f"  {row['label']:12s}: {row['pct']:5.1f}% of days, "
                  f"avg return: {row['mean_return']*100:6.2f}%, "
                  f"volatility: {row['std_return']*100:5.2f}%")


# ============================================================================
# STEP 5: Add Regime Features to Your Feature Engineering
# ============================================================================

def add_regimes_to_features():
    """
    Show how to add regime labels to your existing 146 features.
    This will be used in Week 2 Day 4-5 for ML training.
    """
    print("\n" + "="*80)
    print("ADDING REGIME FEATURES TO YOUR FEATURE SET")
    print("="*80 + "\n")
    
    # Load your existing features (from Week 2 Day 1)
    try:
        features = pd.read_csv('data/processed/ml_features.csv', index_col=0, parse_dates=True)
        print(f"✓ Loaded existing features: {features.shape}")
    except FileNotFoundError:
        print("⚠ Feature file not found. This is normal if you haven't run feature engineering yet.")
        return
    
    # Load regime data
    regime_data = pd.read_csv('results/regimes/all_assets_with_regimes.csv', 
                              index_col=0, parse_dates=True)
    print(f"✓ Loaded regime data: {regime_data.shape}")
    
    # Merge features with regime labels
    features_with_regimes = features.merge(
        regime_data[['asset', 'regime', 'regime_label', 'regime_0', 'regime_1', 'regime_2']],
        left_index=True,
        right_index=True,
        how='left'
    )
    
    print(f"\n✓ Combined features + regimes: {features_with_regimes.shape}")
    print(f"\nNew columns added:")
    print("  - regime (numeric: 0, 1, 2)")
    print("  - regime_label (Low_Vol, Medium_Vol, High_Vol)")
    print("  - regime_0, regime_1, regime_2 (one-hot encoded)")
    
    # Save enhanced features
    features_with_regimes.to_csv('data/processed/ml_features_with_regimes.csv')
    print(f"\n✓ Saved to: data/processed/ml_features_with_regimes.csv")
    print(f"  Total features: {features_with_regimes.shape[1]}")
    
    return features_with_regimes


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this script to integrate regime detection with your project.
    """
    print("\n" + "="*80)
    print("REGIME DETECTION - QUICK START INTEGRATION")
    print("="*80)
    
    # Step 1: Run regime detection on your data
    pipeline = run_regime_detection_on_real_data()
    
    if pipeline:
        # Step 2: Inspect results
        inspect_regime_results(pipeline)
        
        # Step 3: Add to feature engineering (optional, for Week 2 Day 4-5)
        # Uncomment when you're ready for ML training:
        # add_regimes_to_features()
        
        print("\n" + "="*80)
        print("✓ INTEGRATION COMPLETE!")
        print("="*80)
        print("\nYou're now ready for:")
        print("  → Week 2 Day 4-5: ML Model Training")
        print("  → Use: results/regimes/all_assets_with_regimes.csv")
        print("\nNext: Build ML models that adapt to different market regimes!")


# ============================================================================
# ALTERNATIVE: Quick Single-Asset Example
# ============================================================================

def quick_single_asset_example():
    """
    Quick example for a single asset if you want to test first.
    """
    print("\nQuick single-asset regime detection example:\n")
    
    # Load one asset
    wti_prices = pd.read_csv('data/processed/WTI_prices.csv', 
                             index_col=0, parse_dates=True)
    wti_returns = np.log(wti_prices / wti_prices.shift(1)).dropna()
    wti_volatility = pd.read_csv('results/WTI_volatility.csv', 
                                  index_col=0, parse_dates=True)
    
    # Detect regimes
    detector = MarketRegimeDetector(n_regimes=3, method='gmm')
    detector.fit(wti_returns, volatility=wti_volatility)
    
    # Get regimes
    regimes = detector.predict(wti_returns, volatility=wti_volatility)
    
    # Print statistics
    print(detector.get_regime_statistics())
    
    # Plot
    fig = detector.plot_regimes(
        wti_returns.index, 
        wti_returns, 
        regimes, 
        'WTI Crude'
    )
    plt.savefig('results/WTI_regime_example.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved to results/WTI_regime_example.png")
    
    return detector, regimes


# Uncomment to run single-asset example:
# quick_single_asset_example()