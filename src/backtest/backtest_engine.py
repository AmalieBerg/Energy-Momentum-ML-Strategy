"""
Backtesting Engine - Using ML Features + Actual Prices

CRITICAL RULES:
1. Uses ml_features.csv for predictions (compatible with trained models)
2. Uses ACTUAL PRICE DATA for return calculations
3. NEVER uses momentum as a return proxy
4. Filters price columns from prediction features
5. Fails with clear error if price data is missing
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import *
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import warnings

warnings.filterwarnings('ignore')

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Raw data file mapping
RAW_FILES = {
    'WTI_Crude': 'CL_daily.csv',
    'Brent_Crude': 'BZ_daily.csv',
    'Natural_Gas': 'NG_daily.csv',
    'Heating_Oil': 'HO_daily.csv',
    'Gasoline': 'RB_daily.csv'
}


def load_existing_features():
    """Load the ml_features.csv that was used for training"""
    logger.info("\n" + "="*70)
    logger.info("LOADING ML FEATURES")
    logger.info("="*70)
    
    features_path = PROCESSED_PATH / 'ml_features.csv'
    logger.info(f"\nLoading: {features_path}")
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"ml_features.csv not found at {features_path}\n"
            f"This file should contain the features used to train your models."
        )
    
    df = pd.read_csv(features_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info(f"✓ Loaded features")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def load_price_data():
    """Load actual price data for return calculations"""
    logger.info("\n" + "="*70)
    logger.info("LOADING PRICE DATA FOR RETURNS")
    logger.info("="*70)
    
    # Try cached prices first
    cached_path = PROCESSED_PATH / 'cached_prices_and_features.csv'
    if cached_path.exists():
        logger.info(f"\nTrying cached file: {cached_path}")
        df = pd.read_csv(cached_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check if it has price columns
        price_cols = [col for col in df.columns if '_Close' in col]
        if len(price_cols) > 0:
            logger.info(f"✓ Found {len(price_cols)} price columns in cached file")
            df_prices = df[['Date'] + price_cols]
            logger.info(f"  Rows: {len(df_prices)}")
            return df_prices
    
    # Try raw data files
    logger.info(f"\nLoading from raw data files in: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {DATA_PATH}\n"
            f"Cannot calculate returns without price data!"
        )
    
    prices = {}
    missing_files = []
    
    for asset, filename in RAW_FILES.items():
        filepath = DATA_PATH / filename
        
        if not filepath.exists():
            missing_files.append(f"{asset} ({filename})")
            continue
        
        try:
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Take Close price
            if 'Close' not in df.columns:
                logger.warning(f"  WARNING: No 'Close' column in {filename}")
                continue
            
            df = df[['Date', 'Close']].rename(columns={'Close': f'{asset}_Close'})
            prices[asset] = df
            logger.info(f"  ✓ Loaded {asset}: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"  ERROR loading {filename}: {e}")
            missing_files.append(f"{asset} ({filename})")
    
    if len(prices) == 0:
        raise FileNotFoundError(
            f"No price data could be loaded!\n"
            f"Missing files: {missing_files}\n"
            f"Cannot run backtest without actual price data.\n"
            f"CRITICAL: Momentum cannot be used as a proxy for returns!"
        )
    
    if len(missing_files) > 0:
        logger.warning(f"\n  WARNING: Missing price data for: {missing_files}")
        logger.warning(f"  These assets will be skipped in backtesting")
    
    # Merge all price dataframes
    df_prices = prices[list(prices.keys())[0]].copy()
    for asset, df in list(prices.items())[1:]:
        df_prices = pd.merge(df_prices, df, on='Date', how='outer')
    
    df_prices = df_prices.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"\n✓ Merged price data")
    logger.info(f"  Assets: {list(prices.keys())}")
    logger.info(f"  Total rows: {len(df_prices)}")
    logger.info(f"  Date range: {df_prices['Date'].min()} to {df_prices['Date'].max()}")
    
    return df_prices


def load_regime_labels():
    """Load regime labels if available (optional)"""
    regime_path = PROCESSED_PATH / 'regime_labels.csv'
    
    if regime_path.exists():
        logger.info(f"\nLoading regime labels: {regime_path}")
        df = pd.read_csv(regime_path)
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info("  ✓ Regime labels loaded")
        return df
    
    logger.info("\n  Note: No regime labels found (optional)")
    return None


def merge_features_and_prices():
    """Merge ML features with price data"""
    logger.info("\n" + "="*70)
    logger.info("MERGING FEATURES AND PRICES")
    logger.info("="*70)
    
    # Load features (for ML predictions)
    df_features = load_existing_features()
    
    # Load regime labels (optional)
    df_regimes = load_regime_labels()
    if df_regimes is not None:
        df_features = pd.merge(df_features, df_regimes, on='Date', how='left')
    
    # Load prices (for return calculations)
    df_prices = load_price_data()
    
    # Merge
    df = pd.merge(df_features, df_prices, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"\n✓ Merged features and prices")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Total columns: {len(df.columns)}")
    logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Verify we have prices
    price_cols = [col for col in df.columns if '_Close' in col]
    logger.info(f"  Price columns: {len(price_cols)}")
    
    return df


def load_model(asset: str, model_name: str):
    """Load trained model, scaler, and encoder"""
    model_path = MODEL_PATH / f"{asset}_{model_name}_model.pkl"
    scaler_path = MODEL_PATH / f"{asset}_{model_name}_scaler.pkl"
    encoder_path = MODEL_PATH / f"{asset}_{model_name}_encoder.pkl"
    
    if not model_path.exists():
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None
    
    return model, scaler, encoder


def prepare_features_for_prediction(df: pd.DataFrame, asset: str):
    """
    Prepare features for prediction - CRITICAL: exclude price columns
    
    Features are used for prediction only, not for return calculation
    """
    # Get all asset-specific features
    asset_features = [col for col in df.columns if col.startswith(asset)]
    
    # CRITICAL: Filter out price columns
    price_keywords = ['_Close', '_Open', '_High', '_Low', '_Volume', '_Adj_Close']
    asset_features = [
        col for col in asset_features
        if not any(keyword in col for keyword in price_keywords)
    ]
    
    # Market features
    market_features = [
        'VIX_level', 'VIX_change', 'VIX_vs_20',
        'yield_curve_slope',
        'DXY_level', 'DXY_momentum'
    ]
    
    feature_cols = asset_features + [col for col in market_features if col in df.columns]
    
    # Select only numeric features
    X = df[feature_cols].select_dtypes(include=['number']).copy()
    
    # Handle infinite and missing values (fixed deprecated methods)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().fillna(0)
    
    return X


def generate_signals(df: pd.DataFrame, asset: str, model_name='Random_Forest'):
    """Generate trading signals using ML model"""
    logger.info(f"\nGenerating signals for {asset} using {model_name}...")
    
    # Load model
    model, scaler, encoder = load_model(asset, model_name)
    
    if model is None:
        logger.warning(f"  WARNING: Model not found for {asset}")
        return pd.Series(0, index=df.index), pd.Series(np.nan, index=df.index)
    
    # Prepare features (excluding price columns)
    X = prepare_features_for_prediction(df, asset)
    
    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Feature names: {X.columns.tolist()[:5]}... ({len(X.columns)} total)")
    
    if X.shape[1] == 0:
        logger.error(f"  ERROR: No features found for {asset}")
        return pd.Series(0, index=df.index), pd.Series(np.nan, index=df.index)
    
    # Scale features
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    # Predict regimes
    regime_pred_encoded = model.predict(X_scaled)
    
    if encoder is not None:
        regime_pred = encoder.inverse_transform(regime_pred_encoded)
    else:
        regime_pred = regime_pred_encoded
    
    # Convert regimes to signals
    signal_map = {0: -1, 1: 0, 2: 1}
    signals = pd.Series([signal_map.get(r, 0) for r in regime_pred], index=df.index)
    
    logger.info(f"  Signals: Long={sum(signals==1)}, Neutral={sum(signals==0)}, Short={sum(signals==-1)}")
    
    return signals, pd.Series(regime_pred, index=df.index)


def calculate_returns(df: pd.DataFrame, asset: str, signals: pd.Series):
    """Calculate strategy returns using ACTUAL PRICES ONLY"""
    price_col = f'{asset}_Close'
    
    # Check if we have price data
    if price_col not in df.columns:
        raise ValueError(
            f"CRITICAL ERROR: No price data for {asset}!\n"
            f"Expected column: {price_col}\n"
            f"Cannot calculate returns without actual prices."
        )
    
    # Calculate actual returns from prices
    asset_returns = df[price_col].pct_change()
    
    # Strategy returns = signal * asset returns (shifted to avoid look-ahead bias)
    strategy_returns = signals.shift(1) * asset_returns
    
    # Transaction costs when signal changes
    signal_changes = signals.diff().abs()
    transaction_costs = signal_changes * BACKTEST_CONFIG['transaction_cost']
    strategy_returns = strategy_returns - transaction_costs
    
    return strategy_returns


def calculate_performance_metrics(strategy_returns: pd.Series, buy_hold_returns: pd.Series):
    """Calculate performance metrics"""
    strategy_returns = strategy_returns.fillna(0)
    buy_hold_returns = buy_hold_returns.fillna(0)
    
    # Total return
    total_return = (1 + strategy_returns).prod() - 1
    n_years = len(strategy_returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Sharpe ratio
    if strategy_returns.std() > 0:
        sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_days = (strategy_returns > 0).sum()
    total_days = (strategy_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Buy & hold
    buy_hold_return = (1 + buy_hold_returns).prod() - 1
    outperformance = total_return - buy_hold_return
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'volatility': strategy_returns.std() * np.sqrt(252),
        'buy_hold_return': buy_hold_return,
        'outperformance': outperformance
    }


def plot_backtest_results(df, strategy_cum, buy_hold_cum, signals, asset, model_name):
    """Create backtest visualization"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Cumulative returns
    axes[0].plot(df['Date'], strategy_cum, label='Strategy', linewidth=2, color='#2E86AB')
    axes[0].plot(df['Date'], buy_hold_cum, label='Buy & Hold', linewidth=2, alpha=0.7, color='#A23B72')
    axes[0].set_title(f'{asset} - {model_name}: Cumulative Returns', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1, color='black', linestyle='--', alpha=0.3)
    
    # Signals
    colors = ['red' if s == -1 else 'gray' if s == 0 else 'green' for s in signals]
    axes[1].scatter(df['Date'], signals, c=colors, alpha=0.5, s=10)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1].set_title('Trading Signals', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Signal', fontsize=10)
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(['Short', 'Neutral', 'Long'])
    axes[1].grid(True, alpha=0.3)
    
    # Drawdown
    cumulative = strategy_cum
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    axes[2].fill_between(df['Date'], drawdown, 0, alpha=0.3, color='red')
    axes[2].plot(df['Date'], drawdown, color='red', linewidth=1)
    axes[2].set_title('Strategy Drawdown', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Drawdown', fontsize=10)
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'backtest_{asset}_{model_name}.png'
    filepath = RESULTS_PATH / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"  Plot saved: {filename}")
    plt.close()


def backtest_asset(df, asset, model_name='Random_Forest', plot=True):
    """Run backtest for single asset"""
    logger.info("\n" + "="*70)
    logger.info(f"BACKTESTING: {asset} - {model_name}")
    logger.info("="*70)
    
    # Check if we have price data for this asset
    price_col = f'{asset}_Close'
    if price_col not in df.columns:
        logger.warning(f"\n  SKIPPING {asset}: No price data available")
        return None
    
    # Split train/test
    df_train = df[df['Date'] <= BACKTEST_CONFIG['train_end_date']].copy()
    df_test = df[df['Date'] >= BACKTEST_CONFIG['test_start_date']].copy()
    
    logger.info(f"\nTrain: {df_train['Date'].min()} to {df_train['Date'].max()} ({len(df_train)} days)")
    logger.info(f"Test:  {df_test['Date'].min()} to {df_test['Date'].max()} ({len(df_test)} days)")
    
    # Generate signals
    signals, predicted_regimes = generate_signals(df_test, asset, model_name)
    
    # Calculate returns using ACTUAL PRICES
    try:
        strategy_returns = calculate_returns(df_test, asset, signals)
    except ValueError as e:
        logger.error(f"\n  ERROR: {e}")
        return None
    
    # Buy & hold returns
    buy_hold_returns = df_test[price_col].pct_change()
    
    # Cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
    
    # Metrics
    metrics = calculate_performance_metrics(strategy_returns, buy_hold_returns)
    
    logger.info(f"\n{'='*50}")
    logger.info("PERFORMANCE METRICS")
    logger.info(f"{'='*50}")
    logger.info(f"Total Return:        {metrics['total_return']:>10.2%}")
    logger.info(f"Annual Return:       {metrics['annual_return']:>10.2%}")
    logger.info(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    logger.info(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
    logger.info(f"Win Rate:            {metrics['win_rate']:>10.2%}")
    logger.info(f"\n{'VS BUY & HOLD':^50}")
    logger.info(f"B&H Return:          {metrics['buy_hold_return']:>10.2%}")
    logger.info(f"Outperformance:      {metrics['outperformance']:>10.2%}")
    
    # Plot
    if plot:
        plot_backtest_results(df_test, strategy_cumulative, buy_hold_cumulative,
                            signals, asset, model_name)
    
    results = {
        'asset': asset,
        'model': model_name,
        **metrics,
        'test_start': df_test['Date'].min(),
        'test_end': df_test['Date'].max(),
        'n_trades': int(signals.diff().abs().sum() / 2)
    }
    
    return results


def run_backtest(model_name='Random_Forest'):
    """Run backtest for all assets"""
    logger.info("\n" + "="*70)
    logger.info(f"BACKTEST - USING ACTUAL PRICES")
    logger.info(f"Model: {model_name}")
    logger.info("="*70)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Initial Capital:     ${BACKTEST_CONFIG['initial_capital']:,.0f}")
    logger.info(f"  Train Period:        Up to {BACKTEST_CONFIG['train_end_date']}")
    logger.info(f"  Test Period:         {BACKTEST_CONFIG['test_start_date']} onwards")
    logger.info(f"  Position Size:       {BACKTEST_CONFIG['position_size']:.0%}")
    logger.info(f"  Transaction Cost:    {BACKTEST_CONFIG['transaction_cost']:.2%}")
    
    # Load data
    df = merge_features_and_prices()
    
    # Backtest each asset
    all_results = []
    
    for asset in ASSETS:
        try:
            results = backtest_asset(df, asset, model_name, plot=True)
            if results:
                all_results.append(results)
        except Exception as e:
            logger.error(f"\n{'='*70}")
            logger.error(f"ERROR BACKTESTING {asset}")
            logger.error(f"{'='*70}")
            logger.error(f"{e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = RESULTS_PATH / f'backtest_results_{model_name}_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        
        logger.info("\n" + "="*70)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("="*70)
        
        for _, row in results_df.iterrows():
            logger.info(f"\n{row['asset']}:")
            logger.info(f"  Annual Return: {row['annual_return']:.2%}")
            logger.info(f"  Sharpe Ratio:  {row['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown:  {row['max_drawdown']:.2%}")
            logger.info(f"  Outperformance: {row['outperformance']:.2%}")
        
        logger.info(f"\n{'='*70}")
        logger.info("PORTFOLIO STATISTICS")
        logger.info(f"{'='*70}")
        logger.info(f"Avg Annual Return:    {results_df['annual_return'].mean():.2%}")
        logger.info(f"Avg Sharpe Ratio:     {results_df['sharpe_ratio'].mean():.2f}")
        logger.info(f"Avg Outperformance:   {results_df['outperformance'].mean():.2%}")
        
        logger.info(f"\n✓ Results saved: {results_path}")
    else:
        logger.error("\n" + "="*70)
        logger.error("ERROR: NO SUCCESSFUL BACKTESTS")
        logger.error("="*70)
    
    return all_results


if __name__ == "__main__":
    MODEL_TO_TEST = 'Random_Forest'
    
    logger.info("\n" + "="*70)
    logger.info("STARTING BACKTEST")
    logger.info("="*70)
    
    try:
        results = run_backtest(model_name=MODEL_TO_TEST)
        logger.info("\n" + "="*70)
        logger.info("BACKTEST COMPLETED")
        logger.info("="*70)
        
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("ERROR OCCURRED")
        logger.error("="*70)
        logger.error(f"\n{e}")
        import traceback
        traceback.print_exc()
