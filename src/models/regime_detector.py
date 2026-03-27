"""
ML Regime Detection for Energy Markets
Uses GMM clustering to identify market regimes
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import *
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """Load features, returns, and prices"""
    logger.info("\n" + "="*70)
    logger.info("LOADING DATA FOR REGIME DETECTION")
    logger.info("="*70)
    
    # Check if files exist
    required_files = {
        'ML Features': PROCESSED_PATH / 'ml_features.csv',
        'Log Returns': DATA_PATH / 'energy_log_returns.csv',
        'Prices': DATA_PATH / 'energy_prices.csv'
    }
    
    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Please run data_pipeline.py and feature_engineering.py first"
            )
        logger.info(f"  ✅ Found {name}: {path.name}")
    
    # Load data
    features = pd.read_csv(required_files['ML Features'], index_col=0, parse_dates=True)
    returns = pd.read_csv(required_files['Log Returns'], index_col=0, parse_dates=True)
    prices = pd.read_csv(required_files['Prices'], index_col=0, parse_dates=True)
    
    logger.info(f"\n  Features: {features.shape}")
    logger.info(f"  Returns: {returns.shape}")
    logger.info(f"  Prices: {prices.shape}")
    
    # CRITICAL: Align all dataframes to the same index (intersection)
    common_index = features.index.intersection(returns.index).intersection(prices.index)
    features = features.loc[common_index]
    returns = returns.loc[common_index]
    prices = prices.loc[common_index]
    
    logger.info(f"\n  After alignment:")
    logger.info(f"  Features: {features.shape}")
    logger.info(f"  Returns: {returns.shape}")
    logger.info(f"  Prices: {prices.shape}")
    
    return features, returns, prices


def prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare features for regime detection"""
    logger.info("\n🧹 Preparing features for regime detection...")
    
    # Remove low-variation features
    feature_std = features.std()
    valid_features = feature_std[feature_std > 1e-6].index
    features_clean = features[valid_features].copy()
    logger.info(f"   Removed {len(features.columns) - len(valid_features)} low-variation features")
    
    # Remove highly correlated features
    logger.info("   Removing highly correlated features...")
    corr_matrix = features_clean.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr = [column for column in upper_triangle.columns 
                if any(upper_triangle[column] > 0.95)]
    features_clean = features_clean.drop(columns=high_corr)
    logger.info(f"   Removed {len(high_corr)} highly correlated features")
    logger.info(f"   Final feature set: {features_clean.shape[1]} features")
    
    # Handle missing values (fixed deprecated methods)
    features_clean = features_clean.ffill().bfill().fillna(0)
    features_clean = features_clean.replace([np.inf, -np.inf], 0)
    
    return features_clean


def fit_regime_model(features: pd.DataFrame, n_regimes: int = 3) -> tuple:
    """Fit GMM regime detection model"""
    logger.info("\n🤖 Fitting regime detection model...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(features_scaled)
    logger.info(f"   PCA reduced dimensions from {features_scaled.shape[1]} to {features_pca.shape[1]}")
    logger.info(f"   Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Fit GMM
    logger.info(f"   Fitting GMM model with {n_regimes} regimes...")
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type='full',
        random_state=42,
        max_iter=200
    )
    gmm.fit(features_pca)
    
    # Get regime assignments
    regimes = gmm.predict(features_pca)
    regime_probs = gmm.predict_proba(features_pca)
    
    # Silhouette score
    sil_score = silhouette_score(features_pca, regimes)
    logger.info(f"   Silhouette score: {sil_score:.3f}")
    
    return regimes, regime_probs, gmm, scaler, pca


def analyze_regimes(regimes: np.ndarray, returns: pd.DataFrame, n_regimes: int = 3) -> tuple:
    """Analyze regime characteristics"""
    logger.info("\n📊 Analyzing regime characteristics...")
    
    # Use WTI crude as reference asset
    wti_returns = returns['WTI_Crude']
    
    # Data is already aligned, so regimes and returns have same length
    regime_stats = []
    for regime in range(n_regimes):
        mask = regimes == regime
        regime_returns = wti_returns[mask]
        
        if len(regime_returns) > 0:
            stats = {
                'Regime': f'Regime_{regime}',
                'frequency': mask.sum() / len(mask),
                'avg_return': regime_returns.mean() * 252,  # Annualized
                'volatility': regime_returns.std() * np.sqrt(252),
                'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0,
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurt(),
                'n_observations': len(regime_returns)
            }
            regime_stats.append(stats)
    
    regime_df = pd.DataFrame(regime_stats)
    
    logger.info("\n📈 Regime Statistics:")
    logger.info(f"\n{regime_df.round(3)}")
    
    # Label regimes by Sharpe ratio
    sorted_regimes = regime_df.sort_values('sharpe', ascending=False)
    regime_labels = {}
    regime_names = ['High_Momentum', 'Medium_Momentum', 'Low_Momentum']
    
    for i, (idx, row) in enumerate(sorted_regimes.iterrows()):
        regime_num = int(row['Regime'].split('_')[1])
        regime_labels[regime_num] = regime_names[i]
    
    logger.info(f"\n🏷️ Regime Labels:")
    for regime_num, label in regime_labels.items():
        freq = regime_df.loc[regime_df['Regime'] == f'Regime_{regime_num}', 'frequency'].values[0]
        sharpe = regime_df.loc[regime_df['Regime'] == f'Regime_{regime_num}', 'sharpe'].values[0]
        logger.info(f"   Regime {regime_num}: {label} (freq: {freq:.1%}, Sharpe: {sharpe:.2f})")
    
    return regime_df, regime_labels


def save_regime_data(regimes: np.ndarray, regime_probs: np.ndarray, 
                     regime_labels: dict, features_index: pd.Index, n_regimes: int = 3):
    """Save regime labels for ML training"""
    logger.info("\n💾 Exporting regime data for ML training...")
    
    # Data is already aligned, all same length
    regime_output = pd.DataFrame(index=features_index)
    regime_output['regime'] = regimes
    regime_output['regime_label'] = [regime_labels[r] for r in regimes]
    
    # Add regime probabilities
    for i in range(n_regimes):
        regime_output[f'regime_{i}_prob'] = regime_probs[:, i]
    
    # Save
    output_path = PROCESSED_PATH / 'regime_labels.csv'
    regime_output.to_csv(output_path)
    logger.info(f"   Saved to: {output_path}")
    logger.info(f"   Shape: {regime_output.shape}")
    logger.info(f"   Columns: {list(regime_output.columns)}")
    
    return regime_output


def create_regime_plots(regimes: np.ndarray, regime_labels: dict, 
                        prices: pd.DataFrame, returns: pd.DataFrame,
                        features_index: pd.Index, n_regimes: int = 3):
    """Create regime analysis visualizations"""
    logger.info("\n📈 Creating regime analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Market Regime Analysis - WTI Crude', fontsize=16, fontweight='bold')
    
    # Data is already aligned
    wti_prices = prices['WTI_Crude']
    wti_returns = returns['WTI_Crude']
    
    regime_colors = ['red', 'blue', 'green']
    
    # Plot 1: Price timeline with regime coloring
    ax1 = axes[0, 0]
    for regime in range(n_regimes):
        mask = regimes == regime
        regime_dates = features_index[mask]
        regime_prices = wti_prices[mask]
        label = regime_labels[regime]
        ax1.scatter(regime_dates, regime_prices,
                   c=regime_colors[regime], alpha=0.6, s=1, label=label)
    
    ax1.set_title('WTI Crude Price by Regime')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime timeline
    ax2 = axes[0, 1]
    ax2.plot(features_index, regimes, 'o-', markersize=1, alpha=0.7)
    ax2.set_title('Regime Evolution Over Time')
    ax2.set_ylabel('Regime')
    ax2.set_yticks(range(n_regimes))
    ax2.set_yticklabels([regime_labels[i] for i in range(n_regimes)])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Return distributions by regime
    ax3 = axes[1, 0]
    for regime in range(n_regimes):
        mask = regimes == regime
        regime_returns = wti_returns[mask] * 100
        
        if len(regime_returns) > 10:
            label = regime_labels[regime]
            ax3.hist(regime_returns, bins=50, alpha=0.6,
                    color=regime_colors[regime], label=label, density=True)
    
    ax3.set_title('Return Distributions by Regime')
    ax3.set_xlabel('Daily Returns (%)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Regime characteristics
    ax4 = axes[1, 1]
    regime_counts = pd.Series(regimes).value_counts().sort_index()
    bars = ax4.bar(range(n_regimes), regime_counts.values, alpha=0.8)
    for i, bar in enumerate(bars):
        bar.set_color(regime_colors[i])
    ax4.set_title('Regime Frequency')
    ax4.set_xlabel('Regime')
    ax4.set_ylabel('Count')
    ax4.set_xticks(range(n_regimes))
    ax4.set_xticklabels([regime_labels[i] for i in range(n_regimes)])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = RESULTS_PATH / 'regime_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"   Plot saved to: {plot_path}")
    plt.close()


def main():
    """Main regime detection pipeline"""
    logger.info("\n" + "="*70)
    logger.info("🤖 ML REGIME DETECTION PIPELINE")
    logger.info("="*70)
    
    # Load data
    features, returns, prices = load_data()
    
    # Prepare features
    features_clean = prepare_features(features)
    
    # Fit regime model
    n_regimes = ML_CONFIG['n_regimes']
    regimes, regime_probs, gmm, scaler, pca = fit_regime_model(features_clean, n_regimes)
    
    # Analyze regimes
    regime_df, regime_labels = analyze_regimes(regimes, returns, n_regimes)
    
    # Save regime data
    regime_output = save_regime_data(regimes, regime_probs, regime_labels, 
                                      features_clean.index, n_regimes)
    
    # Create plots
    create_regime_plots(regimes, regime_labels, prices, returns, 
                        features_clean.index, n_regimes)
    
    logger.info("\n" + "="*70)
    logger.info("✅ REGIME DETECTION COMPLETE!")
    logger.info("="*70)
    
    logger.info(f"\n📊 Regime Distribution:")
    for regime in range(n_regimes):
        count = (regimes == regime).sum()
        pct = count / len(regimes) * 100
        label = regime_labels[regime]
        logger.info(f"   {label}: {count} days ({pct:.1f}%)")
    
    logger.info(f"\n🎯 Next step: Use regime_labels.csv in ML model training")
    logger.info(f"   Total samples: {len(regime_output)}")


if __name__ == "__main__":
    main()
