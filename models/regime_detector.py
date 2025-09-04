# models/regime_detector.py
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    ML-based market regime detection using unsupervised learning
    Combines your GARCH features with market microstructure indicators
    """
    
    def __init__(self, n_regimes=3, method='gmm'):
        self.n_regimes = n_regimes
        self.method = method  # 'gmm' or 'kmeans'
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.model = None
        self.regimes = None
        self.regime_probs = None
        self.feature_importance = None
        
    def prepare_features(self, features: pd.DataFrame, feature_selection=True):
        """
        Prepare features for regime detection
        """
        print(f"🧹 Preparing {features.shape[1]} features for regime detection...")
        
        # Remove features with too little variation
        feature_std = features.std()
        valid_features = feature_std[feature_std > 1e-6].index
        features_clean = features[valid_features].copy()
        
        print(f"   Removed {len(features.columns) - len(valid_features)} low-variation features")
        
        # Remove highly correlated features if requested
        if feature_selection and len(features_clean.columns) > 50:
            print("   Removing highly correlated features...")
            corr_matrix = features_clean.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation > 0.95
            high_corr = [column for column in upper_triangle.columns 
                        if any(upper_triangle[column] > 0.95)]
            
            features_clean = features_clean.drop(columns=high_corr)
            print(f"   Removed {len(high_corr)} highly correlated features")
        
        # Handle any remaining NaN values
        features_clean = features_clean.fillna(method='ffill').fillna(method='bfill')
        features_clean = features_clean.fillna(0)
        
        print(f"   Final feature set: {features_clean.shape[1]} features")
        return features_clean
    
    def fit_regime_model(self, features: pd.DataFrame):
        """
        Fit the regime detection model
        """
        print(f"🤖 Fitting regime detection model...")
        
        # Prepare features
        features_clean = self.prepare_features(features)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # Apply PCA to reduce dimensionality
        features_pca = self.pca.fit_transform(features_scaled)
        
        print(f"   PCA reduced dimensions from {features_scaled.shape[1]} to {features_pca.shape[1]}")
        print(f"   Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Fit clustering model
        if self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
        else:  # kmeans
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10
            )
        
        print(f"   Fitting {self.method.upper()} model with {self.n_regimes} regimes...")
        self.model.fit(features_pca)
        
        # Get regime assignments and probabilities
        self.regimes = self.model.predict(features_pca)
        
        if self.method == 'gmm':
            self.regime_probs = self.model.predict_proba(features_pca)
        else:
            # For K-means, create hard assignments
            n_samples = len(features_pca)
            self.regime_probs = np.zeros((n_samples, self.n_regimes))
            self.regime_probs[np.arange(n_samples), self.regimes] = 1.0
        
        # Calculate silhouette score
        sil_score = silhouette_score(features_pca, self.regimes)
        print(f"   Silhouette score: {sil_score:.3f}")
        
        # Calculate feature importance (PCA component weights)
        self.feature_importance = pd.DataFrame(
            self.pca.components_.T,
            index=features_clean.columns,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
        )
        
        return features_clean.index
    
    def analyze_regimes(self, returns: pd.Series, regime_dates: pd.DatetimeIndex):
        """
        Analyze the characteristics of detected regimes
        """
        print("📊 Analyzing regime characteristics...")
        
        # Align returns with regime dates
        aligned_returns = returns.reindex(regime_dates, method='ffill')
        
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            mask = self.regimes == regime
            regime_returns = aligned_returns[mask]
            
            if len(regime_returns) > 0:
                regime_stats[f'Regime_{regime}'] = {
                    'frequency': mask.sum() / len(mask),
                    'avg_return': regime_returns.mean() * 252,  # Annualized
                    'volatility': regime_returns.std() * np.sqrt(252),  # Annualized
                    'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurt(),
                    'max_drawdown': self._calculate_max_drawdown(regime_returns),
                    'n_observations': len(regime_returns)
                }
        
        regime_df = pd.DataFrame(regime_stats).T
        
        print("\n📈 Regime Statistics:")
        print(regime_df.round(3))
        
        return regime_df
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown for a return series"""
        if len(returns) < 2:
            return 0
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def label_regimes_by_momentum(self, returns: pd.Series, regime_dates: pd.DatetimeIndex):
        """
        Label regimes based on momentum performance
        """
        regime_stats = self.analyze_regimes(returns, regime_dates)
        
        # Sort regimes by Sharpe ratio (momentum-friendly measure)
        sorted_regimes = regime_stats.sort_values('sharpe', ascending=False)
        
        regime_labels = {}
        regime_names = ['High_Momentum', 'Medium_Momentum', 'Low_Momentum']
        
        for i, (regime_key, _) in enumerate(sorted_regimes.iterrows()):
            regime_num = int(regime_key.split('_')[1])
            if i < len(regime_names):
                regime_labels[regime_num] = regime_names[i]
            else:
                regime_labels[regime_num] = f'Regime_{regime_num}'
        
        self.regime_labels = regime_labels
        print(f"\n🏷️ Regime Labels:")
        for regime_num, label in regime_labels.items():
            freq = regime_stats.loc[f'Regime_{regime_num}', 'frequency']
            sharpe = regime_stats.loc[f'Regime_{regime_num}', 'sharpe']
            print(f"   Regime {regime_num}: {label} (freq: {freq:.1%}, Sharpe: {sharpe:.2f})")
        
        return regime_labels
    
    def plot_regime_analysis(self, features: pd.DataFrame, returns: pd.Series, prices: pd.Series):
        """
        Create comprehensive regime analysis plots
        """
        print("📈 Creating regime analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Market Regime Analysis', fontsize=16, fontweight='bold')
        
        # Align all data to the same index
        common_index = features.index.intersection(returns.index).intersection(prices.index)
        aligned_regimes = pd.Series(self.regimes, index=features.index).reindex(common_index)
        aligned_returns = returns.reindex(common_index)
        aligned_prices = prices.reindex(common_index)
        
        # 1. Price timeline with regime coloring
        regime_colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        ax1 = axes[0, 0]
        for regime in range(self.n_regimes):
            mask = aligned_regimes == regime
            regime_dates = aligned_regimes.index[mask]
            regime_prices = aligned_prices[mask]
            
            label = self.regime_labels.get(regime, f'Regime {regime}')
            ax1.scatter(regime_dates, regime_prices, 
                       c=regime_colors[regime % len(regime_colors)], 
                       alpha=0.6, s=1, label=label)
        
        ax1.set_title('Price Timeline by Regime')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime transition timeline
        ax2 = axes[0, 1]
        ax2.plot(aligned_regimes.index, aligned_regimes.values, 'o-', markersize=1, alpha=0.7)
        ax2.set_title('Regime Evolution Over Time')
        ax2.set_ylabel('Regime')
        ax2.set_yticks(range(self.n_regimes))
        ax2.set_yticklabels([self.regime_labels.get(i, f'Regime {i}') for i in range(self.n_regimes)])
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns distribution by regime
        ax3 = axes[1, 0]
        for regime in range(self.n_regimes):
            mask = aligned_regimes == regime
            regime_returns = aligned_returns[mask] * 100  # Convert to percentage
            
            if len(regime_returns) > 10:
                label = self.regime_labels.get(regime, f'Regime {regime}')
                ax3.hist(regime_returns, bins=50, alpha=0.6, 
                        color=regime_colors[regime % len(regime_colors)], 
                        label=label, density=True)
        
        ax3.set_title('Return Distributions by Regime')
        ax3.set_xlabel('Daily Returns (%)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature importance (top 15)
        ax4 = axes[1, 1]
        
        # Calculate average absolute component weights
        importance_scores = self.feature_importance.abs().mean(axis=1)
        top_features = importance_scores.nlargest(15)
        
        top_features.plot(kind='barh', ax=ax4, color='steelblue')
        ax4.set_title('Top 15 Most Important Features')
        ax4.set_xlabel('Average Absolute PCA Loading')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(RESULTS_PATH, exist_ok=True)
        plot_path = f'{RESULTS_PATH}regime_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved to: {plot_path}")
        
        plt.show()
        
        return fig
    
    def get_regime_signals(self, regime_labels=None):
        """
        Convert regime predictions to trading signals
        """
        if regime_labels is None:
            regime_labels = getattr(self, 'regime_labels', {})
        
        signals = pd.Series(self.regimes, name='regime')
        
        # Map to momentum-friendly signals
        signal_mapping = {}
        for regime_num, label in regime_labels.items():
            if 'High_Momentum' in label:
                signal_mapping[regime_num] = 1.0
            elif 'Medium_Momentum' in label:
                signal_mapping[regime_num] = 0.5
            else:  # Low momentum or bear
                signal_mapping[regime_num] = 0.0
        
        momentum_signals = signals.map(signal_mapping).fillna(0)
        
        return signals, momentum_signals

# Test the regime detector
if __name__ == "__main__":
    print("🤖 Testing ML Regime Detection...")
    
    # Load features and returns
    features = pd.read_csv(f"{PROCESSED_PATH}ml_features.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv(f"{DATA_PATH}energy_log_returns.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv(f"{DATA_PATH}energy_prices.csv", index_col=0, parse_dates=True)
    
    print(f"📊 Loaded data:")
    print(f"   Features: {features.shape}")
    print(f"   Returns: {returns.shape}")
    print(f"   Prices: {prices.shape}")
    
    # Use WTI crude as primary asset for regime analysis
    wti_returns = returns['WTI_Crude']
    wti_prices = prices['WTI_Crude']
    
    # Fit regime detector
    detector = MarketRegimeDetector(n_regimes=3, method='gmm')
    regime_dates = detector.fit_regime_model(features)
    
    # Analyze regimes
    regime_stats = detector.analyze_regimes(wti_returns, regime_dates)
    regime_labels = detector.label_regimes_by_momentum(wti_returns, regime_dates)
    
    # Create plots
    detector.plot_regime_analysis(features, wti_returns, wti_prices)
    
    # Get trading signals
    regime_signals, momentum_signals = detector.get_regime_signals()
    
    print(f"\n📊 Regime Signal Summary:")
    print(f"   Regime distribution:")
    regime_counts = pd.Series(detector.regimes).value_counts().sort_index()
    for regime, count in regime_counts.items():
        label = regime_labels.get(regime, f'Regime {regime}')
        print(f"     {label}: {count} days ({count/len(detector.regimes):.1%})")
    
    print(f"\n✅ Regime detection test complete!")
    print(f"   {detector.n_regimes} regimes identified")
    print(f"   Results saved to: {RESULTS_PATH}regime_analysis.png")