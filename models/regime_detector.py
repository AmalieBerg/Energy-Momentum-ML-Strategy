# models/regime_detector_FIXED.py (Path Issues Resolved)
import sys
import os

# FIXED: Ensure we're working from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

print(f"📂 Working directory: {os.getcwd()}")

from config import *
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("🤖 ML REGIME DETECTION - PATH-FIXED VERSION")
print("="*60)

# FIXED: Use absolute paths
def get_absolute_path(relative_path):
    """Convert relative path to absolute path from project root"""
    return os.path.join(project_root, relative_path)

# Check required files with BETTER error messages
print("\n🔍 CHECKING REQUIRED FILES...")
print("="*60)

files_to_check = {
    'ML Features': f"{PROCESSED_PATH}ml_features.csv",
    'Log Returns': f"{DATA_PATH}energy_log_returns.csv",
    'Prices': f"{DATA_PATH}energy_prices.csv"
}

all_files_exist = True
for name, path in files_to_check.items():
    abs_path = get_absolute_path(path)
    exists = os.path.isfile(abs_path)
    
    if exists:
        size = os.path.getsize(abs_path) / 1024  # KB
        print(f"✅ {name}: {path}")
        print(f"   Size: {size:.1f} KB")
        print(f"   Full path: {abs_path}")
    else:
        print(f"❌ {name}: {path}")
        print(f"   Looking for: {abs_path}")
        print(f"   File exists: {exists}")
        all_files_exist = False

print("="*60)

if not all_files_exist:
    print("\n❌ Some files are missing!")
    print("\n💡 DEBUGGING TIPS:")
    print("   1. Check if files actually exist in VS Code sidebar")
    print("   2. Run: python path_diagnostic.py")
    print("   3. Check your current directory matches project root")
    print("   4. Try using absolute paths in config.py")
    
    # Show what's actually in the directories
    print("\n📋 What's in data/raw/:")
    try:
        raw_path = get_absolute_path(DATA_PATH)
        if os.path.exists(raw_path):
            for f in os.listdir(raw_path):
                print(f"   • {f}")
        else:
            print(f"   ❌ Directory doesn't exist: {raw_path}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n📋 What's in data/processed/:")
    try:
        processed_path = get_absolute_path(PROCESSED_PATH)
        if os.path.exists(processed_path):
            for f in os.listdir(processed_path):
                print(f"   • {f}")
        else:
            print(f"   ❌ Directory doesn't exist: {processed_path}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    sys.exit(1)

print("\n✅ All required files found!")

# Load data with absolute paths
print("\n📊 Loading data...")
features = pd.read_csv(get_absolute_path(f"{PROCESSED_PATH}ml_features.csv"), 
                       index_col=0, parse_dates=True)
returns = pd.read_csv(get_absolute_path(f"{DATA_PATH}energy_log_returns.csv"), 
                      index_col=0, parse_dates=True)
prices = pd.read_csv(get_absolute_path(f"{DATA_PATH}energy_prices.csv"), 
                     index_col=0, parse_dates=True)

print(f"   Features: {features.shape}")
print(f"   Returns: {returns.shape}")
print(f"   Prices: {prices.shape}")

# Continue with regime detection...
print("\n🧹 Preparing features for regime detection...")

# Remove low-variation features
feature_std = features.std()
valid_features = feature_std[feature_std > 1e-6].index
features_clean = features[valid_features].copy()
print(f"   Removed {len(features.columns) - len(valid_features)} low-variation features")

# Remove highly correlated features
print("   Removing highly correlated features...")
corr_matrix = features_clean.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr = [column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.95)]
features_clean = features_clean.drop(columns=high_corr)
print(f"   Removed {len(high_corr)} highly correlated features")
print(f"   Final feature set: {features_clean.shape[1]} features")

# Handle missing values
features_clean = features_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
features_clean = features_clean.replace([np.inf, -np.inf], 0)

# Standardize features
print("\n🤖 Fitting regime detection model...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_clean)

# Apply PCA
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_scaled)
print(f"   PCA reduced dimensions from {features_scaled.shape[1]} to {features_pca.shape[1]}")
print(f"   Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

# Fit GMM
print(f"   Fitting GMM model with 3 regimes...")
n_regimes = 3
gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', 
                      random_state=42, max_iter=200)
gmm.fit(features_pca)

# Get regime assignments
regimes = gmm.predict(features_pca)
regime_probs = gmm.predict_proba(features_pca)

# Silhouette score
sil_score = silhouette_score(features_pca, regimes)
print(f"   Silhouette score: {sil_score:.3f}")

# Analyze regimes
print("\n📊 Analyzing regime characteristics...")

# Use WTI crude as reference asset
wti_returns = returns['WTI_Crude'].reindex(features_clean.index)

regime_stats = []
for regime in range(n_regimes):
    mask = regimes == regime
    regime_returns = wti_returns[mask]
    
    if len(regime_returns) > 0:
        stats = {
            'Regime': f'Regime_{regime}',
            'frequency': mask.sum() / len(mask),
            'avg_return': regime_returns.mean() * 252,  # Annualized
            'volatility': regime_returns.std() * np.sqrt(252),  # Annualized
            'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0,
            'skewness': regime_returns.skew(),
            'kurtosis': regime_returns.kurt(),
            'n_observations': len(regime_returns)
        }
        regime_stats.append(stats)

regime_df = pd.DataFrame(regime_stats)

print("\n📈 Regime Statistics:")
print(regime_df.round(3))

# Label regimes by Sharpe ratio
sorted_regimes = regime_df.sort_values('sharpe', ascending=False)
regime_labels = {}
regime_names = ['High_Momentum', 'Medium_Momentum', 'Low_Momentum']

for i, (idx, row) in enumerate(sorted_regimes.iterrows()):
    regime_num = int(row['Regime'].split('_')[1])
    regime_labels[regime_num] = regime_names[i]

print(f"\n🏷️ Regime Labels:")
for regime_num, label in regime_labels.items():
    freq = regime_df.loc[regime_df['Regime'] == f'Regime_{regime_num}', 'frequency'].values[0]
    sharpe = regime_df.loc[regime_df['Regime'] == f'Regime_{regime_num}', 'sharpe'].values[0]
    print(f"   Regime {regime_num}: {label} (freq: {freq:.1%}, Sharpe: {sharpe:.2f})")

# Create output dataframe
print("\n💾 Exporting regime data for ML training...")
regime_output = pd.DataFrame(index=features_clean.index)
regime_output['regime'] = regimes
regime_output['regime_label'] = [regime_labels[r] for r in regimes]

# Add regime probabilities
for i in range(n_regimes):
    regime_output[f'regime_{i}_prob'] = regime_probs[:, i]

# Save with absolute path
os.makedirs(get_absolute_path(PROCESSED_PATH), exist_ok=True)
output_path = get_absolute_path(f"{PROCESSED_PATH}regime_labels.csv")
regime_output.to_csv(output_path)
print(f"   Saved to: {output_path}")
print(f"   Shape: {regime_output.shape}")
print(f"   Columns: {list(regime_output.columns)}")

# Create plots
print("\n📈 Creating regime analysis plots...")
os.makedirs(get_absolute_path(RESULTS_PATH), exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Market Regime Analysis - WTI Crude', fontsize=16, fontweight='bold')

# Plot 1: Price timeline with regime coloring
ax1 = axes[0, 0]
wti_prices = prices['WTI_Crude'].reindex(features_clean.index)
regime_colors = ['red', 'blue', 'green']

for regime in range(n_regimes):
    mask = regimes == regime
    regime_dates = features_clean.index[mask]
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
ax2.plot(features_clean.index, regimes, 'o-', markersize=1, alpha=0.7)
ax2.set_title('Regime Evolution Over Time')
ax2.set_ylabel('Regime')
ax2.set_yticks(range(n_regimes))
ax2.set_yticklabels([regime_labels[i] for i in range(n_regimes)])
ax2.grid(True, alpha=0.3)

# Plot 3: Return distributions by regime
ax3 = axes[1, 0]
for regime in range(n_regimes):
    mask = regimes == regime
    regime_returns = wti_returns[mask] * 100  # Percentage
    
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
x = np.arange(n_regimes)
width = 0.25

sharpes = [regime_df.loc[regime_df['Regime'] == f'Regime_{i}', 'sharpe'].values[0] 
           for i in range(n_regimes)]
vols = [regime_df.loc[regime_df['Regime'] == f'Regime_{i}', 'volatility'].values[0] 
        for i in range(n_regimes)]
freqs = [regime_df.loc[regime_df['Regime'] == f'Regime_{i}', 'frequency'].values[0] * 100 
         for i in range(n_regimes)]

ax4.bar(x - width, sharpes, width, label='Sharpe Ratio', alpha=0.8)
ax4.bar(x, [v*10 for v in vols], width, label='Volatility (×10)', alpha=0.8)
ax4.bar(x + width, freqs, width, label='Frequency (%)', alpha=0.8)

ax4.set_title('Regime Characteristics')
ax4.set_xticks(x)
ax4.set_xticklabels([regime_labels[i] for i in range(n_regimes)])
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = get_absolute_path(f'{RESULTS_PATH}regime_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"   Plot saved to: {plot_path}")
plt.close()

print("\n" + "="*60)
print("✅ REGIME DETECTION COMPLETE!")
print("="*60)

print(f"\n📁 Output files:")
print(f"   • {output_path}")
print(f"   • {plot_path}")

print(f"\n🎯 Next step: Use regime_labels.csv in ML model training")
print(f"   Total features for ML: {features.shape[1]} + {regime_output.shape[1] - 1} = {features.shape[1] + regime_output.shape[1] - 1}")

print("\n📊 Regime Distribution:")
for regime in range(n_regimes):
    count = (regimes == regime).sum()
    pct = count / len(regimes) * 100
    label = regime_labels[regime]
    print(f"   {label}: {count} days ({pct:.1f}%)")