"""
ML Training - Replace XGBoost with LightGBM
LightGBM handles non-consecutive class labels properly
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Try to import LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("\nWARNING: LightGBM not installed. Will skip LightGBM model.")
    print("To install: pip install lightgbm")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = r"C:\Users\Amalie Berg\Desktop\Energy Momentum ML Strategy\energy-momentum-ml-strategy"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ASSETS = [
    'WTI_Crude',
    'Brent_Crude',
    'Natural_Gas',
    'Heating_Oil',
    'Gasoline'
]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_merge_data():
    """Load ml_features.csv and merge with regime_labels.csv"""
    print("\n" + "="*70)
    print("LOADING AND MERGING DATA")
    print("="*70)
    
    features_path = os.path.join(DATA_DIR, 'ml_features.csv')
    print(f"\nLoading features: {features_path}")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    df_features = pd.read_csv(features_path)
    df_features['Date'] = pd.to_datetime(df_features['Date'])
    print(f"Loaded {len(df_features)} rows, {len(df_features.columns)} columns")
    print(f"Date range: {df_features['Date'].min()} to {df_features['Date'].max()}")
    
    regime_path = os.path.join(DATA_DIR, 'regime_labels.csv')
    print(f"\nLooking for regime labels: {regime_path}")
    
    if not os.path.exists(regime_path):
        print("\nWARNING: regime_labels.csv not found!")
        print("Creating simple 3-regime target based on momentum")
        
        df_features['regime'] = pd.cut(
            df_features['WTI_Crude_momentum_21d'], 
            bins=3, 
            labels=[0, 1, 2]
        ).astype(int)
        print("\nCreated simple 3-regime target")
        
        return df_features
    
    df_regimes = pd.read_csv(regime_path)
    df_regimes['Date'] = pd.to_datetime(df_regimes['Date'])
    print(f"Loaded {len(df_regimes)} regime labels")
    
    regime_cols = [col for col in df_regimes.columns if 'regime' in col.lower()]
    print(f"Regime columns found: {regime_cols}")
    
    df_merged = pd.merge(df_features, df_regimes, on='Date', how='left')
    print(f"\nMerged data: {len(df_merged)} rows")
    
    missing = df_merged[regime_cols].isna().sum()
    if missing.sum() > 0:
        print(f"\nWarning: Missing regime values:")
        print(missing[missing > 0])
        print("Dropping rows with missing regimes")
        df_merged = df_merged.dropna(subset=regime_cols)
    
    print(f"Final data: {len(df_merged)} rows with regime labels")
    
    return df_merged


def prepare_asset_data(df, asset_name):
    """Prepare features for a specific asset"""
    print(f"\n" + "-"*70)
    print(f"PREPARING DATA FOR: {asset_name}")
    print("-"*70)
    
    regime_col = None
    possible_names = [
        f'{asset_name}_regime',
        f'{asset_name.lower()}_regime',
        'regime',
    ]
    
    for col in possible_names:
        if col in df.columns:
            regime_col = col
            break
    
    if regime_col is None:
        regime_cols = [col for col in df.columns if 'regime' in col.lower()]
        if len(regime_cols) > 0:
            regime_col = regime_cols[0]
            print(f"Using generic regime column: {regime_col}")
        else:
            raise ValueError(f"No regime column found for {asset_name}")
    
    print(f"Target variable: {regime_col}")
    y = df[regime_col].copy()
    
    valid_idx = ~y.isna()
    df_valid = df[valid_idx].copy()
    y = y[valid_idx]
    
    print(f"Target distribution:")
    print(y.value_counts().sort_index().to_string())
    
    unique_classes = sorted(y.unique())
    n_classes = len(unique_classes)
    print(f"Number of unique classes: {n_classes}")
    print(f"Classes present: {unique_classes}")
    
    asset_features = [col for col in df_valid.columns if col.startswith(asset_name)]
    market_features = ['VIX_level', 'VIX_change', 'VIX_vs_20', 'yield_curve_slope', 'DXY_level', 'DXY_momentum']
    
    feature_cols = asset_features + [col for col in market_features if col in df_valid.columns]
    
    X = df_valid[feature_cols].select_dtypes(include=['number']).copy()
    
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].mean(), inplace=True)
    
    print(f"\nFeatures prepared:")
    print(f"  Asset-specific features: {len(asset_features)}")
    print(f"  Market features: {len([col for col in market_features if col in df_valid.columns])}")
    print(f"  Total features: {len(X.columns)}")
    print(f"  Sample size: {len(X)}")
    
    return X, y, n_classes


def get_models(n_classes):
    """
    Define ML models including LightGBM instead of XGBoost
    """
    models = {
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient_Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic_Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Add LightGBM if available (replaces XGBoost)
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1  # Suppress output
        )
    
    return models


def train_and_evaluate_model(X, y, model, model_name, asset, n_classes):
    """Train and evaluate a single model"""
    print(f"\n" + "-"*70)
    print(f"Training {model_name} for {asset}")
    print("-"*70)
    
    # Use LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Original classes: {sorted(y.unique())}")
    print(f"Encoded classes: {sorted(np.unique(y_encoded))}")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    scaler = StandardScaler()
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        train_classes = sorted(np.unique(y_train))
        test_classes = sorted(np.unique(y_test))
        
        print(f"  Fold {fold}: Train classes={train_classes}, Test classes={test_classes}", end=" ")
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_scores['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        cv_scores['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        cv_scores['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        print(f"Acc={cv_scores['accuracy'][-1]:.4f}, F1={cv_scores['f1'][-1]:.4f}")
    
    # Average scores
    avg_scores = {k: np.mean(v) for k, v in cv_scores.items()}
    
    print(f"\n{model_name} Average Scores:")
    for metric, score in avg_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # Train final model on all data
    print(f"\nTraining final model on full dataset...")
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y_encoded)
    
    # Save model, scaler, and label encoder
    model_filename = f"{asset}_{model_name}_model.pkl"
    scaler_filename = f"{asset}_{model_name}_scaler.pkl"
    encoder_filename = f"{asset}_{model_name}_encoder.pkl"
    
    model_path = os.path.join(MODEL_DIR, model_filename)
    scaler_path = os.path.join(MODEL_DIR, scaler_filename)
    encoder_path = os.path.join(MODEL_DIR, encoder_filename)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, encoder_path)
    
    print(f"Saved: {model_filename}")
    
    return {
        'model_name': model_name,
        'asset': asset,
        'avg_accuracy': avg_scores['accuracy'],
        'avg_precision': avg_scores['precision'],
        'avg_recall': avg_scores['recall'],
        'avg_f1': avg_scores['f1'],
        'n_features': X.shape[1],
        'n_samples': len(X),
        'n_classes': n_classes
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_all_models():
    """Train all models for all assets"""
    print("\n" + "="*70)
    print("ML MODEL TRAINING PIPELINE")
    print("="*70)
    
    if LIGHTGBM_AVAILABLE:
        print("\nUsing LightGBM instead of XGBoost (better handling of class labels)")
        print("Models: Random Forest, Gradient Boosting, LightGBM, Logistic Regression")
    else:
        print("\nLightGBM not available. Using 3 models only.")
        print("To install LightGBM: pip install lightgbm")
    
    df = load_and_merge_data()
    
    all_results = []
    
    # Train for each asset
    for asset in ASSETS:
        print(f"\n\n" + "="*70)
        print(f"ASSET: {asset}")
        print("="*70)
        
        try:
            # Prepare data for this asset
            X, y, n_classes = prepare_asset_data(df, asset)
            
            # Get models
            models = get_models(n_classes)
            
            # Train each model
            for model_name, model in models.items():
                try:
                    result = train_and_evaluate_model(X, y, model, model_name, asset, n_classes)
                    all_results.append(result)
                except Exception as e:
                    print(f"\nError training {model_name} for {asset}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"\nError processing {asset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
        results_df.to_csv(results_path, index=False)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE - RESULTS SUMMARY")
        print("="*70)
        print(results_df.to_string(index=False))
        print(f"\nResults saved to: {results_path}")
        
        # Best model for each asset
        print("\n" + "="*70)
        print("BEST MODELS BY ASSET")
        print("="*70)
        
        for asset in ASSETS:
            asset_results = results_df[results_df['asset'] == asset]
            if len(asset_results) > 0:
                best = asset_results.loc[asset_results['avg_f1'].idxmax()]
                print(f"\n{asset}:")
                print(f"  Best Model: {best['model_name']}")
                print(f"  F1 Score: {best['avg_f1']:.4f}")
                print(f"  Accuracy: {best['avg_accuracy']:.4f}")
                print(f"  Features: {best['n_features']}")
    else:
        print("\nNo models were successfully trained")
    
    return all_results


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting ML Model Training...")
    print("="*70)
    
    try:
        results = train_all_models()
        print("\n" + "="*70)
        print("Training completed successfully!")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR OCCURRED")
        print("="*70)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()