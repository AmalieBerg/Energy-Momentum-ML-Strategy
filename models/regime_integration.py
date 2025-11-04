"""
Regime Detection Integration Script
====================================

This script integrates regime detection with your existing GARCH volatility
forecasting pipeline to prepare data for ML model training.

Workflow:
1. Load energy futures data from data_pipeline
2. Run GARCH models to get volatility forecasts
3. Detect market regimes using GARCH volatility
4. Add regime labels as features
5. Save processed data with regime information

Author: Amalie Berg
Project: Energy Momentum ML Strategy - Week 2
"""

import sys
import os
sys.path.append('/home/claude')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules (adjust paths as needed)
from regime_detection import MarketRegimeDetector, detect_regimes_for_all_assets

# Note: You'll need to adjust these imports to match your actual file structure
# from data.data_pipeline import download_energy_data, process_data
# from models.garch_model import fit_garch_model


class RegimeIntegration:
    """
    Integrates regime detection with GARCH volatility forecasting.
    """
    
    def __init__(self, config=None):
        """
        Initialize the integration pipeline.
        
        Parameters:
            config (dict): Configuration parameters
        """
        self.config = config or self._default_config()
        self.data = {}
        self.returns = {}
        self.volatility = {}
        self.regimes = {}
        self.regime_models = {}
        
    def _default_config(self):
        """Default configuration parameters."""
        return {
            'assets': ['CL=F', 'BZ=F', 'NG=F', 'HO=F', 'RB=F'],  # Your 5 energy futures
            'asset_names': ['WTI_Crude', 'Brent_Crude', 'Natural_Gas', 'Heating_Oil', 'Gasoline'],
            'start_date': '2010-01-01',
            'end_date': '2024-12-31',
            'n_regimes': 3,
            'regime_method': 'gmm',
            'output_dir': '/home/claude/regime_results'
        }
    
    def load_data(self, data_dict=None, returns_dict=None):
        """
        Load data from existing pipeline or provide new data.
        
        Parameters:
            data_dict (dict): Dictionary with asset names as keys and price data as values
            returns_dict (dict): Dictionary with asset names as keys and returns as values
        """
        if data_dict is not None:
            self.data = data_dict
        
        if returns_dict is not None:
            self.returns = returns_dict
        else:
            # Calculate returns from price data if not provided
            self.returns = {}
            for asset_name, prices in self.data.items():
                self.returns[asset_name] = np.log(prices / prices.shift(1)).dropna()
        
        print(f"✓ Loaded data for {len(self.data)} assets")
        
    def load_garch_volatility(self, volatility_dict):
        """
        Load GARCH volatility forecasts from existing GARCH models.
        
        Parameters:
            volatility_dict (dict): Dictionary with asset names as keys and volatility forecasts as values
        """
        self.volatility = volatility_dict
        print(f"✓ Loaded GARCH volatility for {len(volatility_dict)} assets")
    
    def detect_regimes(self):
        """
        Detect market regimes for all assets using GARCH volatility.
        """
        print("\n" + "="*60)
        print("DETECTING MARKET REGIMES")
        print("="*60 + "\n")
        
        # Detect regimes for all assets
        self.regimes, self.regime_models = detect_regimes_for_all_assets(
            self.returns,
            self.volatility if self.volatility else None,
            n_regimes=self.config['n_regimes'],
            method=self.config['regime_method']
        )
        
        return self.regimes
    
    def create_regime_features(self):
        """
        Create regime-based features for ML models.
        
        Returns:
            dict: Dictionary with asset names as keys and feature DataFrames as values
        """
        regime_features = {}
        
        for asset_name in self.returns.keys():
            # Get returns, volatility, and regimes
            returns = self.returns[asset_name]
            volatility = self.volatility.get(asset_name)
            regimes = self.regimes[asset_name]
            
            # Create DataFrame
            df = pd.DataFrame({
                'returns': returns,
                'regime': regimes
            })
            
            if volatility is not None:
                df['volatility'] = volatility
            
            # One-hot encode regimes
            regime_dummies = pd.get_dummies(df['regime'], prefix='regime')
            df = pd.concat([df, regime_dummies], axis=1)
            
            # Add regime statistics
            df['regime_label'] = df['regime'].map(
                self.regime_models[asset_name].regime_labels
            )
            
            regime_features[asset_name] = df
        
        print(f"✓ Created regime features for {len(regime_features)} assets")
        return regime_features
    
    def plot_all_regimes(self, save=True):
        """
        Create regime visualization for all assets.
        
        Parameters:
            save (bool): Whether to save plots
        """
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nGenerating regime visualizations...")
        
        for asset_name in self.returns.keys():
            returns = self.returns[asset_name]
            regimes = self.regimes[asset_name]
            dates = returns.index
            
            # Create plots
            fig1 = self.regime_models[asset_name].plot_regimes(
                dates, returns, regimes, asset_name=asset_name
            )
            fig2 = self.regime_models[asset_name].plot_regime_characteristics()
            
            if save:
                fig1.savefig(
                    f"{output_dir}/{asset_name}_regime_timeline.png",
                    dpi=300, bbox_inches='tight'
                )
                fig2.savefig(
                    f"{output_dir}/{asset_name}_regime_characteristics.png",
                    dpi=300, bbox_inches='tight'
                )
                plt.close(fig1)
                plt.close(fig2)
                print(f"  ✓ Saved plots for {asset_name}")
        
        print(f"\n✓ All plots saved to {output_dir}/")
    
    def export_regime_data(self):
        """
        Export regime data for use in ML models.
        
        Returns:
            pd.DataFrame: Combined regime data for all assets
        """
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Create regime features
        regime_features = self.create_regime_features()
        
        # Export individual asset data
        for asset_name, df in regime_features.items():
            output_file = f"{output_dir}/{asset_name}_with_regimes.csv"
            df.to_csv(output_file)
            print(f"  ✓ Exported {asset_name} data ({len(df)} rows)")
        
        # Create combined dataset
        combined_data = []
        for asset_name, df in regime_features.items():
            df_copy = df.copy()
            df_copy['asset'] = asset_name
            combined_data.append(df_copy)
        
        combined_df = pd.concat(combined_data, axis=0)
        combined_file = f"{output_dir}/all_assets_with_regimes.csv"
        combined_df.to_csv(combined_file)
        
        print(f"\n✓ Exported combined data: {combined_file}")
        print(f"  Total observations: {len(combined_df)}")
        print(f"  Assets: {len(regime_features)}")
        
        return combined_df
    
    def generate_regime_report(self):
        """
        Generate a summary report of regime detection results.
        """
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        report = []
        report.append("="*80)
        report.append("REGIME DETECTION REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nConfiguration:")
        report.append(f"  - Number of regimes: {self.config['n_regimes']}")
        report.append(f"  - Detection method: {self.config['regime_method'].upper()}")
        report.append(f"  - Assets analyzed: {len(self.returns)}")
        report.append("\n" + "-"*80 + "\n")
        
        for asset_name in self.returns.keys():
            report.append(f"\n{asset_name}")
            report.append("-" * len(asset_name))
            
            # Get regime statistics
            stats = self.regime_models[asset_name].get_regime_statistics()
            report.append(stats.to_string())
            report.append("")
        
        report_text = "\n".join(report)
        
        # Print report
        print("\n" + report_text)
        
        # Save report
        report_file = f"{output_dir}/regime_detection_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to: {report_file}")
        
        return report_text


def run_full_regime_pipeline(data_dict, returns_dict, volatility_dict, config=None):
    """
    Run the complete regime detection pipeline.
    
    Parameters:
        data_dict (dict): Price data
        returns_dict (dict): Returns data
        volatility_dict (dict): GARCH volatility forecasts
        config (dict): Configuration parameters
        
    Returns:
        RegimeIntegration: Fitted pipeline object
    """
    print("\n" + "="*80)
    print("ENERGY MOMENTUM ML STRATEGY - REGIME DETECTION PIPELINE")
    print("="*80 + "\n")
    
    # Initialize pipeline
    pipeline = RegimeIntegration(config=config)
    
    # Load data
    print("Step 1: Loading data...")
    pipeline.load_data(data_dict, returns_dict)
    pipeline.load_garch_volatility(volatility_dict)
    
    # Detect regimes
    print("\nStep 2: Detecting regimes...")
    pipeline.detect_regimes()
    
    # Generate visualizations
    print("\nStep 3: Creating visualizations...")
    pipeline.plot_all_regimes(save=True)
    
    # Export data
    print("\nStep 4: Exporting data...")
    combined_data = pipeline.export_regime_data()
    
    # Generate report
    print("\nStep 5: Generating report...")
    pipeline.generate_regime_report()
    
    print("\n" + "="*80)
    print("✓ REGIME DETECTION PIPELINE COMPLETE")
    print("="*80 + "\n")
    
    return pipeline


if __name__ == "__main__":
    """
    Example usage with synthetic data.
    """
    print("\nREGIME INTEGRATION - EXAMPLE RUN\n")
    
    # Generate synthetic data for 2 assets
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Asset 1: WTI Crude
    wti_returns = np.random.normal(0.001, 0.02, n_samples)
    wti_volatility = np.abs(wti_returns) * 1.5 + np.random.normal(0, 0.005, n_samples)
    wti_prices = 100 * np.exp(np.cumsum(wti_returns))
    
    # Asset 2: Natural Gas
    ng_returns = np.random.normal(0.0005, 0.03, n_samples)
    ng_volatility = np.abs(ng_returns) * 1.3 + np.random.normal(0, 0.008, n_samples)
    ng_prices = 3 * np.exp(np.cumsum(ng_returns))
    
    # Create DataFrames
    data_dict = {
        'WTI_Crude': pd.Series(wti_prices, index=dates),
        'Natural_Gas': pd.Series(ng_prices, index=dates)
    }
    
    returns_dict = {
        'WTI_Crude': pd.Series(wti_returns, index=dates),
        'Natural_Gas': pd.Series(ng_returns, index=dates)
    }
    
    volatility_dict = {
        'WTI_Crude': pd.Series(wti_volatility, index=dates),
        'Natural_Gas': pd.Series(ng_volatility, index=dates)
    }
    
    # Run pipeline
    pipeline = run_full_regime_pipeline(
        data_dict,
        returns_dict,
        volatility_dict,
        config={
            'n_regimes': 3,
            'regime_method': 'gmm',
            'output_dir': '/home/claude/regime_example_results'
        }
    )
    
    print("\n✓ Example run complete! Check /home/claude/regime_example_results/")