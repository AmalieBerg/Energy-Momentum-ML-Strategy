"""
GARCH Model for Energy Volatility Forecasting
Based on Heston-Nandi GARCH expertise
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import *
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnergyGARCHModel:
    """
    GARCH model for energy volatility forecasting
    Based on Heston-Nandi thesis expertise
    """
    
    def __init__(self, p=1, q=1):
        self.p = p  # GARCH lags
        self.q = q  # ARCH lags
        self.model = None
        self.fitted_model = None
        self.results = {}
        
    def fit_garch(self, returns: pd.Series, verbose=True) -> Dict:
        """
        Fit GARCH(p,q) model to returns series
        
        Args:
            returns: Return series
            verbose: Whether to print results
            
        Returns:
            Dictionary of fitted parameters
        """
        # Convert to percentage returns for numerical stability
        returns_pct = returns * 100
        
        # Remove any infinite or NaN values
        returns_clean = returns_pct.replace([np.inf, -np.inf], np.nan).dropna()
        
        logger.info(f"🔍 Fitting GARCH({self.p},{self.q}) to {len(returns_clean)} observations")
        logger.info(f"   Return stats: Mean={returns_clean.mean():.4f}%, Std={returns_clean.std():.4f}%")
        
        # Fit GARCH model
        self.model = arch_model(
            returns_clean, 
            vol='GARCH', 
            p=self.p, 
            q=self.q,
            dist='normal',
            mean='constant'
        )
        
        self.fitted_model = self.model.fit(disp='off' if not verbose else 'final')
        
        # Extract parameters
        params = self.fitted_model.params
        
        self.results = {
            'omega': params['omega'],
            'alpha': params.get('alpha[1]', 0),
            'beta': params.get('beta[1]', 0),
            'mu': params.get('mu', 0),
            'persistence': params.get('alpha[1]', 0) + params.get('beta[1]', 0),
            'log_likelihood': self.fitted_model.loglikelihood,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'n_obs': len(returns_clean)
        }
        
        if verbose:
            logger.info(f"🔥 GARCH({self.p},{self.q}) Results:")
            logger.info(f"   μ (mu):      {self.results['mu']:.6f}")
            logger.info(f"   ω (omega):   {self.results['omega']:.6f}")
            logger.info(f"   α (alpha):   {self.results['alpha']:.6f}")
            logger.info(f"   β (beta):    {self.results['beta']:.6f}")
            logger.info(f"   Persistence: {self.results['persistence']:.6f}")
            logger.info(f"   Log-likelihood: {self.results['log_likelihood']:.2f}")
            logger.info(f"   AIC: {self.results['aic']:.2f}")
            
        return self.results
    
    def forecast_volatility(self, horizon=1) -> pd.Series:
        """
        Generate volatility forecasts
        
        Args:
            horizon: Number of periods ahead to forecast
            
        Returns:
            Annualized volatility forecast
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first!")
            
        # Generate forecasts
        forecast = self.fitted_model.forecast(horizon=horizon)
        
        # Convert from percentage^2 to decimal^2, then annualize
        variance_forecasts = forecast.variance.iloc[-1] / 10000
        vol_forecasts_daily = np.sqrt(variance_forecasts)
        vol_forecasts_annual = vol_forecasts_daily * np.sqrt(252)
        
        return vol_forecasts_annual
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Extract fitted conditional volatility (sigma_t)
        
        Returns:
            Daily conditional volatility series
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first!")
        
        # Get conditional volatility from fitted model (daily, in decimal)
        cond_vol_daily = np.sqrt(self.fitted_model.conditional_volatility / 10000)
        cond_vol_daily.index = self.fitted_model.resid.index
        
        return cond_vol_daily
    
    def diagnostic_plots(self, returns: pd.Series, asset_name: str):
        """
        Create diagnostic plots like in thesis
        
        Args:
            returns: Return series
            asset_name: Name of asset for plot title
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first!")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'GARCH Model Diagnostics: {asset_name}', fontsize=16)
        
        # 1. Returns vs Fitted Volatility
        cond_vol = self.get_conditional_volatility()
        aligned_returns = returns.loc[cond_vol.index]
        
        axes[0, 0].plot(aligned_returns.index, aligned_returns.values, 
                       alpha=0.7, color='blue', linewidth=0.5)
        axes[0, 0].fill_between(aligned_returns.index, -2*cond_vol, 2*cond_vol, 
                                alpha=0.3, color='red')
        axes[0, 0].set_title('Returns vs ±2σ Volatility Bands')
        axes[0, 0].set_ylabel('Returns')
        
        # 2. Conditional Volatility Over Time
        annual_vol = cond_vol * np.sqrt(252)
        axes[0, 1].plot(annual_vol.index, annual_vol.values, color='red', linewidth=1)
        axes[0, 1].set_title('Conditional Volatility (Annualized)')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Standardized Residuals
        std_resid = self.fitted_model.std_resid
        axes[1, 0].plot(std_resid.index, std_resid.values, 
                       alpha=0.7, color='green', linewidth=0.5)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_title('Standardized Residuals')
        axes[1, 0].set_ylabel('Std Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot of Standardized Residuals
        from scipy import stats
        stats.probplot(std_resid.dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot: Standardized Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = RESULTS_PATH / f'garch_diagnostics_{asset_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"   Plot saved to {plot_path}")
        plt.close()
        
        return fig


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("🛢️  Testing GARCH model with Energy Futures")
    logger.info("="*70 + "\n")
    
    # Load the data
    log_returns = pd.read_csv(
        DATA_PATH / 'energy_log_returns.csv',
        index_col=0,
        parse_dates=True
    )
    
    logger.info(f"📊 Loaded returns data: {log_returns.shape}")
    logger.info(f"   Assets: {list(log_returns.columns)}")
    logger.info(f"   Date range: {log_returns.index[0]} to {log_returns.index[-1]}")
    
    # Test with WTI crude
    wti_returns = log_returns['WTI_Crude'].dropna()
    
    logger.info(f"\n🛢️  Fitting GARCH to WTI Crude Oil...")
    logger.info(f"   Sample: {len(wti_returns)} observations")
    logger.info(f"   Period: {wti_returns.index[0]} to {wti_returns.index[-1]}")
    
    # Fit GARCH model
    garch = EnergyGARCHModel(p=1, q=1)
    results = garch.fit_garch(wti_returns, verbose=True)
    
    # Generate forecasts
    vol_forecast = garch.forecast_volatility(horizon=5)
    logger.info(f"\n📊 Next 5-day volatility forecasts (annualized):")
    if isinstance(vol_forecast, pd.Series):
        for i, vol in enumerate(vol_forecast):
            logger.info(f"   Day {i+1}: {vol:.1%}")
    else:
        logger.info(f"   Day 1: {vol_forecast:.1%}")
    
    # Create diagnostic plots
    logger.info(f"\n📈 Creating diagnostic plots...")
    garch.diagnostic_plots(wti_returns, 'WTI_Crude')
    
    logger.info(f"\n✅ GARCH model test completed!")
    
    # Validation check
    logger.info(f"\n🔍 Validation Check:")
    logger.info(f"   Sample return std (daily): {wti_returns.std():.4f} ({wti_returns.std()*np.sqrt(252):.1%} annual)")
    
    cond_vol = garch.get_conditional_volatility()
    logger.info(f"   GARCH avg vol (daily): {cond_vol.mean():.4f} ({cond_vol.mean()*np.sqrt(252):.1%} annual)")
    logger.info(f"   GARCH current vol: {cond_vol.iloc[-1]*np.sqrt(252):.1%} annual")
    
    if cond_vol.mean() * np.sqrt(252) > 0.6:
        logger.warning("   ❌ WARNING: Volatility seems too high!")
    else:
        logger.info("   ✅ Volatility levels look reasonable")
