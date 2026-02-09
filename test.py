# %% [markdown]
# # Single House Optimal Elevation Analysis
# 
# This notebook determines the optimal elevation for a single house under flooding scenarios,
# replicating the R code methodology from Zarekarizi et al. with full uncertainty quantification.
# 
# **Outputs:**
# - Total cost curves (construction + expected damages)
# - Benefit-to-cost ratio curves
# - Reliability (safety) curves
# - Satisficing metrics
# - Comparison table: No elevation vs BFE vs BFE+1 vs Optimal (ignoring uncertainty) vs Optimal (with uncertainty)

# %% [markdown]
# ## 1. Configuration and Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# For MCMC - GEV parameter estimation
import emcee
from scipy.stats import genextreme

# Set random seed for reproducibility
np.random.seed(42)

# %% [markdown]
# ## 2. House Characteristics Input
# 
# Define the characteristics of the single house to analyze.

# %%
# =============================================================================
# HOUSE CHARACTERISTICS - MODIFY THESE VALUES FOR YOUR ANALYSIS
# =============================================================================

house_config = {
    'sqft': 1500,              # House size in square feet
    'Struc_Value': 300000,     # Structure value in USD
    'del': -4,                 # Elevation difference: house elevation - BFE (negative if below BFE)
    'life_span': 30,           # Expected house lifetime in years
    'disc_rate': 0.04,         # Best-guess discount rate (used when uncertainty is off)
}

# =============================================================================
# UNCERTAINTY TOGGLE - Set to True for full uncertainty, False for best-guess only
# =============================================================================
INCLUDE_UNCERTAINTY = True

# =============================================================================
# NUMBER OF STATES OF THE WORLD (Monte Carlo samples)
# =============================================================================
N_SOW = 10000

# =============================================================================
# ELEVATION RANGE TO EVALUATE
# =============================================================================
# Elevations from 0-14 feet, with 0-3 feet marked as infeasible
ELEVATION_RANGE = np.concatenate([[0], np.linspace(3, 14, 20)])

# =============================================================================
# SATISFICING THRESHOLDS
# =============================================================================
THRESHOLD_TOTAL_COST_RATIO = 0.75  # Total cost as fraction of house value
THRESHOLD_BCR = 1.0                 # Benefit-to-cost ratio threshold
THRESHOLD_RELIABILITY = 0.5         # Reliability (safety) threshold

# Print configuration
print("=" * 60)
print("HOUSE CONFIGURATION")
print("=" * 60)
for key, val in house_config.items():
    print(f"  {key}: {val}")
print(f"\nUncertainty Analysis: {'ENABLED' if INCLUDE_UNCERTAINTY else 'DISABLED'}")
print(f"Number of SOWs: {N_SOW}")
print("=" * 60)

# %% [markdown]
# ## 3. Hazard Data Input
# 
# Three options for specifying flood hazard:
# - **(a)** GEV parameters directly
# - **(b)** Annual maximum water level data (fits GEV via MCMC)
# - **(c)** Return period depths at the house location

# %%
# =============================================================================
# HAZARD DATA INPUT - CHOOSE ONE OPTION
# =============================================================================

HAZARD_OPTION = 'a'  # Options: 'a', 'b', 'c'

# -----------------------------------------------------------------------------
# OPTION (a): Provide GEV parameters directly
# -----------------------------------------------------------------------------
# Provide as dictionary with location (mu), scale (sigma), shape (xi)
gev_params_direct = {
    'mu': 20.45,      # Location parameter
    'sigma': 3.31,    # Scale parameter
    'xi': -0.056      # Shape parameter (negative = Weibull-type, bounded upper tail)
}

# -----------------------------------------------------------------------------
# OPTION (b): Provide annual maximum water level data
# The code will fit GEV via MCMC and derive BFE automatically
# -----------------------------------------------------------------------------
# Provide as list or numpy array of annual maximum water levels
annual_max_water_levels = None  # Example: np.array([25.3, 22.1, 28.4, ...])

# -----------------------------------------------------------------------------
# OPTION (c): Provide return period depths at house location
# Dictionary with return period (years) as key and depth (feet) as value
# -----------------------------------------------------------------------------
return_period_depths = {
    10: 2.5,    # 10-year return period depth
    50: 4.1,    # 50-year return period depth
    100: 5.2,   # 100-year return period depth (this is used as BFE)
    500: 7.8    # 500-year return period depth
}

# %% [markdown]
# ## 4. Depth-Damage Function Selection

# %%
# =============================================================================
# DEPTH-DAMAGE FUNCTION SELECTION
# =============================================================================

DDF_OPTION = 'hazus'  # Options: 'hazus', 'eu', 'naccs', 'deep' (samples between them)

# HAZUS Depth-Damage Function (FEMA)
# Depths in feet relative to first floor, damage as % of structure value
HAZUS_DDF = {
    'depths': np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
    'damage_pct': np.array([0, 0, 4, 8, 12, 15, 20, 23, 28, 33, 37, 43, 48, 51, 53,
                            55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81])
}

# EU/Global Flood Depth-Damage Function (Huizinga et al., 2017)
# Converted to feet from meters
EU_DDF = {
    'depths': np.array([0, 1.64, 3.28, 4.92, 6.56, 9.84, 13.12, 16.40]),
    'damage_pct': np.array([20, 44, 58, 68, 78, 85, 92, 96])
}

# NACCS Depth-Damage Function (simplified residential)
# Similar structure to HAZUS but different values
NACCS_DDF = {
    'depths': np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24]),
    'damage_pct': np.array([0, 5, 10, 16, 22, 27, 32, 37, 42, 46, 50, 53, 56, 61, 66, 70, 74, 77, 80, 83])
}

# %% [markdown]
# ## 5. Discount Rate Data

# %%
# =============================================================================
# DISCOUNT RATE DATA
# =============================================================================
# Load historical discount rate data from CSV
# Expected format: columns 'Year' and 'DR' (discount rate as percentage)

DISCOUNT_RATE_FILE = 'discount.csv'  # Path to your discount rate CSV file

# Try to load discount rate data, use default if file not found
try:
    discount_df = pd.read_csv(DISCOUNT_RATE_FILE)
    discount_df.columns = ['Year', 'DR']  # Ensure consistent column names
    print(f"Loaded discount rate data: {len(discount_df)} years ({discount_df['Year'].min()}-{discount_df['Year'].max()})")
except FileNotFoundError:
    print(f"Warning: {DISCOUNT_RATE_FILE} not found. Using default discount rate data.")
    # Default discount rate data (simplified version)
    years = np.arange(1919, 2019)
    # Simulated historical rates with declining trend
    dr_values = 4.5 + 2 * np.sin(np.linspace(0, 4*np.pi, len(years))) + np.random.normal(0, 0.5, len(years))
    dr_values = np.clip(dr_values, 1, 10)
    discount_df = pd.DataFrame({'Year': years, 'DR': dr_values})

# %% [markdown]
# ## 6. Core Functions

# %% [markdown]
# ### 6.1 GEV Distribution Functions

# %%
def gev_cdf(x, mu, sigma, xi):
    """
    Cumulative distribution function for GEV distribution.
    
    Parameters:
    -----------
    x : float or array
        Value(s) at which to evaluate CDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    xi : float
        Shape parameter
    
    Returns:
    --------
    Probability P(X <= x)
    """
    # Use scipy's genextreme (note: scipy uses -xi convention)
    return genextreme.cdf(x, -xi, loc=mu, scale=sigma)


def gev_quantile(p, mu, sigma, xi):
    """
    Quantile function (inverse CDF) for GEV distribution.
    
    Parameters:
    -----------
    p : float or array
        Probability (between 0 and 1)
    mu : float
        Location parameter
    sigma : float
        Scale parameter
    xi : float
        Shape parameter
    
    Returns:
    --------
    Quantile value (return level)
    """
    return genextreme.ppf(p, -xi, loc=mu, scale=sigma)


def return_level(return_period, mu, sigma, xi):
    """
    Calculate return level for a given return period.
    
    Parameters:
    -----------
    return_period : float
        Return period in years
    mu, sigma, xi : float
        GEV parameters
    
    Returns:
    --------
    Return level (flood height)
    """
    p = 1 - 1/return_period
    return gev_quantile(p, mu, sigma, xi)

# %% [markdown]
# ### 6.2 MCMC for GEV Parameter Estimation

# %%
def gev_log_likelihood(params, data):
    """Log-likelihood for GEV distribution."""
    mu, sigma, xi = params
    if sigma <= 0:
        return -np.inf
    
    try:
        ll = np.sum(genextreme.logpdf(data, -xi, loc=mu, scale=sigma))
        if np.isnan(ll) or np.isinf(ll):
            return -np.inf
        return ll
    except:
        return -np.inf


def gev_log_prior(params):
    """Log-prior for GEV parameters (weakly informative)."""
    mu, sigma, xi = params
    
    # Priors based on R code: normal with large variance
    # mu ~ N(0, 1000), sigma ~ N(0, 100), xi ~ N(0, 1)
    if sigma <= 0:
        return -np.inf
    
    log_prior = 0
    log_prior += stats.norm.logpdf(mu, 0, np.sqrt(1000))
    log_prior += stats.norm.logpdf(sigma, 0, np.sqrt(100))
    log_prior += stats.norm.logpdf(xi, 0, 1)
    
    return log_prior


def gev_log_posterior(params, data):
    """Log-posterior for GEV parameters."""
    lp = gev_log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gev_log_likelihood(params, data)


def fit_gev_mcmc(data, n_walkers=32, n_steps=10000, n_burn=5000):
    """
    Fit GEV distribution using MCMC (emcee).
    
    Parameters:
    -----------
    data : array
        Annual maximum water level data
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_burn : int
        Number of burn-in steps to discard
    
    Returns:
    --------
    dict with 'mu_chain', 'sigma_chain', 'xi_chain', 'map_params'
    """
    print("Fitting GEV distribution via MCMC...")
    
    # Get MLE estimates as starting point
    xi_mle, mu_mle, sigma_mle = genextreme.fit(data)
    xi_mle = -xi_mle  # Convert scipy convention
    
    # Initialize walkers around MLE
    ndim = 3
    pos = np.array([mu_mle, sigma_mle, xi_mle]) + 0.1 * np.random.randn(n_walkers, ndim)
    pos[:, 1] = np.abs(pos[:, 1])  # Ensure sigma > 0
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, gev_log_posterior, args=(data,))
    
    # Burn-in
    print(f"  Running burn-in ({n_burn} steps)...")
    state = sampler.run_mcmc(pos, n_burn, progress=True)
    sampler.reset()
    
    # Production run
    print(f"  Running production ({n_steps} steps)...")
    sampler.run_mcmc(state, n_steps, progress=True)
    
    # Extract chains (flatten across walkers)
    samples = sampler.get_chain(flat=True)
    mu_chain = samples[:, 0]
    sigma_chain = samples[:, 1]
    xi_chain = samples[:, 2]
    
    # Find MAP estimate
    log_probs = sampler.get_log_prob(flat=True)
    map_idx = np.argmax(log_probs)
    map_params = {
        'mu': mu_chain[map_idx],
        'sigma': sigma_chain[map_idx],
        'xi': xi_chain[map_idx]
    }
    
    print(f"  MAP estimates: mu={map_params['mu']:.3f}, sigma={map_params['sigma']:.3f}, xi={map_params['xi']:.3f}")
    
    return {
        'mu_chain': mu_chain,
        'sigma_chain': sigma_chain,
        'xi_chain': xi_chain,
        'map_params': map_params
    }


def fit_gev_from_return_periods(rp_depths):
    """
    Estimate GEV parameters from return period depths.
    Uses least-squares fitting to match return levels.
    
    Parameters:
    -----------
    rp_depths : dict
        Dictionary with return period (years) as key and depth as value
    
    Returns:
    --------
    dict with GEV parameters
    """
    from scipy.optimize import minimize
    
    return_periods = np.array(list(rp_depths.keys()))
    depths = np.array(list(rp_depths.values()))
    
    def objective(params):
        mu, sigma, xi = params
        if sigma <= 0:
            return 1e10
        predicted = np.array([return_level(rp, mu, sigma, xi) for rp in return_periods])
        return np.sum((predicted - depths)**2)