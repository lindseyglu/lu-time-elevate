# -*- coding: utf-8 -*-
"""
Filename: adaptive_elev.py
Author: Lindsey Lu
Created: 2026-02-05
Version: 1.0
Description: 
"""

# import libraries
import pandas as pd
import numpy as np
from scipy.stats import genextreme
from scipy.stats import weibull_min
from scipy.stats import uniform
from scipy.stats import qmc             # for Latin Hypercube Sampling
import matplotlib
matplotlib.use('Agg')                   # Use the 'Agg' backend to avoid the Qt/DBus error while plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set print
verbose = False

# Set rng
rng = np.random.default_rng()

# Set deterministic house characteristics and discount rate
sqft = 1500
struc_value = 300000
del_elev = -4           # difference in house elev and BFE
life_span = 30
dr_i = np.arange(201)
disc_rate = np.exp(-1 * (0.04 * dr_i))
bfe = 34.7              # generated in the R code
init_elev = bfe + del_elev  # Initial house elev
if verbose: 
    print("House parameters")
    print(f"Structure value: ${struc_value}")
    print(f"Lifetime: {life_span}")
    print(f"Initial elevation: {init_elev} ft")
    print(f"Base flood elevation: {bfe}")
    print(f"Discount rate: {disc_rate}")

# Set deterministic GEV parameters (currently using GEV from Zarekarizi et al. 2020 - R code generated)
mu = 19.8718901487264
sigma = 3.16814792683425
xi = 0.00515921024408503
if verbose:
    print("GEV parameters")
    print(f"location: {mu}")
    print(f"scale: {sigma}")
    print(f"shape: {xi}")

# Set elevation height strategies to evaluate (first strategy must be 0)
delta_h_seq = np.array([0,3,4,5,6,7,8,9,10,11,12,13,14])

# Set depth-damage function
# HAZUS DDF:
depth = np.array([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])  # defined relative the FFE
damage_fac = np.array([0,0,4,8,12,15,20,23,28,33,37,43,48,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81])

## ------------------------------------------------------------------
## DEFINE GEV FUNCTION FOR FASTER COMPUTATION
## ------------------------------------------------------------------

def fast_gev_cdf(x, c, loc, scale):
    # Standard GEV formula using NumPy vectorized operations
    # Note: Using your variable 'xi' directly as the shape param
    z = (x - loc) / scale
    # To handle the 1 + xi*z > 0 constraint
    inner = 1 + c * z
    return np.exp(-np.power(np.maximum(inner, 1e-10), -1/c))

## ------------------------------------------------------------------
## OBJECTIVES FUNCTIONS
## ------------------------------------------------------------------

# Step 1: Calculate lifetime expected damages
def lifetime_expected_damages(struc_value, init_elev, delta_h, life_span, disc_rate, mu, sigma, xi, DD_Depth, DD_Damage, yr_elev):
    """
    Docstring for lifetime_expected_damages
    
    :param struc_value: structure value (shape: (nsow, life_span))
    :param init_elev: initial house elevation
    :param delta_h: height the house is being raised by
    :param life_span: house lifespan (shape: (nsow,))
    :param disc_rate: array of discount rates for each year of the house lifespan (shape: (nsow, life_span))
    :param mu: location parameter for generalized extreme value (GEV) distribution (shape: (nsow, life_span))
    :param sigma: scale parameter for GEV (shape: (nsow, life_span))
    :param xi: shape parameter for GEV (shape: (nsow, life_span))
    :param DD_Depth: depths from the depth-damage function, defined relative to FFE (shape: (num_depths,))
    :param DD_Damage: damage factor from the depth-damage function, defined out of 100 (shape: (nsow, num_depths))
    """
    nsow = len(life_span)
    curr_elev = init_elev + delta_h # elevation of house after being elevated
    
    # Damage value lost at each depth, which depends on house value
    # shape: (nsow, num_depths, life_span)
    damage_vals = (DD_Damage[:, :, np.newaxis]/100) * struc_value[:, np.newaxis, :]

    # Critical depths are depths where the damage factor changes.
    # Calculates the stage of critical depths
    crit_depths_elev = DD_Depth + curr_elev              # shape: (num_depths,)
    crit_depths_init = DD_Depth + init_elev              # shape: (num_depths,)

    # Probability that water level exceeds each critical depth in one year
    # When c < 0, Frechet-type tail
    # We reshape parameters to (nsow, 1) to broadcast against crit_depths (1, num_depths, 1)
    # Resulting crit_probs shape: (nsow, num_depths)
    crit_probs_elev = 1 - fast_gev_cdf(
        x=crit_depths_elev,              # Shape: (1, num_depths, 1)
        c=xi[:,np.newaxis],         # Shape: (nsow, 1)
        loc=mu[:,np.newaxis],       # Shape: (nsow, 1)
        scale=sigma[:,np.newaxis]   # Shape: (nsow, 1)
    )

    crit_probs_init = 1 - fast_gev_cdf(
        x=crit_depths_init,              # Shape: (1, num_depths, 1)
        c=xi[:,np.newaxis],         # Shape: (nsow, 1)
        loc=mu[:,np.newaxis],       # Shape: (nsow, 1)
        scale=sigma[:,np.newaxis]   # Shape: (nsow, 1)
    )

    # NEEDS ALTERING? Final safety catch for floating point precision issues
    crit_probs_elev = np.nan_to_num(crit_probs_elev, nan=0.0)         # shape: (nsow, num_depths, life_span)
    crit_probs_init = np.nan_to_num(crit_probs_init, nan=0.0)         # shape: (nsow, num_depths, life_span)

    # Calculate the expected annual damages (EAD) for each year
    # Note: it is different each year depending on the house value and whether the house is elevated that year or not
    prob_diffs_elev = -np.diff(crit_probs_elev, axis=1, append=0)     # shape: (nsow, num_depths)
    prob_diffs_init = -np.diff(crit_probs_init, axis=1, append=0)     # shape: (nsow, num_depths)
    ead_elev = np.sum(prob_diffs_elev[:, :, np.newaxis] * damage_vals, axis=1)          # shape: (nsow, life_span)
    ead_init = np.sum(prob_diffs_init[:, :, np.newaxis] * damage_vals, axis=1)          # shape: (nsow, life_span)

    # Combine elevated and non-elevated EAD
    ead = np.hstack((ead_init[:,0:yr_elev], ead_elev[:,yr_elev:201]))

    # Apply the year-by-year discount rate
    disc_ead = ead * disc_rate     # shape: (nsow, life_span)*(nsow, life_span)=(nsow, life_span)

    # Create a mask: True if year < house_lifetime
    years_i = np.arange(disc_rate.shape[1])
    lifespan_mask = years_i < life_span[:, np.newaxis]
    
    # Calculate expected damage
    # Sum across the life_span (axis 1) to get one value per SOW
    exp_dam = np.sum(disc_ead * lifespan_mask, axis=1)

    return exp_dam

# Step 2: Calculate construction cost
def construction_cost(delta_h, sqft, yr_elev, disc_rate):
    """
    Docstring for construction_cost
    
    :param delta_h: height the house is being raised by
    :param sqft: house square footage
    :param yr_elev: year of house elevation
    :param yr_elev: discount rate (shape: (nsow, life_span))

    :return npv_cost: the NPV cost of raising a house with inflation and discount rate (shape: (nsow,1))
    """
    # Cost of elevating according to CLARA are as the following:
    # 82.5/sqft (3 to 7)
    # 86.25/sqft (7 to 10)
    # 103.75/sqft (10 to 14)
  
    # There is a base cost for elevating any house as the following. For more information, see appendix A of CLARA model through the link above 
    base_cost= 10000 + 300 + 470 + 4300 + 2175 + 3500
    Hs=np.array([3,5,8.5,12,14])
    Rates=np.array([80.36,82.5,86.25,103.75,113.75])

    # The cost of elevating the house after the base cost depends on the size of the house
    # Linear interpolate to find the new rate per sqft
    if 3 <= delta_h <= 14:
        rate=np.interp(delta_h, Hs, Rates)
    else:     
        rate=0
  
    # total cost of elevating the house:
    raise_cost = base_cost + rate*sqft
    
    # There is no cost for not elevating the house
    if(delta_h==0): raise_cost=0
    
    # Calculate total inflation (assume a constant 3%)
    # infl is 1 if yr_elev=0
    infl = (1+0.03)**yr_elev
    # Find the disc_rate for the year of elevation
    dr = disc_rate[:, yr_elev]
    # Adjust raise_cost by inflation and discount rate
    npv_cost = raise_cost * infl * dr

    return(npv_cost)

# Step 3: Calculate reliability (probability of not being flooded at all during the lifetime of the house)
def lifetime_reliability(life_span, mu, sigma, xi, init_elev, delta_h, yr_elev):
    """
    Docstring for lifetime_reliability
    
    :param life_span: house lifetime (shape: (nsow,))
    :param mu: location parameter for GEV (shape: (nsow, life_span))
    :param sigma: scale parameter for GEV (shape: (nsow, life_span))
    :param xi: shape parameter for GEV (shape: (nsow, life_span))
    :param init_elev: house initial elevation
    :param delta_h: height the house is being raised by

    Returns:
        safety: (shape: (nsow,))
    """
    curr_elev = init_elev + delta_h

    # Calculate the probability of being flooded (nsow)
    # When c < 0, Frechet-type tail
    prob_init = fast_gev_cdf(x=init_elev, c=xi, loc=mu, scale=sigma)
    prob_elev = fast_gev_cdf(x=curr_elev, c=xi, loc=mu, scale=sigma)

    # Need to ensure that yr_elev < life_span, so define a variable (yrs_post) years post-elevation
    yrs_post = np.where(life_span-yr_elev < 0, 0, life_span-yr_elev)
    safety = (prob_init ** (yr_elev//1)) * (prob_elev ** (yrs_post//1))

    return(safety)

# Evaluate satisficing criteria (robustness)
# From Zarekarizi et al. 2020: BCR > 1, reliability > 0.5, total cost/structure value < 1
def satisficing_all(bcr, reliability, total_cost, struc_val):
    """
    Docstring for satisficing_all
    
    :param bcr: benefit-cost ratio of strategy
    :param reliability: lifetime reliability of strategy
    :param total_cost: total cost of strategy
    :param struc_val: house value
    """
    return np.array([bcr>1, reliability>0.5, total_cost/struc_val<1])


## ------------------------------------------------------------------
## UNCERTAINTY FUNCTIONS
## ------------------------------------------------------------------

# Consider 5 uncertainties:
# 1. Discount rate [deep]
#       a. random walk
#       b. mean-reverting
#       c. background linear trend (log-scale)
# 2. House lifetime
#       a. Weibull distribution (shape=2.8, scale=73.5)
# 3. Depth-damage function [deep]
#       a. European Commission DDF (uniform 30% unc)
#       b. HAZUS (uniform 30% unc)
# 4. Flooding frequency
#       a. GEV distribution
# 5. House value
#       a. simple linear function

# 1. Discount rate
def discount_rate_unc(obs_discount, nsow, dr_func="deep", life_span=200):
    """
    Docstring for discount_unc
    returns a discount rate for each year of house lifetime and each SOW (nsow, life_span+1)
    
    :param obs_discount: historical observed discount rate
    :param nsow: number state of the worlds
    :param dr_func: type of discount function
    :param life_span: house lifetime (set as 200 as max)
    """
    # 1. Prepare data (log scale as per Newell & Pizer methodology)
    # Assumes obs_discount is a 2D array-like with interest rates in the 2nd column
    d = np.log(np.array(obs_discount)[:, 1])
    n_cols = life_span + 1
    
    # Internal function runs model type
    def run_ar3(model_type, pars):
        # Allocate matrix (nsow rows, life_span + 3 years for AR(3) lag)
        eps = np.full((nsow, life_span + 3), np.nan)
        
        # Initial Values logic
        last_3 = d[-3:]
        # Drift
        if model_type == "drift":
            offset = len(d) - 1
            # Trend calculation
            time_range = np.arange(offset - 2, life_span + offset + 1)
            tr = pars['int'] + pars['slope'] * time_range
            eps[:, :3] = last_3 - tr[:3]
        # Mean reverting
        elif model_type == "mrv": 
            eps[:, :3] = last_3 - np.log(pars['eta'])
        # Random walk
        elif model_type == "rw":
            first_valid = d[~np.isnan(d)][0]
            eps[:, :3] = last_3 - first_valid
            
        # Stochastic Simulation (Looping over time is required for auto-regressive processes)
        sigma = np.sqrt(pars['sigma_sq'])
        for i in range(3, life_span + 3):
            # Vectorized across all nsow states of the world
            innovation = rng.normal(0, sigma, nsow)
            eps[:, i] = (pars['rho1'] * eps[:, i-1] + 
                         pars['rho2'] * eps[:, i-2] + 
                         pars['rho3'] * eps[:, i-3] + innovation)
        
        # Convert innovations back to discount rates
        if model_type == "drift":
            rates = np.exp(eps[:, 3:] + tr[3:])
        elif model_type == "mrv":
            rates = np.exp(np.log(pars['eta']) + eps[:, 3:])
        elif model_type == "rw":
            rates = np.exp(first_valid + eps[:, 3:])
            
        # Calculate cumulative discount factors: exp(-sum(r_i / 100))
        # np.cumsum along axis 1 (time) calculates the running sum for each SOW
        dfactors = np.exp(-1 * np.cumsum(rates / 100, axis=1))
        # Prepend 1.0 for year 0
        return np.hstack([np.ones((nsow, 1)), dfactors])

    # 2. Parameters (Best-estimates from R script)
    params_rw = {'rho1': 1.7429, 'rho2': -1.0455, 'rho3': 0.3010, 'sigma_sq': 0.0034}
    params_mrv = {'eta': 3.405, 'rho1': 1.7371, 'rho2': -1.0270, 'rho3': 0.2806, 'sigma_sq': 0.0034}
    params_drift = {'int': 1.9289, 'slope': -0.0058, 'rho1': 1.6965, 'rho2': -0.9755, 'rho3': 0.2388, 'sigma_sq': 0.0033}

    # 3. Model Execution & Selection
    if dr_func == "cert-4%":
        t = np.arange(n_cols)
        return np.tile(np.exp(-0.04 * t), (nsow, 1))

    m_rw = run_ar3("rw", params_rw)
    if dr_func == "rw": return m_rw
    
    m_mrv = run_ar3("mrv", params_mrv)
    if dr_func == "mrv": return m_mrv
    
    m_drift = run_ar3("drift", params_drift)
    if dr_func == "drift": return m_drift
    
    if dr_func == "deep":
        # Vectorized choice using a mask
        # Create an array of 0, 1, or 2 foreach SOW (mask)
        selector = rng.choice([0, 1, 2], size=nsow)   # 0: rw, 1: mrv, 2: drift
        # Allocate matrix
        m_deep = np.zeros((nsow, n_cols))
        # For each SOW, fill the row where the mask is true with the corresponding model
        m_deep[selector == 0] = m_rw[selector == 0]
        m_deep[selector == 1] = m_mrv[selector == 1]
        m_deep[selector == 2] = m_drift[selector == 2]
        return m_deep
    
    raise ValueError('Options are "rw", "mrv", "drift", "deep", or "cert-4%".')

# 2. House lifetime
def lifetime_unc(nsow, lifetime_func="weibull"):
    # Randomly sample from weibull dist
    if lifetime_func == "weibull": 
        lt_unc = weibull_min.rvs(c=2.8, scale=73.5, size=nsow, random_state=rng)
    else: raise ValueError(f"House lifetime function {lifetime_func} unknown")

    return lt_unc

# 3. Depth-damage function
def depth_damage_unc(nsow, ddf_type="deep"):
    # Define European Commission's depth-damage function
    depth1 = np.array([0, 1.64, 3.28, 4.92, 6.56, 9.84, 13.12, 16.40])
    damage_fac1 = np.array([0.20, 0.44, 0.58, 0.68, 0.78, 0.85, 0.92, 0.96])*100
    # Define HAZUS depth-damage function
    depth2 = np.array([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    damage_fac2 = np.array([0,0,4,8,12,15,20,23,28,33,37,43,48,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81])

    # Create an array of numbers from the minimum depth of either function to the maximum depth of either function
    depths = np.linspace(min(min(depth1),min(depth2)), max(max(depth1), max(depth2)))
    if verbose: 
        print("Depths generated:")
        print(depths)
    
    # Interpolate between given damage factors
    d1_interp = np.interp(depths, depth1, damage_fac1, left=0, right=damage_fac1[-1])
    d2_interp = np.interp(depths, depth2, damage_fac2, left=0, right=damage_fac1[-1])

    # Generate a random matrix of uncertainties
    error1 = uniform.rvs(loc=-0.3, scale=0.6, size=(nsow,1), random_state=rng)
    error2 = uniform.rvs(loc=-0.3, scale=0.6, size=(nsow,1), random_state=rng)
    # Apply uncertainties to damage factors
    damage_unc1 = d1_interp * (1+error1)
    damage_unc2 = d2_interp * (1+error2)

    # Return depth-damage function
    if ddf_type == "deep":
        # Randomly choose between curve 1 and 2 for each row
        selector = rng.choice([0, 1], size=nsow)
        # Apply the chosen damage factors to all the depths in the row (SOW)
        ret_damage = np.where(selector[:, None] == 0, damage_unc1, damage_unc2)
    elif ddf_type == "eu": ret_damage = damage_unc1
    elif ddf_type == "hazus": ret_damage = damage_unc2
    else: raise ValueError(f"Depth-damage function {ddf_type} unknown")
    
    # Array size: nrow: 50 (default from np.linspace), ncol: nsow+1 (the first row is depths)
    return np.vstack((depths, ret_damage))

# 4. Flooding frequency (uncertainty around GEV parameters)
# The GEV parameters are sampled so that they are different for each year
def gev_param_unc(nsow, mu_chain, sigma_chain, xi_chain):
    # Generate random INTEGER indices from 0 to len(mu_chain)-1
    # Re-sample for each year of ife_span
    # replace=True means sampled with replacement
    indices = rng.choice(len(mu_chain), size=nsow, replace=True)
    
    # Allocate the matrix (nsow rows, 3*ncol cols)
    params_unc = np.empty((nsow, 3))
    
    # Extract matching values using the integer indices
    params_unc[:, 0] = mu_chain[indices]
    params_unc[:, 1] = sigma_chain[indices]
    params_unc[:, 2] = xi_chain[indices]
    
    return params_unc

# 5. House value
# Dependent on the intensity of flooding (to be implemented later)
def house_value_unc(init_value, nsow, delta_h=0, life_span=200, elev_year=0):
    # Dummy values for a deterministic, linear change of house value
    appr_rate = 0.035   # Appreciation based on attractiveness
    stdev = 0.02
    # risk_rate = -0.04   # Depreciation rate based on flood risk
    # elev_rate = 0.01    # Impact of each foot of elevation

    # Sample appreciation rates across nsows
    rates = rng.normal(loc=appr_rate, scale=stdev, size=nsow)

    years = np.arange(0, life_span)
    factor = (1 + rates[:, np.newaxis]) ** years    # make rates shape: (nsow, 1), factor shape: (nsow, life_span)
    # Prepend a factor of 1 for year 0
    factors = np.hstack([np.ones((nsow, 1)), factor])
    val_unc = init_value * factors                  # shape: (nsow, life_span+1)

    return val_unc

# 6. Evolving GEV coefficients
# Sample coefficient values that evolve the location (mu) and scale (sigma) of the GEV function
def coefficient_unc(nsow):
    # beta_1 is the coefficient value for mu in the GEV function
    # where mu(t) = mu_0 + beta_1*t
    b1 = 0
    b1_std = 0
    beta_1 = rng.normal(loc=b1, scale=b1_std, size=(nsow,1))

    # beta_2 is the coefficient value for sigma in the GEV function
    # where sigma(t) = exp(ln(sigma_0) + beta_2*t)
    b2 = 0
    b2_std = 0
    beta_2 = rng.normal(loc=b2, scale=b2_std, size=(nsow,1))

    coeffs = np.vstack((beta_1, beta_2))
    return coeffs

## ------------------------------------------------------------------
## GENERATE PARETO FRONT
## ------------------------------------------------------------------

# Set parameters
delta_h_seq = np.linspace(start=0, stop=14, num=30)
nsow = 10000
num_strat = len(delta_h_seq)
yr_elev = 10     # What year the house is elevated

# Read in data files
obs_discount = pd.read_csv('discount.csv')                          # historical discount rate
mu_chain = pd.read_csv('mu_chain.csv').to_numpy().flatten()         # mu chain generated from R
sigma_chain = pd.read_csv('sigma_chain.csv').to_numpy().flatten()
xi_chain = pd.read_csv('xi_chain.csv').to_numpy().flatten()

# --- 1. Generate uncertainties ---
dr_unc = discount_rate_unc(obs_discount, nsow)
lt_unc = lifetime_unc(nsow)
ddf_unc = depth_damage_unc(nsow)
gev_unc = gev_param_unc(nsow, mu_chain, sigma_chain, xi_chain)
val_unc = house_value_unc(struc_value, nsow)

# Allocate ensemble array and perform Latin hypercube sampling
ens = np.empty((nsow, 1056))        # 201 + 201 + 201 + 201 + 1 + 50 + 201
sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=nsow)

i_sow = np.floor(sample * nsow).astype(int)
# Avoid sampling row 0 (depths) for the depth-damage function
i_sow[:, 3] = np.floor(sample[:, 3] * (nsow - 2)).astype(int) + 1

# Map parameters to ensemble matrix
ens[:, 0:3] = gev_unc[i_sow[:,0], :]         
ens[:, 3:204] = dr_unc[i_sow[:,1], :]        
ens[:, 204] = lt_unc[i_sow[:,2]]             
ens[:, 205:255] = ddf_unc[i_sow[:,3], :] 
ens[:, 255:456] = val_unc[i_sow[:,4], :]

# --- 2. Evaluate Strategies ---
led_ens = np.zeros((num_strat, nsow))   # allocate lifetime expected damages
cc_ens = np.zeros((num_strat, nsow))    # allocate construction cost
lr_ens = np.zeros((num_strat, nsow))    # allocate reliability

# Get parameters outside of the loop
mus, sigmas, xis = ens[:, 0], ens[:, 1], ens[:, 2]  # get mu, sigma, and xi from ensemble
drs = ens[:, 3:204]               # get discount rates from ensemble (trimmed to house lifetime in led function)
lifespans = ens[:, 204]             # get house lifetime from ensemble
dd_depths = ddf_unc[0, :]           # depths are the same regardles of SOW
dd_damages = ens[:, 205:255]        # get damage values from ensemble
house_vals = ens[:, 255:456]       # get house values (trimmed to house lifetime in led function)

# Determine construction cost, lifetime damages, and reliability for each strategy
for i, dh in enumerate(delta_h_seq):
    cc_ens[i] = construction_cost(dh, sqft, yr_elev, drs)

    led_ens[i, :] = lifetime_expected_damages(
        house_vals, init_elev, dh, lifespans, drs, mus, 
        sigmas, xis, dd_depths, dd_damages, yr_elev
    )
    lr_ens[i, :] = lifetime_reliability(
        lifespans, mus, sigmas, xis, init_elev, dh, yr_elev
    )

# --- 3. Calculate Objectives ---
# Total cost
tc_ens = led_ens + cc_ens

# BCR for all strategies
baseline_led = led_ens[0, :]    # Lifetime expected damages if no elevation
bcr_ens = (baseline_led - led_ens) / cc_ens

# Robustness (satisficing) for each strategy
# This creates a boolean mask for all strategies/SOWs at once
robustness_mask = (
    (bcr_ens > 1) & 
    (lr_ens > 0.5) & 
    ((tc_ens / struc_value) < 1)
)
# Mean across SOWs (axis=1) gives the score for each strategy
robustness_scores = np.mean(robustness_mask, axis=1) * 100

# Store results
results = []
for i, dh in enumerate(delta_h_seq):
    results.append({
        'nsow': nsow,
        'dh': dh,
        'upfront_cost': np.mean(cc_ens[i, :]),
        'total_cost': np.mean(tc_ens[i, :]),
        'bcr': np.mean(bcr_ens[i, :]) if dh > 0 else np.nan,
        'reliability': np.mean(lr_ens[i, :]),
        'satisficing': robustness_scores[i]
    })

df_results = pd.DataFrame(results)
df_results.to_csv('objectives_yr10_hv3.csv', index=False)
if verbose: print("\nResults saved to 'objectives_yr10_hv3.csv'")

## ------------------------------------------------------------------
## CONVERGENCE TESTING & UNCERTAINTY ENSEMBLES
## ------------------------------------------------------------------

# # Set parameters for convergence testing
# delta_h_seq = np.array([0,3,9,14])
# nsow_values = [10, 100, 1000, 10000, 100000]
# iterations = 10
# yr_elev = 0     # What year the house is elevated

# rng = np.random.default_rng()
# verbose = True

# # Read in data files (Load outside the loop to save time)
# obs_discount = pd.read_csv('discount.csv')
# mu_chain = pd.read_csv('mu_chain.csv').to_numpy().flatten()
# sigma_chain = pd.read_csv('sigma_chain.csv').to_numpy().flatten()
# xi_chain = pd.read_csv('xi_chain.csv').to_numpy().flatten()

# # List to store our convergence results
# convergence_results = []
# num_strat = len(delta_h_seq)

# print("Starting Convergence Testing...\n")

# # Outer loop: Iterate through different sample sizes
# for current_nsow in nsow_values:
#     print(f"Testing nsow = {current_nsow}...")
    
#     # Inner loop: Run iterations for the current sample size
#     for it in range(iterations):
#         if verbose: print(f"\tIteration {it+1}/{iterations}")
        
#         # --- 1. Generate uncertainties ---
#         dr_unc = discount_rate_unc(obs_discount, current_nsow)
#         lt_unc = lifetime_unc(current_nsow)
#         ddf_unc = depth_damage_unc(current_nsow)
#         gev_unc = gev_param_unc(current_nsow, mu_chain, sigma_chain, xi_chain)
#         val_unc = house_value_unc(struc_value, current_nsow)
        
#         # Allocate ensemble array and perform Latin hypercube sampling
#         ens = np.empty((current_nsow, 1056))        # 201 + 201 + 201 + 201 + 1 + 50 + 201
#         sampler = qmc.LatinHypercube(d=5)
#         sample = sampler.random(n=current_nsow)

#         i_sow = np.floor(sample * current_nsow).astype(int)
#         # Avoid sampling row 0 (depths) for the depth-damage function
#         i_sow[:, 3] = np.floor(sample[:, 3] * (current_nsow - 2)).astype(int) + 1

#         # Map parameters to ensemble matrix
#         ens[:, 0:3] = gev_unc[i_sow[:,0], :]         
#         ens[:, 3:204] = dr_unc[i_sow[:,1], :]        
#         ens[:, 204] = lt_unc[i_sow[:,2]]             
#         ens[:, 205:255] = ddf_unc[i_sow[:,3], :] 
#         ens[:, 255:456] = val_unc[i_sow[:,4], :]
        
#         # --- 2. Evaluate Strategies ---
#         led_ens = np.zeros((num_strat, current_nsow))   # allocate lifetime expected damages
#         cc_ens = np.zeros((num_strat, current_nsow))    # allocate construction cost
#         lr_ens = np.zeros((num_strat, current_nsow))    # allocate reliability

#         # Get parameters outside of the loop
#         mus, sigmas, xis = ens[:, 0], ens[:, 1], ens[:, 2]  # get mu, sigma, and xi from ensemble
#         drs = ens[:, 3:204]               # get discount rates from ensemble (trimmed to house lifetime in led function)
#         lifespans = ens[:, 204]             # get house lifetime from ensemble
#         dd_depths = ddf_unc[0, :]           # depths are the same regardles of SOW
#         dd_damages = ens[:, 205:255]        # get damage values from ensemble
#         house_vals = ens[:, 255:456]       # get house values (trimmed to house lifetime in led function)

#         # Determine construction cost, lifetime damages, and reliability for each strategy
#         for i, dh in enumerate(delta_h_seq):
#             cc_ens[i] = construction_cost(dh, sqft, yr_elev, drs)

#             led_ens[i, :] = lifetime_expected_damages(
#                 house_vals, init_elev, dh, lifespans, drs, mus, 
#                 sigmas, xis, dd_depths, dd_damages, yr_elev
#             )
#             lr_ens[i, :] = lifetime_reliability(
#                 lifespans, mus, sigmas, xis, init_elev, dh, yr_elev
#             )
        
#         # --- 3. Calculate Objectives ---
#         # Total cost
#         tc_ens = led_ens + cc_ens

#         # BCR for all strategies
#         baseline_led = led_ens[0, :]    # Lifetime expected damages if no elevation
#         bcr_ens = (baseline_led - led_ens) / cc_ens

#         # Robustness (satisficing) for each strategy
#         # This creates a boolean mask for all strategies/SOWs at once
#         robustness_mask = (
#             (bcr_ens > 1) & 
#             (lr_ens > 0.5) & 
#             ((tc_ens / struc_value) < 1)
#         )
#         # Mean across SOWs (axis=1) gives the score for each strategy
#         robustness_scores = np.mean(robustness_mask, axis=1) * 100

#         # Store results
#         for i, dh in enumerate(delta_h_seq):
#             convergence_results.append({
#                 'nsow': current_nsow,
#                 'iteration': it+1,
#                 'dh': dh,
#                 'upfront_cost': np.mean(cc_ens[i, :]),
#                 'total_cost': np.mean(tc_ens[i, :]),
#                 'bcr': np.mean(bcr_ens[i, :]) if dh > 0 else np.nan,
#                 'reliability': np.mean(lr_ens[i, :]),
#                 'satisficing': robustness_scores[i]
#             })

# # Convert to DataFrame
# df_convergence = pd.DataFrame(convergence_results)
# df_convergence.to_csv('convergence_hv35sd2.csv', index=False)
# print("\nResults saved to 'convergence_hv35sd2.csv'")

# print("\n" + "="*60)
# print("CONVERGENCE TESTING RESULTS")
# print("="*60)
# print(df_convergence)

# ## ------------------------------------------------------------------
# ## PLOT CONVERGENCE TESTING
# ## ------------------------------------------------------------------
# def plot_convergence(df, heights_to_plot=[3, 9, 14]):
#     metrics = ['total_cost', 'bcr', 'reliability']
#     titles = ['Total Cost ($)', 'Benefit-Cost Ratio', 'Lifetime Reliability']
    
#     for height in heights_to_plot:
#         # Filter data for specific height
#         data_subset = df[df['dh'] == height]
        
#         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#         fig.suptitle(f'Convergence Analysis for Heightening Strategy dh = {height}ft', fontsize=16)
        
#         for idx, metric in enumerate(metrics):
#             # Using stripplot to show all 10 points (iterations) per nsow
#             sns.stripplot(ax=axes[idx], data=data_subset, x='nsow', y=metric, 
#                           jitter=0.2, alpha=0.6, palette="viridis")
            
#             # Add a line to show the trend of the mean across iterations
#             sns.pointplot(ax=axes[idx], data=data_subset, x='nsow', y=metric, 
#                           color='black', markers='D', scale=0.5)
            
#             axes[idx].set_title(titles[idx])
#             axes[idx].grid(True, linestyle='--', alpha=0.7)
            
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.savefig(f'convergence_hv35sd2_{height}')

# # Run the plotting function
# plot_convergence(df_convergence)
