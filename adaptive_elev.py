# -*- coding: utf-8 -*-
"""
Filename: adaptive_elev.py
Author: Lindsey Lu
Created: 2026-02-05
Version: 1.0
Description: Deterministic cost function that takes in house 
characteristics and determines the cost function
"""

# import libraries
import pandas as pd
import numpy as np
from scipy.stats import genextreme
from scipy.stats import weibull_min
from scipy.stats import uniform
from scipy.stats import qmc             # for Latin Hypercube Sampling
import matplotlib.pyplot as plt
import seaborn as sns

# Set print
verbose = False

# Set house characteristics and discount rate
sqft = 1500
struc_value = 300000
del_elev = -4           # difference in house elev and BFE
life_span = 30
dr_i = np.arange(201)
disc_rate = np.exp(-1 * (0.04 * dr_i))      # **needs to be changed**
bfe = 34.7              # generated in the R code
# Initial house elev
init_elev = bfe + del_elev
if verbose: 
    print("House parameters")
    print(f"Structure value: ${struc_value}")
    print(f"Lifetime: {life_span}")
    print(f"Initial elevation: {init_elev} ft")
    print(f"Base flood elevation: {bfe}")
    print(f"Discount rate: {disc_rate}")

# Set GEV parameters (currently using GEV from Zarekarizi et al. 2020)
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

## Total cost is lifetime expected damages + construction cost

# Step 1: Calculate lifetime expected damages
def lifetime_expected_damages(struc_value, init_elev, delta_h, life_span, disc_rate, mu, sigma, xi, DD_Depth, DD_Damage):
    """
    Docstring for lifetime_expected_damages
    
    :param struc_value: structure value
    :param init_elev: initial house elevation
    :param delta_h: height the house is being raised by
    :param life_span: house lifespan
    :param disc_rate: array of discount rates for each year of the house lifespan
    :param mu: location parameter for generalized extreme value (GEV) distribution
    :param sigma: scale parameter for GEV
    :param xi: shape parameter for GEV
    :param DD_Depth: depths from the depth-damage function, defined relative to FFE
    :param DD_Damage: damage factor from the depth-damage function, defined out of 100
    """
    curr_elev = init_elev + delta_h # elevation of house after being elevated
    
    # Damage value lost at each depth, which depends on house value
    damage_vals = (DD_Damage/100) * struc_value

    # Critical depths are depths where the damage factor changes.
    crit_depths = DD_Depth + curr_elev # Calculates the stage of critical depths

    # Probability that water level exceeds each critical depth in one year
    # When c < 0, Frechet-type tail
    crit_probs = 1 - genextreme.cdf(x=crit_depths, c=-xi, loc=mu, scale=sigma)

    # Block avoids NaNs and NAs
    if np.any(np.isnan(crit_probs)):
        # Generate samples to find the empirical range of the distribution
        test_x = genextreme.rvs(c=-xi, loc=mu, scale=sigma, size=10**6)
        
        # Where crit_probs is NaN, check if depth is below min or above max observed
        mask_nan = np.isnan(crit_probs)
        crit_probs[mask_nan & (crit_depths < np.min(test_x))] = 0
        crit_probs[mask_nan & (crit_depths > np.max(test_x))] = 1
        
        # If NaNs still exist after clamping, raise error
        if np.any(np.isnan(crit_probs)):
            raise ValueError("I dont know what to do.... (NaNs persisted in crit_probs)")

    # Calculate the expected annual damages (EAD)
    EADfrac = np.empty(len(crit_depths))
    for i in range(len(crit_depths)):
        if i==len(crit_depths)-1:
            EADfrac[i] = crit_probs[i] * damage_vals[i]
        else:
            EADfrac[i] = (crit_probs[i] - crit_probs[i+1]) * damage_vals[i]
    EAD = sum(EADfrac)

    # [ensure that disc_rate is the same length as the lifespan]
    disc_sum = sum(disc_rate)
    exp_dam = EAD*disc_sum

    return exp_dam

# Step 2: Calculate construction cost
def construction_cost(delta_h, sqft):
    """
    Docstring for construction_cost
    
    :param delta_h: height the house is being raised by
    :param sqft: house square footage
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
    
    return(raise_cost)

## Reliability (safety)

# Step 3: Calculate reliability (probability of not being flooded at all during the lifetime of the house)
def lifetime_reliability(life_span, mu, sigma, xi, init_elev, delta_h):
    """
    Docstring for lifetime_reliability
    
    :param life_span: house lifetime
    :param mu: location parameter for GEV
    :param sigma: scale parameter for GEV
    :param xi: shape parameter for GEV
    :param init_elev: house initial elevation
    :param delta_h: height the house is being raised by
    """
    curr_elev = init_elev + delta_h

    # Safety is probability of zero floods during the next n years where n is the expected lifetime of the house
    # When c < 0, Frechet-type tail
    safety = genextreme.cdf(x=curr_elev, c=-xi, loc=mu, scale=sigma) ** (life_span//1)
    return(safety)

## Benefit-cost ratio
# Benefit is the cost of not elevating (delta_h=0) minus the cost of elevating
# BCR is benefit divided by the construction cost

## Satisficing (robustness)
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

# ## Evaluate strategies
# # Create empty arrays for damages, construction cost, reliability, and satisficing
# led = np.empty(len(delta_h_seq))
# cc = np.empty(len(delta_h_seq))
# lr = np.empty(len(delta_h_seq))
# sa = np.empty(len(delta_h_seq))

# # Iterate through strategies
# for i in range(len(delta_h_seq)):
#     if verbose: print(f"Evaluating strategy {i+1} of {len(delta_h_seq)}")
#     # Step 1: lifetime expected damages
#     led[i] = lifetime_expected_damages(struc_value, init_elev, delta_h_seq[i], life_span, 
#                                     disc_rate, mu, sigma, xi, depth, damage_fac)
#     # Step 2: construction cost
#     cc[i] = construction_cost(delta_h_seq[i], sqft)
#     # Step 3: reliability
#     lr[i] = lifetime_reliability(life_span, mu, sigma, xi, init_elev, delta_h_seq[i])

# # Step 4: total cost
# tc = led+cc
# # Step 5: benefit-cost ratio
# bcr_cost = cc
# bcr_benefit = led[0]-led
# bcr = bcr_cost / bcr_benefit

# # Find optimal strategy not considering uncertainty
# i_min = np.nanargmin(tc)       # can change to maximize BCR instead of minimize TC
# opt_h = delta_h_seq[i_min]  # optimal height
# opt_h_led = led[i_min]      # damages at optimal height
# opt_h_cc = cc[i_min]        # construction cost at opt h
# opt_h_tc = tc[i_min]        # total cost at opt h
# opt_h_bcr = bcr[i_min]      # benefit-cost ratio at opt h
# opt_h_lr = lr[i_min]        # reliability at opt h
# # Step 6: satisficing
# opt_h_sa = satisficing_all(opt_h_bcr, opt_h_lr, opt_h_tc, struc_value)

# if verbose:
#     print(f"Optimal height without uncertainty: {opt_h}")
#     print(f"\tDamages: {opt_h_led}")
#     print(f"\tTotal cost: {opt_h_tc}")
#     print(f"\tBenefit-cost ratio: {opt_h_bcr}")
#     print(f"\tLifetime reliability: {opt_h_lr}")
#     print(f"\tSatisfies BCR: {opt_h_sa[0]}")
#     print(f"\tSatisfies reliability: {opt_h_sa[1]}")
#     print(f"\tSatisfies total cost / structure value: {opt_h_sa[2]}")

# ## Evaluate federal and state strategies

# # FEMA recommendation
# fema_h = bfe+1
# fema_delta_h = fema_h - init_elev   # raised height needed
# # Massachusetts elevation
# mass_h = bfe+2
# mass_delta_h = mass_h - init_elev   # raised height needed

# # Evaluate FEMA recommendation
# # Step 1: lifetime expected damages
# if verbose: print(f"Evaluating FEMA strategy (raise by {fema_delta_h})")
# fema_led = lifetime_expected_damages(struc_value, init_elev, fema_delta_h, life_span, 
#                                      disc_rate, mu, sigma, xi, depth, damage_fac)
# # Step 2: construction cost
# fema_cc = construction_cost(fema_delta_h, sqft)
# # Step 3: reliability
# fema_lr = lifetime_reliability(life_span, mu, sigma, xi, init_elev, fema_delta_h)
# # Step 4: total cost
# fema_tc = fema_led+fema_cc
# # Step 5: benefit-cost ratio
# fema_cost = fema_cc
# fema_benefit = led[0]-fema_led
# fema_bcr = fema_cost / fema_benefit
# # Step 6: satisficing
# fema_sa = satisficing_all(fema_bcr, fema_lr, fema_tc, struc_value)
# if verbose:
#     print(f"FEMA height to elevate: {fema_delta_h}")
#     print(f"\tDamages: {fema_led}")
#     print(f"\tTotal cost: {fema_tc}")
#     print(f"\tBenefit-cost ratio: {fema_bcr}")
#     print(f"\tLifetime reliability: {fema_lr}")
#     print(f"\tSatisfies BCR: {fema_sa[0]}")
#     print(f"\tSatisfies reliability: {fema_sa[1]}")
#     print(f"\tSatisfies total cost / structure value: {fema_sa[2]}")

# # Evaluate Massachusetts elevation
# # Step 1: lifetime expected damages
# if verbose: print(f"Evaluating Massachusetts strategy (raise by {mass_delta_h})")
# mass_led = lifetime_expected_damages(struc_value, init_elev, mass_delta_h, life_span, 
#                                      disc_rate, mu, sigma, xi, depth, damage_fac)
# # Step 2: construction cost
# mass_cc = construction_cost(mass_delta_h, sqft)
# # Step 3: reliability
# mass_lr = lifetime_reliability(life_span, mu, sigma, xi, init_elev, mass_delta_h)
# # Step 4: total cost
# mass_tc = mass_led+mass_cc
# # Step 5: benefit-cost ratio
# mass_cost = mass_cc
# mass_benefit = led[0]-mass_led
# mass_bcr = mass_cost / mass_benefit
# # Step 6: satisficing
# mass_sa = satisficing_all(mass_bcr, mass_lr, mass_tc, struc_value)
# if verbose:
#     print(f"Massachusetts height to elevate: {mass_delta_h}")
#     print(f"\tDamages: {mass_led}")
#     print(f"\tTotal cost: {mass_tc}")
#     print(f"\tBenefit-cost ratio: {mass_bcr}")
#     print(f"\tLifetime reliability: {mass_lr}")
#     print(f"\tSatisfies BCR: {mass_sa[0]}")
#     print(f"\tSatisfies reliability: {mass_sa[1]}")
#     print(f"\tSatisfies total cost / structure value: {mass_sa[2]}")

## Create uncertainty ensembles

# Consider 4 uncertainties:
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

    # 2. Parameters (Best-estimates from script)
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
def gev_param_unc(nsow, mu_chain, sigma_chain, xi_chain):
    # Generate random INTEGER indices from 0 to len(mu_chain)-1
    # replace=True means sampled with replacement
    indices = rng.choice(len(mu_chain), size=nsow, replace=True)
    
    # Allocate the matrix (nsow rows, 3 cols)
    params_unc = np.empty((nsow, 3))
    
    # Extract matching values using the integer indices
    params_unc[:, 0] = mu_chain[indices]
    params_unc[:, 1] = sigma_chain[indices]
    params_unc[:, 2] = xi_chain[indices]
    
    return params_unc


## ------------------------------------------------------------------
## Test finding the optimal elevation using SOW ensemble generated by the R code
## ------------------------------------------------------------------
# verbose = True

# if verbose: print("Loading pre-generated SOWs")
# ens = pd.read_csv('SOWs.csv').to_numpy()
# num_strat = len(delta_h_seq)
# current_nsow = 10000     # get number of columns (nsow) in ensemble

# # --- 2. Evaluate Strategies ---
# led_ens = np.zeros((num_strat, current_nsow))
# cc_ens = np.zeros(num_strat)
# lr_ens = np.zeros((num_strat, current_nsow))
# dd_depths = np.linspace(-4, 24, 100)
        
# for i, dh in enumerate(delta_h_seq):
#     if verbose: print(f"Evaluating strategy dh = {dh}")
#     cc_ens[i] = construction_cost(dh, sqft)

#     for j in range(current_nsow):
#         # Index 0 is the 'Unnamed' R index. Data starts at 1.
#         mu_sow, sigma_sow, xi_sow = ens[j, 1:4]

#         # Lifetime is at index 205
#         life_sow = int(np.floor(ens[j, 205]))

#         # Discount factors start at index 4
#         dr_sow = ens[j, 4 : 4 + min(life_sow, 201)]

#         # Damage factors are the 100 values from index 206 to 305
#         dd_damage_sow = ens[j, 206:306]
        
#         led_ens[i, j] = lifetime_expected_damages(
#             struc_value, init_elev, dh, life_sow, dr_sow, mu_sow, 
#             sigma_sow, xi_sow, dd_depths, dd_damage_sow
#             )
#         lr_ens[i, j] = lifetime_reliability(
#             life_sow, mu_sow, sigma_sow, xi_sow, init_elev, dh
#             )

# # --- 3. Calculate Derived Objectives ---
# mean_led_per_strategy = np.mean(led_ens, axis=1)
# if verbose:
#     print("Lifetime expected damages per strategy")
#     print(mean_led_per_strategy)
#     print("Construction cost per strategy")
#     print(cc_ens)
# tc_ens = led_ens + cc_ens[:, np.newaxis]
# bcr_ens = np.zeros((num_strat, current_nsow))
# for i in range(1, num_strat):
#     # Avoid division by zero if cc_ens is 0
#     if cc_ens[i] > 0:
#         bcr_ens[i, :] = (led_ens[0, :] - led_ens[i, :]) / cc_ens[i]

# # --- 4. Find Optimal Strategy ---
# mean_tc_per_strategy = np.mean(tc_ens, axis=1)
# if verbose: 
#     print(mean_tc_per_strategy)
# idx_opt_unc = np.argmin(mean_tc_per_strategy)
        
# opt_h_unc = delta_h_seq[idx_opt_unc]
# opt_tc = mean_tc_per_strategy[idx_opt_unc]
# opt_rel = np.mean(lr_ens[idx_opt_unc, :])
# opt_bcr = np.mean(bcr_ens[idx_opt_unc, :])
        
# # Calculate Robustness (Satisficing)
# robustness_mask = (
#     (bcr_ens[idx_opt_unc, :] > 1) & 
#     (lr_ens[idx_opt_unc, :] > 0.5) & 
#     ((tc_ens[idx_opt_unc, :] / struc_value) < 1)
# )
# robustness_score = np.mean(robustness_mask) * 100
        
# # --- 5. Store Results ---
# convergence_results = []
# convergence_results.append({
#     'nsow': current_nsow,
#     'optimal_height': opt_h_unc,
#     'mean_total_cost': opt_tc,
#     'mean_reliability': opt_rel,
#     'mean_bcr': opt_bcr,
#     'satisficing_score': robustness_score
# })

# # --- 6. Final Output ---
# # Convert results to DataFrame for easy viewing and saving
# df_convergence = pd.DataFrame(convergence_results)

# print("\n" + "="*60)
# print("CONVERGENCE TESTING RESULTS")
# print("="*60)
# print(df_convergence)

## ------------------------------------------------------------------
## CONVERGENCE TESTING & UNCERTAINTY ENSEMBLES
## ------------------------------------------------------------------

# Set parameters for convergence testing
delta_h_seq = np.array([3,9,14])
nsow_values = [10, 100, 1000, 10000]
iterations = 10
rng = np.random.default_rng()

# Read in data files (Load outside the loop to save time)
obs_discount = pd.read_csv('discount.csv')
mu_chain = pd.read_csv('mu_chain.csv').to_numpy().flatten()
sigma_chain = pd.read_csv('sigma_chain.csv').to_numpy().flatten()
xi_chain = pd.read_csv('xi_chain.csv').to_numpy().flatten()

# List to store our convergence results
convergence_results = []
num_strat = len(delta_h_seq)

print("Starting Convergence Testing...\n")

# Outer loop: Iterate through different sample sizes
for current_nsow in nsow_values:
    print(f"Testing nsow = {current_nsow}...")
    
    # Inner loop: Run iterations for the current sample size
    for it in range(iterations):
        if verbose: print(f"\tIteration {it+1}/{iterations}")
        
        # --- 1. Generate Uncertainties for this iteration ---
        dr_unc = discount_rate_unc(obs_discount, current_nsow)
        lt_unc = lifetime_unc(current_nsow)
        ddf_unc = depth_damage_unc(current_nsow)
        gev_unc = gev_param_unc(current_nsow, mu_chain, sigma_chain, xi_chain)
        
        # Allocate ensemble array and perform LHS
        ens = np.empty((current_nsow, 255))
        sampler = qmc.LatinHypercube(d=4)
        sample = sampler.random(n=current_nsow)
        
        i_sow = np.floor(sample * current_nsow).astype(int)
        # Avoid sampling row 0 (depths) for the depth-damage function
        i_sow[:, 3] = np.floor(sample[:, 3] * (current_nsow - 2)).astype(int) + 1
        
        # Map parameters to ensemble matrix
        ens[:, 0:3] = gev_unc[i_sow[:,0], :]         
        ens[:, 3:204] = dr_unc[i_sow[:,1], :]        
        ens[:, 204] = lt_unc[i_sow[:,2]]             
        ens[:, 205:255] = ddf_unc[i_sow[:,3], :]     
        
        # --- 2. Evaluate Strategies ---
        led_ens = np.zeros((num_strat, current_nsow))
        cc_ens = np.zeros(num_strat)
        lr_ens = np.zeros((num_strat, current_nsow))
        dd_depths = ddf_unc[0, :]
        
        for i, dh in enumerate(delta_h_seq):
            cc_ens[i] = construction_cost(dh, sqft)
            for j in range(current_nsow):
                mu_sow, sigma_sow, xi_sow = ens[j, 0:3]
                life_sow = int(np.floor(ens[j, 204]))
                dr_sow = ens[j, 3 : 3 + min(life_sow, 201)]
                dd_damage_sow = ens[j, 205:255]
                
                # Use your corrected lifetime_expected_damages function here
                led_ens[i, j] = lifetime_expected_damages(
                    struc_value, init_elev, dh, life_sow, 
                    dr_sow, mu_sow, sigma_sow, xi_sow, dd_depths, dd_damage_sow
                )
                lr_ens[i, j] = lifetime_reliability(
                    life_sow, mu_sow, sigma_sow, xi_sow, init_elev, dh
                )

        # --- 3. Calculate Derived Objectives ---
        tc_ens = led_ens + cc_ens[:, np.newaxis]
        bcr_ens = np.zeros((num_strat, current_nsow))
        for i in range(1, num_strat):
            # Avoid division by zero if cc_ens is 0
            if cc_ens[i] > 0:
                bcr_ens[i, :] = (led_ens[0, :] - led_ens[i, :]) / cc_ens[i]

        # --- 4. Find Optimal Strategy ---
        mean_tc_per_strategy = np.mean(tc_ens, axis=1)
        idx_opt_unc = np.argmin(mean_tc_per_strategy)
        
        opt_h_unc = delta_h_seq[idx_opt_unc]
        opt_tc = mean_tc_per_strategy[idx_opt_unc]
        opt_rel = np.mean(lr_ens[idx_opt_unc, :])
        opt_bcr = np.mean(bcr_ens[idx_opt_unc, :])
        
        # Calculate Robustness (Satisficing)
        robustness_mask = (
            (bcr_ens[idx_opt_unc, :] > 1) & 
            (lr_ens[idx_opt_unc, :] > 0.5) & 
            ((tc_ens[idx_opt_unc, :] / struc_value) < 1)
        )
        robustness_score = np.mean(robustness_mask) * 100
        
        # --- 5. Store Results ---
        convergence_results.append({
            'nsow': current_nsow,
            'iteration': it + 1,
            'optimal_height': opt_h_unc,
            'mean_total_cost': opt_tc,
            'mean_reliability': opt_rel,
            'mean_bcr': opt_bcr,
            'satisficing_score': robustness_score
        })

# --- 6. Final Output ---
# Convert results to DataFrame for easy viewing and saving
df_convergence = pd.DataFrame(convergence_results)

print("\n" + "="*60)
print("CONVERGENCE TESTING RESULTS")
print("="*60)
print(df_convergence)

# Optional: Save results to CSV so you can analyze or plot them later
df_convergence.to_csv('convergence_testing_results_1.csv', index=False)
print("\nResults saved to 'convergence_testing_results_1.csv'")