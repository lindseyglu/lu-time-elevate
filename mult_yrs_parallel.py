# -*- coding: utf-8 -*-
"""
Filename: mult_yrs_parallel.py
Author: Lindsey Lu
Created: 2026-02-05
Version: 2.0
Description: Evaluates both height strategies and implementation years 
             leveraging an identical uncertainty ensemble. Uses joblib
             to parallelize the functions to run faster.
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
import time
from joblib import Parallel, delayed

# For calculating runtime
start = time.time()

# Set print
verbose = True

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

# Set depth-damage function
depth = np.array([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])  # defined relative the FFE
damage_fac = np.array([0,0,4,8,12,15,20,23,28,33,37,43,48,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81])

## ------------------------------------------------------------------
## DEFINE GEV FUNCTION FOR FASTER COMPUTATION
## ------------------------------------------------------------------

def fast_gev_cdf(x, c, loc, scale):
    z = (x - loc) / scale
    inner = 1 + c * z
    return np.exp(-np.power(np.maximum(inner, 1e-10), -1/c))

## ------------------------------------------------------------------
## VECTORIZED PROFILE ENGINE
## ------------------------------------------------------------------

def compute_chunk_ead(struc_value, elev, mu, sigma, xi, DD_Depth, DD_Damage):
    """Computes a full time-series EAD matrix for a chunk across all SOWs."""
    crit_depths = DD_Depth + elev              
    crit_probs = 1 - fast_gev_cdf(
        x=crit_depths[np.newaxis, :, np.newaxis],    
        c=xi[:, np.newaxis, np.newaxis],                  
        loc=mu[:, np.newaxis, :],                         
        scale=sigma[:, np.newaxis, :]                     
    )
    crit_probs = np.nan_to_num(crit_probs, nan=0.0)         
    prob_diffs = -np.diff(crit_probs, axis=1, append=0)     
    return np.einsum('sdt,sd->st', prob_diffs, DD_Damage / 100) * struc_value

def construction_cost(delta_h, sqft, yr_elev, disc_rate):
    base_cost = 10000 + 300 + 470 + 4300 + 2175 + 3500
    Hs = np.array([3, 5, 8.5, 12, 14])
    Rates = np.array([80.36, 82.5, 86.25, 103.75, 113.75])

    if 3 <= delta_h <= 14:
        rate = np.interp(delta_h, Hs, Rates)
    else:     
        rate = 0
  
    raise_cost = base_cost + rate * sqft
    if delta_h == 0: 
        return np.zeros(len(disc_rate))
    
    infl = (1 + 0.03) ** yr_elev
    dr = disc_rate[:, yr_elev]
    return raise_cost * infl * dr

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
    appr_rate = 0   # Appreciation based on attractiveness
    stdev = 0
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
    b1 = 0.03
    b1_std = 0.3*b1
    # Could use a uniform distribution if there is deep uncertainty
    # Sweet et al. 2022 estimate 0.40m [0.31,0.49] of sea level rise from 2000 to 2050 = 0.0262 ft/yr
    beta_1 = rng.normal(loc=b1, scale=b1_std, size=nsow)

    # beta_2 is the coefficient value for sigma in the GEV function
    # where sigma(t) = exp(ln(sigma_0) + beta_2*t)
    # Normal: (0.001,0.001)
    b2 = 0.005
    b2_std = 0.3*b2
    beta_2 = rng.normal(loc=b2, scale=b2_std, size=nsow)

    coeffs = np.column_stack((beta_1, beta_2))
    return coeffs

## ------------------------------------------------------------------
## PARALLEL WORKER FUNCTION
## ------------------------------------------------------------------

def process_single_chunk(start_idx, end_idx, delta_h_seq, yr_elev_seq, init_elev, sqft,
                         house_vals, lifespans, drs, mus, sigmas, xis, dd_damages, dd_depths):
    """Processes a distinct subset of SOWs across all strategies and years."""
    num_strat = len(delta_h_seq)
    num_years = len(yr_elev_seq)
    chunk_len = end_idx - start_idx
    years_arr = np.arange(201)
    
    # Isolate parameters for this chunk
    hv_c = house_vals[start_idx:end_idx]
    ls_c = lifespans[start_idx:end_idx]
    dr_c = drs[start_idx:end_idx]
    mu_c = mus[start_idx:end_idx]
    sig_c = sigmas[start_idx:end_idx]
    xi_c = xis[start_idx:end_idx]
    ddd_c = dd_damages[start_idx:end_idx]
    
    lifespan_mask = years_arr < ls_c[:, np.newaxis]
    
    # Temporary storage arrays for the chunk's results
    led_chunk = np.zeros((num_strat, num_years, chunk_len))
    cc_chunk = np.zeros((num_strat, num_years, chunk_len))
    lr_chunk = np.zeros((num_strat, num_years, chunk_len))
    
    # Baseline computed exactly once per chunk
    ead_init = compute_chunk_ead(hv_c, init_elev, mu_c, sig_c, xi_c, dd_depths, ddd_c)
    prob_init = fast_gev_cdf(x=init_elev, c=xi_c[:, np.newaxis], loc=mu_c, scale=sig_c)
    
    for i, dh in enumerate(delta_h_seq):
        curr_elev = init_elev + dh
        
        # Strategy-specific metrics computed exactly once per chunk
        ead_elev = compute_chunk_ead(hv_c, curr_elev, mu_c, sig_c, xi_c, dd_depths, ddd_c)
        prob_elev = fast_gev_cdf(x=curr_elev, c=xi_c[:, np.newaxis], loc=mu_c, scale=sig_c)
        
        for j, y_ev in enumerate(yr_elev_seq):
            # Calculate construction costs natively inside the chunk context
            cc_chunk[i, j, :] = construction_cost(dh, sqft, y_ev, dr_c)
            
            # Slice damage profiles contextually at year of execution
            ead = np.hstack((ead_init[:, 0:y_ev], ead_elev[:, y_ev:201]))
            disc_ead = ead * dr_c
            led_chunk[i, j, :] = np.sum(disc_ead * lifespan_mask, axis=1)
            
            # Slice annual survival parameters contextually at year of execution
            p_annual = np.where(years_arr < y_ev, prob_init, prob_elev)
            p_annual = np.where(lifespan_mask, p_annual, 1.0)
            lr_chunk[i, j, :] = np.prod(p_annual, axis=1)
            
    return led_chunk, cc_chunk, lr_chunk

## ------------------------------------------------------------------
## SIMULATION COORDINATOR
## ------------------------------------------------------------------

if __name__ == '__main__':
    delta_h_seq = np.linspace(start=3, stop=14, num=30, endpoint=True)
    delta_h_seq = np.insert(delta_h_seq, 0, 0)
    yr_elev_seq = np.array([0, 10, 20])  

    nsow = 500000
    num_strat = len(delta_h_seq)
    num_years = len(yr_elev_seq)

    # Read in data files
    obs_discount = pd.read_csv('discount.csv')
    mu_chain = pd.read_csv('mu_chain.csv').to_numpy().flatten()
    sigma_chain = pd.read_csv('sigma_chain.csv').to_numpy().flatten()
    xi_chain = pd.read_csv('xi_chain.csv').to_numpy().flatten()

    # --- 1. Generate uncertainties ---
    gev_unc = gev_param_unc(nsow, mu_chain, sigma_chain, xi_chain)
    dr_unc = discount_rate_unc(obs_discount, nsow)
    lt_unc = lifetime_unc(nsow)
    ddf_unc = depth_damage_unc(nsow)
    val_unc = house_value_unc(struc_value, nsow)
    coe_unc = coefficient_unc(nsow)

    ens = np.empty((nsow, 458))
    sampler = qmc.LatinHypercube(d=6)
    sample = sampler.random(n=nsow)

    i_sow = np.floor(sample * nsow).astype(int)
    i_sow[:, 3] = np.floor(sample[:, 3] * (nsow - 2)).astype(int) + 1

    ens[:, 0:3] = gev_unc[i_sow[:,0], :]         
    ens[:, 3:204] = dr_unc[i_sow[:,1], :]        
    ens[:, 204] = lt_unc[i_sow[:,2]]             
    ens[:, 205:255] = ddf_unc[i_sow[:,3], :] 
    ens[:, 255:456] = val_unc[i_sow[:,4], :]
    ens[:, 456:458] = coe_unc[i_sow[:,5], :]

    mus_0, sigmas_0, xis = ens[:, 0], ens[:, 1], ens[:, 2]
    drs = ens[:, 3:204]
    lifespans = ens[:, 204]
    dd_depths = ddf_unc[0, :]
    dd_damages = ens[:, 205:255]
    house_vals = ens[:, 255:456]
    beta_1, beta_2 = ens[:, 456], ens[:, 457]

    t = np.arange(201)
    mus = mus_0[:, np.newaxis] + (beta_1[:, np.newaxis] * t)
    sigmas = np.exp(np.log(sigmas_0[:, np.newaxis]) + (beta_2[:, np.newaxis] * t))

    # --- 2. Parallel Evaluation ---
    chunk_size = 5000
    
    if verbose:
        print(f"Spawning parallel workers to process {nsow} SOWs in chunks of {chunk_size}...")

    parallel_outputs = Parallel(n_jobs=45)(
        delayed(process_single_chunk)(
            start_idx, min(start_idx + chunk_size, nsow),
            delta_h_seq, yr_elev_seq, init_elev, sqft,
            house_vals, lifespans, drs, mus, sigmas, xis, dd_damages, dd_depths
        )
        for start_idx in range(0, nsow, chunk_size)
    )

    if verbose:
        print("Consolidating multidimensional arrays from chunk processes...")
        
    # Stitch tracking matrices back together along the SOW axis (axis=2)
    led_ens = np.concatenate([out[0] for out in parallel_outputs], axis=2)
    cc_ens = np.concatenate([out[1] for out in parallel_outputs], axis=2)
    lr_ens = np.concatenate([out[2] for out in parallel_outputs], axis=2)

    # --- 3. Calculate Objectives via Vectorized Slicing ---
    tc_ens = led_ens + cc_ens

    # Broadcast the base un-elevated ledger (Strategy 0) across all heights
    baseline_led = led_ens[0, :, :] 
    bcr_ens = (baseline_led[np.newaxis, :, :] - led_ens) / cc_ens

    # Multidimensional Robustness (satisficing) evaluation
    robustness_mask = (
        (bcr_ens > 1) & 
        (lr_ens > 0.5) & 
        ((tc_ens / struc_value) < 1)
    )
    robustness_scores = np.mean(robustness_mask, axis=2) * 100 

    # Store results
    results = []
    for i, dh in enumerate(delta_h_seq):
        for j, y_ev in enumerate(yr_elev_seq):
            results.append({
                'nsow': nsow,
                'dh': dh,
                'yr_elev': y_ev,
                'upfront_cost': np.mean(cc_ens[i, j, :]),
                'damages': np.mean(led_ens[i, j, :]),
                'total_cost': np.mean(tc_ens[i, j, :]),
                'bcr': np.mean(bcr_ens[i, j, :]) if dh > 0 else np.nan,
                'reliability': np.mean(lr_ens[i, j, :]),
                'satisficing': robustness_scores[i, j]
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv('data/objectives_evGEV_coeff3.csv', index=False)
    if verbose: print("\nResults saved to 'objectives_evGEV_coeff3.csv'")

    end = time.time()
    print(f"Runtime: {end - start} seconds")