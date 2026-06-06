# -*- coding: utf-8 -*-
"""
Filename: sensitivity.py
Author: Lindsey Lu
Created: 2026-05-14
Version: 2.0
Description: Conducts Sobol sensitivity analysis for elevation at 
             year 0 with the included uncertainties of evolving
             house value and GEV coefficients. Uses SAlib and joblib.
"""

# import libraries
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from scipy.stats import gaussian_kde
import time
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from joblib import Parallel, delayed, cpu_count

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

# Heightening strategy and year of elevation for evaluation
dh = 14
yr = 0

# Set depth-damage function (HAZUS)
depth = np.array([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])  # defined relative the FFE
damage_fac = np.array([0,0,4,8,12,15,20,23,28,33,37,43,48,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81])

# Define model inputs
# First, get values for the boundaries
mu_chain = pd.read_csv('mu_chain.csv').values.flatten()
sigma_chain = pd.read_csv('sigma_chain.csv').values.flatten()
xi_chain = pd.read_csv('xi_chain.csv').values.flatten()

# Sample the following parameters in Zarekarizi et al. (2020) "most-likely scenario"
# mu, sigma, and xi are sampled using Gaussian Kernel Density Estimate (KDE)
mu_kde = gaussian_kde(mu_chain, bw_method='silverman')
sigma_kde = gaussian_kde(sigma_chain, bw_method='silverman')
xi_kde = gaussian_kde(xi_chain, bw_method='silverman')

# Problem definition
problem = {
    'num_vars': 9,
    'names': ['mu_u', 'sigma_u', 'xi_u', 'dr_u', 'lt_u', 'dd_err', 'hv_rate', 'b1', 'b2'],
    'bounds': [[0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [-30, 30],
               [-0.025, 0.095],
               [0, 0.03],
               [0, 0.005]]
}

# Generates N*(2D+2) samples where N=32768 -> 655,360 rows
param_u = sobol_sample.sample(problem, 32768)

# Transform parameters
def transform_parameters(matrix_u, mu_kde, sigma_kde, xi_kde):
    physical_matrix = np.empty_like(matrix_u)
    physical_matrix[:, 5:9] = matrix_u[:, 5:9]
    
    pool_size = 500000
    physical_matrix[:, 0] = np.percentile(mu_kde.resample(pool_size), matrix_u[:, 0] * 100)
    physical_matrix[:, 1] = np.percentile(sigma_kde.resample(pool_size), matrix_u[:, 1] * 100)
    physical_matrix[:, 2] = np.percentile(xi_kde.resample(pool_size), matrix_u[:, 2] * 100)

    continuous_dr = 0.5 + (matrix_u[:, 3] * 10000.0)
    physical_matrix[:, 3] = np.round(continuous_dr).astype(int)

    physical_matrix[:, 4] = weibull_min.ppf(matrix_u[:, 4], c=2.8, scale=73.5)
    return physical_matrix

# Get true parameters
param_values = transform_parameters(param_u, mu_kde, sigma_kde, xi_kde)

# GEV calculation helper
def fast_gev_cdf(x, c, loc, scale):
    z = (x - loc) / scale
    inner = 1 + c * z
    return np.exp(-np.power(np.maximum(inner, 1e-10), -1/c))

# Calculate lifetime expected damages
def lifetime_expected_damages(struc_value, init_elev, delta_h, life_span, disc_rate, mu, sigma, xi, DD_Depth, DD_Damage, yr_elev):
    curr_elev = init_elev + delta_h 
    damage_vals = (DD_Damage[:, np.newaxis]/100) * struc_value[np.newaxis, :]

    crit_depths_elev = DD_Depth + curr_elev              
    crit_depths_init = DD_Depth + init_elev              

    crit_probs_elev = 1 - fast_gev_cdf(
        x=crit_depths_elev[:,np.newaxis],               
        c=xi,                                           
        loc=mu[np.newaxis,:],                           
        scale=sigma[np.newaxis,:]                       
    )

    crit_probs_init = 1 - fast_gev_cdf(
        x=crit_depths_init[:,np.newaxis],               
        c=xi,                                           
        loc=mu[np.newaxis,:],                           
        scale=sigma[np.newaxis,:]                       
    )

    crit_probs_elev = np.nan_to_num(crit_probs_elev, nan=0.0)         
    crit_probs_init = np.nan_to_num(crit_probs_init, nan=0.0)         

    prob_diffs_elev = -np.diff(crit_probs_elev, axis=0, append=0)     
    prob_diffs_init = -np.diff(crit_probs_init, axis=0, append=0)     
    ead_elev = np.sum(prob_diffs_elev * damage_vals, axis=0)          
    ead_init = np.sum(prob_diffs_init * damage_vals, axis=0)          

    years = np.arange(disc_rate.shape[0])
    ead = np.where(years < yr_elev, ead_init, ead_elev)
    disc_ead = ead * disc_rate     

    lifespan_mask = years < life_span
    exp_dam = np.sum(disc_ead * lifespan_mask)

    return exp_dam

# Fixed construction cost function (properly indexed for 1D slices and uses global sqft)
def construction_cost(delta_h, yr_elev, disc_rate, house_sqft):
    base_cost = 10000 + 300 + 470 + 4300 + 2175 + 3500
    Hs = np.array([3, 5, 8.5, 12, 14])
    Rates = np.array([80.36, 82.5, 86.25, 103.75, 113.75])

    if 3 <= delta_h <= 14:
        rate = np.interp(delta_h, Hs, Rates)
    else:     
        rate = 0
  
    raise_cost = base_cost + rate * house_sqft
    if delta_h == 0: 
        return 0.0
    
    infl = (1 + 0.03) ** yr_elev
    dr = disc_rate[yr_elev]  # Indexes the correct float element from 1D array
    return raise_cost * infl * dr

# Generate discount rate trajectories
def discount_rate_unc(obs_discount, nsow, dr_func="deep", life_span=200):
    d = np.log(np.array(obs_discount)[:, 1])
    n_cols = life_span + 1
    
    def run_ar3(model_type, pars):
        eps = np.full((nsow, life_span + 3), np.nan)
        last_3 = d[-3:]
        if model_type == "drift":
            offset = len(d) - 1
            time_range = np.arange(offset - 2, life_span + offset + 1)
            tr = pars['int'] + pars['slope'] * time_range
            eps[:, :3] = last_3 - tr[:3]
        elif model_type == "mrv": 
            eps[:, :3] = last_3 - np.log(pars['eta'])
        elif model_type == "rw":
            first_valid = d[~np.isnan(d)][0]
            eps[:, :3] = last_3 - first_valid
            
        sigma = np.sqrt(pars['sigma_sq'])
        for i in range(3, life_span + 3):
            innovation = rng.normal(0, sigma, nsow)
            eps[:, i] = (pars['rho1'] * eps[:, i-1] + 
                         pars['rho2'] * eps[:, i-2] + 
                         pars['rho3'] * eps[:, i-3] + innovation)
        
        if model_type == "drift":
            rates = np.exp(eps[:, 3:] + tr[3:])
        elif model_type == "mrv":
            rates = np.exp(np.log(pars['eta']) + eps[:, 3:])
        elif model_type == "rw":
            rates = np.exp(first_valid + eps[:, 3:])
            
        dfactors = np.exp(-1 * np.cumsum(rates / 100, axis=1))
        return np.hstack([np.ones((nsow, 1)), dfactors])

    params_rw = {'rho1': 1.7429, 'rho2': -1.0455, 'rho3': 0.3010, 'sigma_sq': 0.0034}
    params_mrv = {'eta': 3.405, 'rho1': 1.7371, 'rho2': -1.0270, 'rho3': 0.2806, 'sigma_sq': 0.0034}
    params_drift = {'int': 1.9289, 'slope': -0.0058, 'rho1': 1.6965, 'rho2': -0.9755, 'rho3': 0.2388, 'sigma_sq': 0.0033}

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
        selector = rng.choice([0, 1, 2], size=nsow)   
        m_deep = np.zeros((nsow, n_cols))
        m_deep[selector == 0] = m_rw[selector == 0]
        m_deep[selector == 1] = m_mrv[selector == 1]
        m_deep[selector == 2] = m_drift[selector == 2]
        return m_deep
    
    raise ValueError('Options are "rw", "mrv", "drift", "deep", or "cert-4%".')

obs_disc = pd.read_csv('discount.csv')
dr_trajectories = discount_rate_unc(obs_disc, 10000, life_span=200)

# --- Define the batch processing job function for Joblib ---
def process_batch(batch_indices, param_values_chunk, struc_value, init_elev, dh, depth, damage_fac, yr, dr_trajectories, sqft):
    local_Y = np.zeros(len(batch_indices))
    t = np.arange(201)
    
    for local_idx, X in enumerate(param_values_chunk):
        mu, sigma, xi, dr_float, lt, dd, hv_rate, b1, b2 = X
        dr_i = int(dr_float)

        disc_rate = dr_trajectories[dr_i-1]

        mu_t = mu + b1*t
        sigma_t = np.exp(np.log(sigma) + b2*t)
        house_value = struc_value * (1 + hv_rate*t)

        damage_adj = damage_fac + damage_fac*(dd/100)
        damage_adj = np.clip(damage_adj, 0, 100)    

        led = lifetime_expected_damages(house_value, init_elev, dh, lt, disc_rate, mu_t, sigma_t, xi, depth, damage_adj, yr)
        cc = construction_cost(dh, yr, disc_rate, sqft)

        local_Y[local_idx] = led + cc
        
    return batch_indices, local_Y

# --- Parallel Processing Setup ---
# Determine usable CPUs (leaves 1 core free)
n_jobs = max(1, cpu_count() - 1)
# Create array splits to chunk workload into large pieces
chunks = np.array_split(np.arange(param_values.shape[0]), n_jobs)

print(f"Starting parallel execution of model across {n_jobs} cores...")

results = Parallel(n_jobs=n_jobs, backend='loky')(
    delayed(process_batch)(
        chunk_indices, 
        param_values[chunk_indices], 
        struc_value, 
        init_elev, 
        dh, 
        depth, 
        damage_fac, 
        yr, 
        dr_trajectories,
        sqft
    ) for chunk_indices in chunks
)

# Allocate master matrix and re-stich calculations back together 
Y = np.zeros([param_values.shape[0]])
for chunk_indices, local_Y in results:
    Y[chunk_indices] = local_Y

print("Model execution completed. Running Sobol Analysis...")

# Perform analysis
Si = sobol_analyze.analyze(problem, Y)

# Save as csv - Separate Files for Main and Interacting Effects
df_main = pd.DataFrame({
    'Parameter': problem['names'],
    'S1': Si['S1'], 
    'S1_conf': Si['S1_conf'],
    'ST': Si['ST'], 
    'ST_conf': Si['ST_conf']
})
df_main.to_csv(f'sobol_main_effects_tc{dh}.csv', index=False)

if 'S2' in Si and Si['S2'] is not None:
    s2_rows = []
    p_names = problem['names']
    num_vars = len(p_names)
    
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            s2_rows.append({
                'Parameter_1': p_names[i],
                'Parameter_2': p_names[j],
                'S2': Si['S2'][i, j],
                'S2_conf': Si['S2_conf'][i, j]
            })
            
    df_s2 = pd.DataFrame(s2_rows).dropna(subset=['S2'])
    df_s2.to_csv(f'sobol_second_order_interactions_tc{dh}.csv', index=False)
    print(f"Successfully saved 'sobol_main_effects_tc{dh}.csv' and 'sobol_second_order_interactions_tc{dh}.csv'")
else:
    print("Main effects saved. No second-order indices found to export.")

end = time.time()
print(f"Runtime: {end - start} seconds")