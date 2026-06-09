# -*- coding: utf-8 -*-
"""
Filename: convergence_parallel.py
Author: Lindsey Lu
Description: Runs convergence testing parallelized.
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
from joblib import Parallel, delayed    # Parallelization library

# For calculating runtime
start = time.time()
verbose = True

# Set deterministic house characteristics and discount rate
sqft = 1500
struc_value = 300000
del_elev = -4           
life_span = 30
dr_i = np.arange(201)
disc_rate = np.exp(-1 * (0.04 * dr_i))
bfe = 34.7              
init_elev = bfe + del_elev  

# Set deterministic GEV parameters
mu = 19.8718901487264
sigma = 3.16814792683425
xi = 0.00515921024408503

# Set elevation height strategies to evaluate
delta_h_seq = np.array([0,3,4,5,6,7,8,9,10,11,12,13,14])

# Set depth-damage function
depth = np.array([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]) 
damage_fac = np.array([0,0,4,8,12,15,20,23,28,33,37,43,48,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81])

## ------------------------------------------------------------------
## DEFINE GEV FUNCTION FOR FASTER COMPUTATION
## ------------------------------------------------------------------

def fast_gev_cdf(x, c, loc, scale):
    z = (x - loc) / scale
    inner = 1 + c * z
    return np.exp(-np.power(np.maximum(inner, 1e-10), -1/c))

## ------------------------------------------------------------------
## OBJECTIVES FUNCTIONS
## (Unchanged from your original code)
## ------------------------------------------------------------------

def construction_cost(delta_h, sqft, yr_elev, disc_rate):
    base_cost= 10000 + 300 + 470 + 4300 + 2175 + 3500
    Hs=np.array([3,5,8.5,12,14])
    Rates=np.array([80.36,82.5,86.25,103.75,113.75])

    if 3 <= delta_h <= 14:
        rate=np.interp(delta_h, Hs, Rates)
    else:     
        rate=0
  
    raise_cost = base_cost + rate*sqft
    if(delta_h==0): raise_cost=0
    
    infl = (1+0.03)**yr_elev
    dr = disc_rate[:, yr_elev]
    npv_cost = raise_cost * infl * dr

    return(npv_cost)

def lifetime_expected_damages_chunk(struc_value, init_elev, delta_h, life_span, disc_rate, mu, sigma, xi, DD_Depth, DD_Damage, yr_elev, ead_init=None):
    curr_elev = init_elev + delta_h 
    years = np.arange(201)

    crit_depths_elev = DD_Depth + curr_elev             
    crit_probs_elev = 1 - fast_gev_cdf(
        x=crit_depths_elev[np.newaxis, :, np.newaxis],   
        c=xi[:, np.newaxis, np.newaxis],                 
        loc=mu[:, np.newaxis, :],                        
        scale=sigma[:, np.newaxis, :]                    
    )
    crit_probs_elev = np.nan_to_num(crit_probs_elev, nan=0.0)        
    prob_diffs_elev = -np.diff(crit_probs_elev, axis=1, append=0)    
    
    ead_elev = np.einsum('sdt,sd->st', prob_diffs_elev, DD_Damage / 100) * struc_value

    if ead_init is None:
        crit_depths_init = DD_Depth + init_elev             
        crit_probs_init = 1 - fast_gev_cdf(
            x=crit_depths_init[np.newaxis, :, np.newaxis],   
            c=xi[:, np.newaxis, np.newaxis],                 
            loc=mu[:, np.newaxis, :],                        
            scale=sigma[:, np.newaxis, :]                    
        )
        crit_probs_init = np.nan_to_num(crit_probs_init, nan=0.0)        
        prob_diffs_init = -np.diff(crit_probs_init, axis=1, append=0)    
        ead_init = np.einsum('sdt,sd->st', prob_diffs_init, DD_Damage / 100) * struc_value

    ead = np.hstack((ead_init[:, 0:yr_elev], ead_elev[:, yr_elev:201]))
    disc_ead = ead * disc_rate     
    lifespan_mask = years < life_span[:, np.newaxis]
    
    return np.sum(disc_ead * lifespan_mask, axis=1), ead_init

def lifetime_reliability_chunk(life_span, mu, sigma, xi, init_elev, delta_h, yr_elev, prob_init=None):
    curr_elev = init_elev + delta_h
    years = np.arange(201)

    if prob_init is None:
        prob_init = fast_gev_cdf(x=init_elev, c=xi[:, np.newaxis], loc=mu, scale=sigma)
        
    prob_elev = fast_gev_cdf(x=curr_elev, c=xi[:, np.newaxis], loc=mu, scale=sigma)

    p_annual = np.where(years < yr_elev, prob_init, prob_elev)  
    lifespan_mask = years < life_span[:, np.newaxis]
    p_annual = np.where(lifespan_mask, p_annual, 1.0)

    return np.prod(p_annual, axis=1), prob_init

## ------------------------------------------------------------------
## UNCERTAINTY FUNCTIONS (Updated to accept RNG parameter)
## ------------------------------------------------------------------

def discount_rate_unc(obs_discount, nsow, rng, dr_func="deep", life_span=200):
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
        
        if model_type == "drift": rates = np.exp(eps[:, 3:] + tr[3:])
        elif model_type == "mrv": rates = np.exp(np.log(pars['eta']) + eps[:, 3:])
        elif model_type == "rw": rates = np.exp(first_valid + eps[:, 3:])
            
        dfactors = np.exp(-1 * np.cumsum(rates / 100, axis=1))
        return np.hstack([np.ones((nsow, 1)), dfactors])

    params_rw = {'rho1': 1.7429, 'rho2': -1.0455, 'rho3': 0.3010, 'sigma_sq': 0.0034}
    params_mrv = {'eta': 3.405, 'rho1': 1.7371, 'rho2': -1.0270, 'rho3': 0.2806, 'sigma_sq': 0.0034}
    params_drift = {'int': 1.9289, 'slope': -0.0058, 'rho1': 1.6965, 'rho2': -0.9755, 'rho3': 0.2388, 'sigma_sq': 0.0033}

    if dr_func == "cert-4%": return np.tile(np.exp(-0.04 * np.arange(n_cols)), (nsow, 1))

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

def lifetime_unc(nsow, rng, lifetime_func="weibull"):
    if lifetime_func == "weibull": 
        lt_unc = weibull_min.rvs(c=2.8, scale=73.5, size=nsow, random_state=rng)
    else: raise ValueError(f"House lifetime function {lifetime_func} unknown")
    return lt_unc

def depth_damage_unc(nsow, rng, ddf_type="deep"):
    depth1 = np.array([0, 1.64, 3.28, 4.92, 6.56, 9.84, 13.12, 16.40])
    damage_fac1 = np.array([0.20, 0.44, 0.58, 0.68, 0.78, 0.85, 0.92, 0.96])*100
    depth2 = np.array([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    damage_fac2 = np.array([0,0,4,8,12,15,20,23,28,33,37,43,48,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81])

    depths = np.linspace(min(min(depth1),min(depth2)), max(max(depth1), max(depth2)))
    d1_interp = np.interp(depths, depth1, damage_fac1, left=0, right=damage_fac1[-1])
    d2_interp = np.interp(depths, depth2, damage_fac2, left=0, right=damage_fac1[-1])

    error1 = uniform.rvs(loc=-0.3, scale=0.6, size=(nsow,1), random_state=rng)
    error2 = uniform.rvs(loc=-0.3, scale=0.6, size=(nsow,1), random_state=rng)
    damage_unc1 = d1_interp * (1+error1)
    damage_unc2 = d2_interp * (1+error2)

    if ddf_type == "deep":
        selector = rng.choice([0, 1], size=nsow)
        ret_damage = np.where(selector[:, None] == 0, damage_unc1, damage_unc2)
    elif ddf_type == "eu": ret_damage = damage_unc1
    elif ddf_type == "hazus": ret_damage = damage_unc2
    else: raise ValueError(f"Depth-damage function {ddf_type} unknown")
    
    return np.vstack((depths, ret_damage))

def gev_param_unc(nsow, rng, mu_chain, sigma_chain, xi_chain):
    indices = rng.choice(len(mu_chain), size=nsow, replace=True)
    params_unc = np.empty((nsow, 3))
    params_unc[:, 0] = mu_chain[indices]
    params_unc[:, 1] = sigma_chain[indices]
    params_unc[:, 2] = xi_chain[indices]
    return params_unc

def house_value_unc(init_value, nsow, rng, delta_h=0, life_span=200, elev_year=0):
    appr_rate = 0.035   
    stdev = 0.02
    rates = rng.normal(loc=appr_rate, scale=stdev, size=nsow)
    years = np.arange(0, life_span)
    factor = (1 + rates[:, np.newaxis]) ** years    
    factors = np.hstack([np.ones((nsow, 1)), factor])
    return init_value * factors                  

def coefficient_unc(nsow, rng):
    b1 = 0.02
    b1_std = b1*0.3
    beta_1 = rng.normal(loc=b1, scale=b1_std, size=nsow)

    b2 = 0.003
    b2_std = b2*0.3
    beta_2 = rng.normal(loc=b2, scale=b2_std, size=nsow)
    return np.column_stack((beta_1, beta_2))

# ------------------------------------------------------------------
# PARALLEL WORKER FUNCTION
# ------------------------------------------------------------------

def simulate_iteration(it, current_nsow, delta_h_seq, mu_chain, sigma_chain, xi_chain, obs_discount, init_elev, struc_value, sqft, yr_elev):
    """
    Executes a single Monte Carlo iteration. Runs on an isolated CPU core.
    """
    # 1. Spawn a unique RNG state for this core to prevent duplicated random numbers
    local_rng = np.random.default_rng(seed=int(time.time()) + it)
    num_strat = len(delta_h_seq)
    
    # 2. Generate uncertainties using the local rng
    gev_unc = gev_param_unc(current_nsow, local_rng, mu_chain, sigma_chain, xi_chain)
    dr_unc = discount_rate_unc(obs_discount, current_nsow, local_rng)                  
    lt_unc = lifetime_unc(current_nsow, local_rng)                                     
    ddf_unc = depth_damage_unc(current_nsow, local_rng)                                
    val_unc = house_value_unc(struc_value, current_nsow, local_rng)                    
    coe_unc = coefficient_unc(current_nsow, local_rng)                                 

    # 3. LHS (qmc internal RNG handles its own seeding based on system entropy by default)
    ens = np.empty((current_nsow, 458))        
    sampler = qmc.LatinHypercube(d=6, seed=int(time.time()) + it) 
    sample = sampler.random(n=current_nsow)

    i_sow = np.floor(sample * current_nsow).astype(int)
    i_sow[:, 3] = np.floor(sample[:, 3] * (current_nsow - 2)).astype(int) + 1

    ens[:, 0:3] = gev_unc[i_sow[:,0], :]         
    ens[:, 3:204] = dr_unc[i_sow[:,1], :]        
    ens[:, 204] = lt_unc[i_sow[:,2]]             
    ens[:, 205:255] = ddf_unc[i_sow[:,3], :] 
    ens[:, 255:456] = val_unc[i_sow[:,4], :]
    ens[:, 456:458] = coe_unc[i_sow[:,5], :]

    led_ens = np.zeros((num_strat, current_nsow))   
    cc_ens = np.zeros((num_strat, current_nsow))    
    lr_ens = np.zeros((num_strat, current_nsow))    

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

    for i, dh in enumerate(delta_h_seq):
        cc_ens[i] = construction_cost(dh, sqft, yr_elev, drs)

    chunk_size = 5000
    for start_idx in range(0, current_nsow, chunk_size):
        end_idx = min(start_idx + chunk_size, current_nsow)
        
        hv_c = house_vals[start_idx:end_idx]
        ls_c = lifespans[start_idx:end_idx]
        dr_c = drs[start_idx:end_idx]
        mu_c = mus[start_idx:end_idx]
        sig_c = sigmas[start_idx:end_idx]
        xi_c = xis[start_idx:end_idx]
        ddd_c = dd_damages[start_idx:end_idx]
        
        cached_ead_init = None
        cached_prob_init = None
        
        for i, dh in enumerate(delta_h_seq):
            led_res, cached_ead_init = lifetime_expected_damages_chunk(
                hv_c, init_elev, dh, ls_c, dr_c, mu_c, sig_c, xi_c, 
                dd_depths, ddd_c, yr_elev, ead_init=cached_ead_init
            )
            led_ens[i, start_idx:end_idx] = led_res
            
            lr_res, cached_prob_init = lifetime_reliability_chunk(
                ls_c, mu_c, sig_c, xi_c, init_elev, dh, yr_elev, prob_init=cached_prob_init
            )
            lr_ens[i, start_idx:end_idx] = lr_res

    tc_ens = led_ens + cc_ens
    baseline_led = led_ens[0, :]    
    bcr_ens = (baseline_led - led_ens) / cc_ens

    robustness_mask = (
        (bcr_ens > 1) & 
        (lr_ens > 0.5) & 
        ((tc_ens / struc_value) < 1)
    )
    robustness_scores = np.mean(robustness_mask, axis=1) * 100

    # Package results for this iteration
    iteration_results = []
    for i, dh in enumerate(delta_h_seq):
        iteration_results.append({
            'nsow': current_nsow,
            'iteration': it + 1,
            'dh': dh,
            'upfront_cost': np.mean(cc_ens[i, :]),
            'total_cost': np.mean(tc_ens[i, :]),
            'bcr': np.mean(bcr_ens[i, :]) if dh > 0 else np.nan,
            'reliability': np.mean(lr_ens[i, :]),
            'satisficing': robustness_scores[i]
        })
        
    return iteration_results

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------

if __name__ == '__main__':
    delta_h_seq = np.array([0,3,9,14])
    nsow_values = [100, 1000, 10000, 100000, 500000]
    iterations = 10
    yr_elev = 0     

    # Read in data files
    obs_discount = pd.read_csv('inputs/discount.csv')
    mu_chain = pd.read_csv('inputs/mu_chain.csv').to_numpy().flatten()
    sigma_chain = pd.read_csv('inputs/sigma_chain.csv').to_numpy().flatten()
    xi_chain = pd.read_csv('inputs/xi_chain.csv').to_numpy().flatten()

    all_convergence_results = []

    print("Starting Parallel Convergence Testing...\n")

    for current_nsow in nsow_values:
        print(f"Testing nsow = {current_nsow} with {iterations} iterations...")
        
        # Deploy parallel workers! 
        # n_jobs=-1 uses all available cores. 
        results_list = Parallel(n_jobs=-1)(
            delayed(simulate_iteration)(
                it, current_nsow, delta_h_seq, mu_chain, sigma_chain, 
                xi_chain, obs_discount, init_elev, struc_value, sqft, yr_elev
            ) 
            for it in range(iterations)
        )
        
        # Flatten the list of lists returned by Parallel
        for res_group in results_list:
            all_convergence_results.extend(res_group)

    # Convert to DataFrame
    df_convergence = pd.DataFrame(all_convergence_results)
    df_convergence.to_csv('convergence_evHV_evGEV_5e6.csv', index=False)
    print("\nResults saved to 'convergence_evHV_evGEV_5e6.csv'")

    print("\n" + "="*60)
    print("CONVERGENCE TESTING RESULTS")
    print("="*60)
    print(df_convergence.head(15)) # Showing head to avoid huge console prints

    # ------------------------------------------------------------------
    # PLOT CONVERGENCE TESTING
    # ------------------------------------------------------------------
    def plot_convergence(df, heights_to_plot=[0, 3, 9, 14]):
        metrics = ['otal_cost', 'bcr', 'reliability']
        titles = ['Total Cost ($)', 'Benefit-Cost Ratio', 'Lifetime Reliability']
        
        for height in heights_to_plot:
            data_subset = df[df['dh'] == height]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Convergence Analysis for Heightening Strategy dh = {height}ft', fontsize=16)
            
            for idx, metric in enumerate(metrics):
                sns.stripplot(ax=axes[idx], data=data_subset, x='nsow', y=metric, 
                              jitter=0.2, alpha=0.6, palette="viridis")
                
                sns.pointplot(ax=axes[idx], data=data_subset, x='nsow', y=metric, 
                              color='black', markers='D', scale=0.5)
                
                axes[idx].set_title(titles[idx])
                axes[idx].grid(True, linestyle='--', alpha=0.7)
                
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'convergence_evGEV_evHV_5e6_{height}.png')

    plot_convergence(df_convergence)

    end = time.time()
    print(f"Runtime: {end - start} seconds")