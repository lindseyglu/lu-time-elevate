# -*- coding: utf-8 -*-
"""
Filename: sensitivity.py
Author: Lindsey Lu
Created: 2026-05-14
Version: 2.0
Description: Condicts Sobol sensitivity analysis for elevation at 
             year 0 with the included uncertainties of evolving
             house value and GEV coefficients. Uses SAlib.
"""

# import libraries
import pandas as pd
import numpy as np
from scipy.stats import genextreme
from scipy.stats import weibull_min
from scipy.stats import uniform
from scipy.stats import qmc             # for Latin Hypercube Sampling
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')                   # Use the 'Agg' backend to avoid the Qt/DBus error while plotting
import matplotlib.pyplot as plt
import seaborn as sns
import time
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import networkx as nx

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
# NOTE: need to determine if the kde should be bounded by mu_min and mu_max
# depth-damage error is sampled using uniform dist from -30 to +30
# discount rate index chooses a specific discount rate trajectory 
# (mean-reverting-with-trend model) from 1 to 10,000
# house lifetime is sampled using a Weibull dist (shape=2.8, scale=73.5)
# Sample house value and evolving GEV coefficients using uniform dist

# For parameters that are not a continuous uniform dist, sample from [0,1]
# and then transform into the desired distribution
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
               [-0.01, 0.1],
               [-0.003, 0.005]]
}

# Generates N*(2D+2) samples where N=1024
param_u = sobol_sample.sample(problem, 1024)

# Transform parameters
def transform_parameters(matrix_u, mu_kde, sigma_kde, xi_kde):
    # Create a copy to hold physical values
    physical_matrix = np.empty_like(matrix_u)
    
    # Pass through the already-correct uniform columns
    physical_matrix[:, 5:9] = matrix_u[:, 5:9]
    
    # 1. Transform KDEs using a high-density percentile map of your empirical data
    # (We draw a large pool from the KDE to act as our lookup distribution)
    pool_size = 500000
    physical_matrix[:, 0] = np.percentile(mu_kde.resample(pool_size), matrix_u[:, 0] * 100)
    physical_matrix[:, 1] = np.percentile(sigma_kde.resample(pool_size), matrix_u[:, 1] * 100)
    physical_matrix[:, 2] = np.percentile(xi_kde.resample(pool_size), matrix_u[:, 2] * 100)

    # 2. Transform discrete uniform
    # Scale the [0, 1] float to span from 0.5 to 10000.5
    continuous_dr = 0.5 + (matrix_u[:, 3] * 10000.0)
    # Round to the nearest integer and cast to int to get an exact discrete span [1, 10000]
    physical_matrix[:, 3] = np.round(continuous_dr).astype(int)

    # 3. Transform Weibull using its Percent Point Function (PPF / Inverse CDF)
    physical_matrix[:, 4] = weibull_min.ppf(matrix_u[:, 4], c=2.8, scale=73.5)
    
    return physical_matrix

# Get true parameters
param_values = transform_parameters(param_u, mu_kde, sigma_kde, xi_kde)

# Run model: here we are interested in the lifetime expected damages
# Define functions
def fast_gev_cdf(x, c, loc, scale):
    # Standard GEV formula using NumPy vectorized operations
    # Note: Using your variable 'xi' directly as the shape param
    z = (x - loc) / scale
    # To handle the 1 + xi*z > 0 constraint
    inner = 1 + c * z
    return np.exp(-np.power(np.maximum(inner, 1e-10), -1/c))

# Calculate lifetime expected damages
def lifetime_expected_damages(struc_value, init_elev, delta_h, life_span, disc_rate, mu, sigma, xi, DD_Depth, DD_Damage, yr_elev):
    """
    Docstring for lifetime_expected_damages
    
    :param struc_value: structure value (shape: (life_span,))
    :param init_elev: initial house elevation (scalar)
    :param delta_h: height the house is being raised by (scalar)
    :param life_span: house lifespan (scalar)
    :param disc_rate: array of discount rates for each year of the house lifespan (shape: (life_span,))
    :param mu: location parameter for generalized extreme value (GEV) distribution (shape: (life_span,))
    :param sigma: scale parameter for GEV (shape: (life_span,))
    :param xi: shape parameter for GEV (scalar)
    :param DD_Depth: depths from the depth-damage function, defined relative to FFE (shape: (num_depths,))
    :param DD_Damage: damage factor from the depth-damage function, defined out of 100 (shape: (num_depths,))
    """
    curr_elev = init_elev + delta_h # elevation of house after being elevated
    
    # Damage value lost at each depth, which depends on house value
    # shape: (num_depths, life_span)
    damage_vals = (DD_Damage[:, np.newaxis]/100) * struc_value[np.newaxis, :]

    # Critical depths are depths where the damage factor changes.
    # Calculates the stage of critical depths
    crit_depths_elev = DD_Depth + curr_elev              # shape: (num_depths,)
    crit_depths_init = DD_Depth + init_elev              # shape: (num_depths,)

    # Probability that water level exceeds each critical depth in one year
    # We reshape parameters to (1, life_span) to broadcast against crit_depths (num_depths, 1)
    # Resulting crit_probs shape: (num_depths, life_span)
    crit_probs_elev = 1 - fast_gev_cdf(
        x=crit_depths_elev[:,np.newaxis],               # Shape: (num_depths, 1)
        c=xi,                                           # Shape: scalar
        loc=mu[np.newaxis,:],                           # Shape: (1, life_span)
        scale=sigma[np.newaxis,:]                       # Shape: (1, life_span)
    )

    crit_probs_init = 1 - fast_gev_cdf(
        x=crit_depths_init[:,np.newaxis],               # Shape: (num_depths, 1)
        c=xi,                                           # Shape: scalar
        loc=mu[np.newaxis,:],                           # Shape: (1, life_span)
        scale=sigma[np.newaxis,:]                       # Shape: (1, life_span)
    )

    # NEEDS ALTERING? Final safety catch for floating point precision issues
    crit_probs_elev = np.nan_to_num(crit_probs_elev, nan=0.0)         # shape: (num_depths, life_span)
    crit_probs_init = np.nan_to_num(crit_probs_init, nan=0.0)         # shape: (num_depths, life_span)

    # Calculate the expected annual damages (EAD) for each year
    # Note: it is different each year depending on
    # (a) the house value, 
    # (b) whether the house is elevated that year or not,
    # (c) mu and sigma
    prob_diffs_elev = -np.diff(crit_probs_elev, axis=0, append=0)     # shape: (num_depths, life_span)
    prob_diffs_init = -np.diff(crit_probs_init, axis=0, append=0)     # shape: (num_depths, life_span)
    ead_elev = np.sum(prob_diffs_elev * damage_vals, axis=0)          # shape: (life_span)
    ead_init = np.sum(prob_diffs_init * damage_vals, axis=0)          # shape: (life_span)

    # Combine elevated and non-elevated EAD
    years = np.arange(disc_rate.shape[0])
    ead = np.where(years < yr_elev, ead_init, ead_elev)

    # Apply the year-by-year discount rate
    disc_ead = ead * disc_rate     # shape: (life_span)*(life_span)=(life_span)

    # Create a mask: True if year < house_lifetime
    lifespan_mask = years < life_span
    
    # Calculate expected damage
    # Sum across the life_span (axis 1) to get one value per SOW
    exp_dam = np.sum(disc_ead * lifespan_mask)

    return exp_dam

# Generate discount rate
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

obs_disc = pd.read_csv('discount.csv')
dr_trajectories = discount_rate_unc(obs_disc, 10000, life_span=200)

# Allocate matrix to hold output values
Y = np.zeros([param_values.shape[0]])

# Run model
for i, X in enumerate(param_values):
    mu, sigma, xi, dr_float, lt, dd, hv_rate, b1, b2 = X
    dr_i = int(dr_float)

    # Need to pull discount rate trajectory from discount rate index
    disc_rate = dr_trajectories[dr_i-1]

    # Adjust mu and sigma by b1 and b2 respectively
    t = np.arange(201)
    mu_t = mu + b1*t
    sigma_t = np.exp(np.log(sigma) + b2*t)

    # Adjust struc_value to account for hv_rate
    house_value = struc_value * (1 + hv_rate*t)

    # Calculate damage factors, adjusted for the sampled error
    damage_adj = damage_fac + damage_fac*(dd/100)
    damage_adj = np.clip(damage_adj, 0, 100)    # Ensure that damage is still between 0 and 100

    Y[i] = lifetime_expected_damages(house_value, init_elev, 0, lt, disc_rate, mu_t, sigma_t, xi, depth, damage_adj, 0)

# Perform analysis
Si = sobol_analyze.analyze(problem, Y)

# Save as csv
# Save as csv - Separate Files for Main and Interacting Effects

# 1. Save Main 1D Effects (S1 & ST)
df_main = pd.DataFrame({
    'Parameter': problem['names'],
    'S1': Si['S1'], 
    'S1_conf': Si['S1_conf'],
    'ST': Si['ST'], 
    'ST_conf': Si['ST_conf']
})
df_main.to_csv('sobol_main_effects.csv', index=False)

# 2. Flatten and Save Second-Order Interactions (S2 Matrix pairs)
if 'S2' in Si and Si['S2'] is not None:
    s2_rows = []
    p_names = problem['names']
    num_vars = len(p_names)
    
    # Loop through unique parameter pairs to flatten the 2D matrix into 1D rows
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            s2_rows.append({
                'Parameter_1': p_names[i],
                'Parameter_2': p_names[j],
                'S2': Si['S2'][i, j],
                'S2_conf': Si['S2_conf'][i, j]
            })
            
    df_s2 = pd.DataFrame(s2_rows).dropna(subset=['S2'])
    df_s2.to_csv('sobol_second_order_interactions.csv', index=False)
    print("Successfully saved 'sobol_main_effects.csv' and 'sobol_second_order_interactions.csv'")
else:
    print("Main effects saved. No second-order indices found to export.")

# =============================================================================
# REPLICATING THE RADIAL NETWORK PLOT USING ACTUAL SALIB RESULTS
# =============================================================================

# # 1. Map your problem names to clean, beautiful labels for display
# label_mapping = {
#     'mu_u': 'Location parameter',
#     'sigma_u': 'Scale parameter',
#     'xi_u': 'Shape parameter',
#     'dr_u': 'Discount rate',
#     'lt_u': 'Lifetime',
#     'dd_err': 'Depth-damage',
#     'hv_rate': 'HV rate',
#     'b1': 'b1 parameter',
#     'b2': 'b2 parameter'
# }

# # Convert names to list matching your problem dictionary order
# raw_names = problem['names']
# display_labels = [label_mapping[name] for name in raw_names]

# # 2. Convert Si arrays into structured dictionaries mapped by display names
# s1_dict = dict(zip(display_labels, Si['S1']))
# st_dict = dict(zip(display_labels, Si['ST']))

# # 3. Safely extract 2nd-order interaction indices (S2 matrix)
# s2_interactions = {}
# num_vars = len(raw_names)

# # Check if S2 matrix exists in the output (depends on SALib calc_second_order settings)
# if 'S2' in Si and Si['S2'] is not None:
#     for i in range(num_vars):
#         for j in range(i + 1, num_vars):
#             s2_val = Si['S2'][i, j]
            
#             # CRITICAL FILTER: Only plot connections with a noticeable impact (>1%)
#             # This keeps the network from turning into a messy spiderweb
#             if s2_val > 0.01:
#                 p1_display = label_mapping[raw_names[i]]
#                 p2_display = label_mapping[raw_names[j]]
#                 s2_interactions[(p1_display, p2_display)] = s2_val

# # 4. Initialize NetworkX Graph structure
# G = nx.Graph()
# for label in display_labels:
#     G.add_node(label)

# # 5. Fix layout coordinates in a perfect circular ring
# pos = nx.circular_layout(G)

# # Rotate the network layout by 90 degrees so 'Location parameter' starts at the top right
# theta = np.deg2rad(45)
# rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# for node in pos:
#     pos[node] = np.dot(pos[node], rot_matrix)

# # 6. Initialize Plot Canvas
# fig, ax = plt.subplots(figsize=(9, 9))
# ax.set_aspect('equal')
# plt.axis('off')

# # Add the soft gray grounding circle background seen in the reference PDF
# bg_circle = plt.Circle((0, 0), 1.05, color='#F0F0F0', zorder=0)
# ax.add_artist(bg_circle)

# # 7. Draw Second-Order Interaction Edges (Blue Lines)
# if s2_interactions:
#     # Scale lines linearly: maximum interaction value maps to a thickness of 8.0
#     max_s2 = max(s2_interactions.values()) if len(s2_interactions) > 0 else 1.0
#     for (u, v), s2_val in s2_interactions.items():
#         edge_width = np.interp(s2_val, [0.01, max_s2], [1.5, 8.0])
        
#         # FIXED: Removed 'zorder=1' to prevent the unexpected keyword argument error
#         nx.draw_networkx_edges(
#             G, pos, edgelist=[(u, v)], 
#             width=edge_width, edge_color='#1A237E', # Deep navy blue
#             ax=ax
#         )
# # 8. Draw Nodes Layer by Layer (Total-order ring + First-order center)
# # Dynamically determine bounds to avoid rendering microscopic or giant nodes
# all_s1 = list(s1_dict.values())
# all_st = list(st_dict.values())
# min_s1, max_s1 = min(all_s1), max(all_s1)
# min_st, max_st = min(all_st), max(all_st)

# for node in display_labels:
#     x, y = pos[node]
    
#     # Map index percentage bounds cleanly to display pixel sizing markers
#     # Safe interp protects against divide-by-zero if variance is completely uniform
#     s1_size = np.interp(s1_dict[node], [max(0, min_s1), max(0.01, max_s1)], [150, 2500])
#     st_size = np.interp(st_dict[node], [max(0, min_st), max(0.01, max_st)], [220, 3200])
    
#     # Draw Outer Black Circle (Total Order Index, ST)
#     ax.scatter(x, y, s=st_size, color='black', zorder=2)
#     # Draw White Buffer Mask Ring
#     ax.scatter(x, y, s=st_size * 0.85, color='white', zorder=3)
#     # Draw Inner Salmon Circle (First Order Index, S1)
#     ax.scatter(x, y, s=s1_size, color='#FF6B6B', zorder=4)

# # 9. Add Context Labels Text Around the Perimeter Outer Boundary
# for node, (x, y) in pos.items():
#     # Push text outwards slightly beyond the gray background circle boundary
#     radial_offset = 1.25
#     text_x = x * radial_offset
#     text_y = y * radial_offset
    
#     # Intelligently align label alignment based on position quadrant
#     ha = 'left' if x >= 0 else 'right'
#     va = 'center'
    
#     if abs(x) < 0.15:
#         ha = 'center'
#         va = 'bottom' if y > 0 else 'top'

#     ax.text(
#         text_x, text_y, node, 
#         fontsize=10, color='#212121', weight='bold',
#         horizontalalignment=ha, verticalalignment=va
#     )

# # 10. Generate Output Image
# plt.tight_layout()
# output_filename = 'SA_RadialPlot.png'
# plt.savefig(output_filename, dpi=300, bbox_inches='tight')
# print(f"Radial network sensitivity visualization successfully saved as '{output_filename}'")