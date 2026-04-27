# -*- coding: utf-8 -*-
"""
Filename: extra_code.py
Author: Lindsey Lu
Created: 2026-04-27
Version: 1.0
Description: Holds extra code that was used in developing adaptive_elev.py.
This is an intermediary holder to ensure that this code can be cleaned up and remain.
It maintains that adaptive_elev is a cleaner file.
"""
## ------------------------------------------------------------------
## Evaluate deterministic strategies
## ------------------------------------------------------------------
# # Note: this section should be altered so that Mass and FEMA standards can be compared to other results
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

# ## ------------------------------------------------------------------
# ## CONVERGENCE TESTING & UNCERTAINTY ENSEMBLES
# ## ------------------------------------------------------------------

# # Set parameters for convergence testing
# delta_h_seq = np.array([0,3,9,14])
# nsow_values = [10, 100, 1000, 10000]
# iterations = 10
# rng = np.random.default_rng()

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
        
#         # --- 1. Generate Uncertainties for this iteration ---
#         dr_unc = discount_rate_unc(obs_discount, current_nsow)
#         lt_unc = lifetime_unc(current_nsow)
#         ddf_unc = depth_damage_unc(current_nsow)
#         gev_unc = gev_param_unc(current_nsow, mu_chain, sigma_chain, xi_chain)
        
#         # Allocate ensemble array and perform LHS
#         ens = np.empty((current_nsow, 255))
#         sampler = qmc.LatinHypercube(d=4)
#         sample = sampler.random(n=current_nsow)
        
#         i_sow = np.floor(sample * current_nsow).astype(int)
#         # Avoid sampling row 0 (depths) for the depth-damage function
#         i_sow[:, 3] = np.floor(sample[:, 3] * (current_nsow - 2)).astype(int) + 1
        
#         # Map parameters to ensemble matrix
#         ens[:, 0:3] = gev_unc[i_sow[:,0], :]         
#         ens[:, 3:204] = dr_unc[i_sow[:,1], :]        
#         ens[:, 204] = lt_unc[i_sow[:,2]]             
#         ens[:, 205:255] = ddf_unc[i_sow[:,3], :]     
        
#         # --- 2. Evaluate Strategies ---
#         led_ens = np.zeros((num_strat, current_nsow))
#         cc_ens = np.zeros(num_strat)
#         lr_ens = np.zeros((num_strat, current_nsow))
#         dd_depths = ddf_unc[0, :]
        
#         for i, dh in enumerate(delta_h_seq):
#             cc_ens[i] = construction_cost(dh, sqft)
#             for j in range(current_nsow):
#                 mu_sow, sigma_sow, xi_sow = ens[j, 0:3]
#                 life_sow = int(np.floor(ens[j, 204]))
#                 dr_sow = ens[j, 3 : 3 + min(life_sow, 201)]
#                 dd_damage_sow = ens[j, 205:255]
                
#                 # Use your corrected lifetime_expected_damages function here
#                 led_ens[i, j] = lifetime_expected_damages(
#                     struc_value, init_elev, dh, life_sow, 
#                     dr_sow, mu_sow, sigma_sow, xi_sow, dd_depths, dd_damage_sow
#                 )
#                 lr_ens[i, j] = lifetime_reliability(
#                     life_sow, mu_sow, sigma_sow, xi_sow, init_elev, dh
#                 )

#         # --- 3. Calculate Objectives for ALL strategies ---
#         tc_ens = led_ens + cc_ens[:, np.newaxis]
        
#         # Loop through each strategy to calculate means and satisficing
#         for i, dh in enumerate(delta_h_seq):
#             # Calculate BCR (only if dh > 0)
#             if dh > 0:
#                 bcr_array = (led_ens[0, :] - led_ens[i, :]) / cc_ens[i]
#                 mean_bcr = np.mean(bcr_array)
#             else:
#                 bcr_array = np.zeros(current_nsow)
#                 mean_bcr = np.nan # BCR isn't applicable for dh=0

#             mean_tc = np.mean(tc_ens[i, :])
#             mean_rel = np.mean(lr_ens[i, :])
            
#             # Robustness / Satisficing Score for THIS strategy
#             robustness_mask = (
#                 (bcr_array > 1) & 
#                 (lr_ens[i, :] > 0.5) & 
#                 ((tc_ens[i, :] / struc_value) < 1)
#             )
#             robustness_score = np.mean(robustness_mask) * 100
            
#             # --- 4. Store Results for every height ---
#             convergence_results.append({
#                 'nsow': current_nsow,
#                 'iteration': it + 1,
#                 'dh': dh,
#                 'total_cost': mean_tc,
#                 'bcr': mean_bcr,
#                 'reliability': mean_rel,
#                 'satisficing': robustness_score
#             })

# # Convert to DataFrame
# df_convergence = pd.DataFrame(convergence_results)
# df_convergence.to_csv('convergence_data_full.csv', index=False)
# print("\nResults saved to 'convergence_data_full.csv'")

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
#         plt.savefig(f'convergence_dh_{height}')

# # Run the plotting function
# plot_convergence(df_convergence)
