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

# Set print
verbose = True

# Set house characteristics and discount rate
sqft = 1000
struc_value = 250000
del_elev = -6           # difference in house elev and BFE
life_span = 30
disc_rate = np.full(shape=(life_span,), fill_value=0.04)
bfe = 8
# Initial house elev
init_elev = bfe + del_elev
if verbose: 
    print("House parameters")
    print(f"Structure value: ${struc_value}")
    print(f"Lifetime: {life_span}")
    print(f"Initial elevation: {init_elev} ft")
    print(f"Base flood elevation: {bfe}")
    print(f"Discount rate: {disc_rate}")

# Set GEV parameters [currently using GEV from Zarekarizi et al. 2020]
mu = 19.8718901487264
sigma = 3.16814792683425
xi = 0.00515921024408503
if verbose:
    print("GEV parameters")
    print(f"location: {mu}")
    print(f"scale: {sigma}")
    print(f"shape: {xi}")

# Set elevation height strategies to evaluate
delta_h_seq = np.array(0,3,4,5,6,7,8,9,10,11,12,13,14)

# Set depth-damage function
# HAZUS DDF:
depth = np.array(-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)  # defined relative the FFE
damage_fac = np.array(0,0,4,8,12,15,20,23,28,33,37,43,48,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81)
# Upper and lower bounds for sampling uncertainty
damage_fac_ub = damage_fac + damage_fac*0.3
damage_fac_lb = damage_fac - damage_fac*0.3
damage_fac_ub[damage_fac_ub>100] = 100

## Total cost is lifetime expected damages + construction cost

# Step 1: Calculate lifetime expected damages
def lifetime_expected_damages(struc_value, init_elev, delta_h, life_span, disc_rate, mu, sigma, xi, DD_Depth, DD_Damage):
    """
    Docstring for lifetime_expected_damages
    
    :param struc_value: Description
    :param init_elev: Description
    :param delta_h: Description
    :param life_span: house lifespan
    :param disc_rate: array of discount rates for each year of the house lifespan
    :param mu: Description
    :param sigma: Description
    :param xi: Description
    :param DD_Depth: depths from the depth-damage function, defined relative to FFE
    :param DD_Damage: Description
    """
    curr_elev = init_elev + delta_h # elevation of house after being elevated
    
    # Damage value lost at each depth, which depends on house value
    damage_vals = (DD_Damage/100) * struc_value

    # Critical depths are depths where the damage factor changes. **ask**
    crit_depths = DD_Depth + curr_elev # Calculates the stage of critical depths

    # Probability that water level exceeds each critical depth in one year
    crit_probs = 1 - genextreme.cdf(x=crit_depths, c=xi, loc=mu, scale=sigma)

    # [add a block that avoids NaNs and NAs]

    # Calculate the expected annual damages (EAD)
    EADfrac = np.empty(crit_depths)
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
    
    :param delta_h: Description
    :param sqft: Description
    """
    # Cost of elevating according to CLARA are as the following:
    # 82.5/sqft (3 to 7)
    # 86.25/sqft (7 to 10)
    # 103.75/sqft (10 to 14)
  
    # There is a base cost for elevating any house as the following. For more information, see appendix A of CLARA model through the link above 
    base_cost= 10000 + 300 + 470 + 4300 + 2175 + 3500
    Hs=np.array(3,5,8.5,12,14)
    Rates=np.array(80.36,82.5,86.25,103.75,113.75)

    # The cost of elevating the house after the base cost depends on the size of the house
    # Interpolate to find the new 
    if delta_h>=3 & delta_h<=14:
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
    
    :param life_span: Description
    :param mu: Description
    :param sigma: Description
    :param xi: Description
    :param init_elev: Description
    :param delta_h: Description
    """
    curr_elev = init_elev + delta_h

    # Safety is probability of zero floods during the next n years where n is the expected lifetime of the house
    safety = genextreme.cdf(x=curr_elev, c=xi, loc=mu, scale=sigma) ** (life_span//1)
    return(safety)

## Benefit-cost ratio
# Benefit is the cost of not elevating (delta_h=0) minus the cost of elevating
# BCR is benefit divided by the construction cost

## Satisficing (robustness)
# From Zarekarizi et al. 2020: BCR > 1, reliability > 0.5, total cost/structure value < 1
def satisficing_all(bcr, reliability, total_cost, struc_val):
    """
    Docstring for satisficing_all
    
    :param bcr: Description
    :param reliability: Description
    :param total_cost: Description
    :param struc_val: Description
    """
    return np.array(bcr>1, reliability>0.5, total_cost/struc_val<1)

## Evaluate strategies
led = np.empty(len(delta_h_seq))
cc = np.empty(len(delta_h_seq))
lr = np.empty(len(delta_h_seq))
sa = np.empty(len(delta_h_seq))

# Iterate through strategies
for i in range(len(delta_h_seq)):
    if verbose: print(f"Evaluating strategy {i+1} of {len(delta_h_seq)}")
    # Step 1: lifetime expected damages
    led[i] = lifetime_expected_damages(struc_value, init_elev, delta_h_seq[i], life_span, 
                                    disc_rate, mu, sigma, xi, depth, damage_fac)
    # Step 2: construction cost
    cc[i] = construction_cost(delta_h_seq[i], sqft)
    # Step 3: reliability
    lr[i] = lifetime_reliability(life_span, mu, sigma, xi, init_elev, delta_h_seq[i])

# Step 4: total cost
tc = led+cc
# Step 5: benefit-cost ratio
bcr_cost = cc
bcr_benefit = led[0]-led
bcr = bcr_cost / bcr_benefit

# Find optimal strategy not considering uncertainty
i_min = np.argmin(tc)       # can change to maximize BCR instead of minimize TC
opt_h = delta_h_seq[i_min]  # optimal height
opt_h_led = led[i_min]      # damages at optimal height
opt_h_cc = cc[i_min]        # construction cost at opt h
opt_h_tc = tc[i_min]        # total cost at opt h
opt_h_bcr = bcr[i_min]      # benefit-cost ratio at opt h
opt_h_lr = lr[i_min]        # reliability at opt h
# Step 6: satisficing
opt_h_sa = satisficing_all(opt_h_bcr, opt_h_tc, struc_value)

if verbose:
    print(f"Optimal height without uncertainty: {opt_h}")
    print(f"Total cost: {opt_h_tc}")
    print(f"Benefit-cost ratio: {opt_h_bcr}")
    print(f"Lifetime reliability: {opt_h_lr}")
    print(f"Satisficing: {opt_h_sa}")