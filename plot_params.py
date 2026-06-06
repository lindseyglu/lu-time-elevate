# -*- coding: utf-8 -*-
"""
Filename: plot_params.py
Author: Lindsey Lu
Created: 2026-06-03
Version: 1.0
Description: Plot the weibull function representing house value.
             Plot depth-damage functions.
             Plot CLARA upfront cost.
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from scipy.stats import weibull_min

# HOUSE LIFETIME
yrs = np.linspace(start=0, stop=200, num=400)
pdf = weibull_min.pdf(yrs, c=2.8, scale=73.5)

# Create the plot
fig, ax1 = plt.subplots()

# Plot baseline
ax1.plot(yrs, pdf, 'k-', label='shape = 2.8, scale = 73.5')

ax1.set_ylabel('Density', fontsize=12)
ax1.set_xlabel('House Lifetime [yr]', fontsize=12)
ax1.legend()
ax1.tick_params(axis='y')

plt.savefig('figures/lifetime_weibull.png', dpi=300)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# DEPTH-DAMAGE FUNCTION
# Define European Commission's depth-damage function
depth1 = np.array([0, 1.64, 3.28, 4.92, 6.56, 9.84, 13.12, 16.40])
damage_fac1 = np.array([0.20, 0.44, 0.58, 0.68, 0.78, 0.85, 0.92, 0.96])*100

# Define HAZUS depth-damage function
depth2 = np.array([-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
damage_fac2 = np.array([0,0,4,8,12,15,20,23,28,33,37,43,48,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81])

# Create the plot
fig, ax1 = plt.subplots()

# Plot baseline lines
ax1.plot(depth1, damage_fac1, 'b-', label='European Commission')
ax1.plot(depth2, damage_fac2, 'g-', label='HAZUS')

# Add translucent bands (+/- 30%)
# alpha=0.2 controls the translucency, np.clip keeps values between 0 and 100
ax1.fill_between(depth1, 
                 np.clip(damage_fac1 * 0.70, 0, 100), 
                 np.clip(damage_fac1 * 1.30, 0, 100), 
                 color='blue', alpha=0.2)

ax1.fill_between(depth2, 
                 np.clip(damage_fac2 * 0.70, 0, 100), 
                 np.clip(damage_fac2 * 1.30, 0, 100), 
                 color='green', alpha=0.2)

ax1.set_ylabel('Damage Factor [%]', fontsize=12)
ax1.set_xlabel('Depth [ft]', fontsize=12)
ax1.set_ylim(0,100)
ax1.set_xlim(-4,24)
ax1.legend()
ax1.tick_params(axis='y')

plt.tight_layout()
plt.savefig('figures/depth_damage.png', dpi=300)

# CLARA UPFRONT COST
base_cost = 10000 + 300 + 470 + 4300 + 2175 + 3500
Hs = np.array([3, 5, 8.5, 12, 14])
Rates = np.array([80.36, 82.5, 86.25, 103.75, 113.75])
raise_cost = (Rates*1500) + base_cost
raise_cost_k = raise_cost/1000

# Create the plot
fig, ax1 = plt.subplots()

# Plot baseline lines
ax1.plot(Hs, raise_cost_k, 'k-', label='CLARA')
ax1.set_ylabel('Construction Cost [$]', fontsize=12)
ax1.set_xlabel('Elevation Height [ft]', fontsize=12)
rect = patches.Rectangle(xy=(0,0), width=3, height=1000, facecolor='lightgrey')
ax1.add_patch(rect)
ax1.set_xlim(0,14)
ax1.set_ylim(125,200)
ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}k'))
ax1.legend()
ax1.tick_params(axis='y')

plt.tight_layout()
plt.savefig('figures/upfront_cost.png', dpi=300)