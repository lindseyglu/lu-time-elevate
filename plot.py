# -*- coding: utf-8 -*-
"""
Filename: plot.py
Author: Lindsey Lu
Created: 2026-04-08
Version: 1.0
Description: 
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read in results csv
objs = pd.read_csv('objectives_evolve_house_val_35_unc.csv')

# Remove points between dh=(0,3]
objs_3 = objs[objs['dh'] >= 3]
# Get objectives from not heightening
objs_0 = objs[objs['dh'] == 0]

# Plot upfront cost on the x-axis, reliability on the y-axis
fig1, ax1 = plt.subplots()
ax1.plot('upfront_cost', 'reliability', 'bo', data=objs_3, label='dh > 3 ft')
ax1.plot('upfront_cost', 'reliability', 'ro', data=objs_0, label='dh = 0 ft')
ax1.set_xlabel('Upfront cost [$]')
ax1.set_ylabel('Reliability')
ax1.legend()
plt.savefig(f'upcost_reliability_pareto_evolve_35_unc')

# Plot total cost on the x-axis, reliability on the y-axis
fig2, ax2 = plt.subplots()
ax2.plot('total_cost', 'reliability', 'bo', data=objs_3, label='dh > 3 ft')
ax2.plot('total_cost', 'reliability', 'ro', data=objs_0, label='dh = 0 ft')
ax2.set_xlabel('Total cost [$]')
ax2.set_ylabel('Reliability')
ax2.legend()
plt.savefig(f'totcost_reliability_pareto_evolve_35_unc')

# Plot NPV of total costs for different heightening strategies
fig3, ax3 = plt.subplots()
ax3.set_xlim(0,14)
ax3.plot('dh', 'total_cost', 'bo', data=objs_3, label='dh > 3 ft')
ax3.plot('dh', 'total_cost', 'ro', data=objs_0, label='dh = 0 ft')
ax3.set_ylabel('Total cost [$]')
ax3.set_xlabel('Elevation height')
ax3.legend()
plt.savefig(f'height_totcost_evolve_35_unc')