# -*- coding: utf-8 -*-
"""
Filename: plot.py
Author: Lindsey Lu
Created: 2026-04-08
Version: 1.0
Description: Plot Pareto fronts and single objective optimization 
for different methods of evaluating home elevation height
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import genextreme

# # Read in results csv
# objs = pd.read_csv('objectives.csv')

# # Remove points between dh=(0,3]
# objs_3 = objs[objs['dh'] >= 3]
# # Get objectives from not heightening
# objs_0 = objs[objs['dh'] == 0]

# # Plot upfront cost on the x-axis, reliability on the y-axis
# fig1, ax1 = plt.subplots()
# ax1.plot('upfront_cost', 'reliability', 'bo', data=objs_3, label='dh > 3 ft')
# ax1.plot('upfront_cost', 'reliability', 'ro', data=objs_0, label='dh = 0 ft')
# ax1.set_xlabel('Upfront cost [$]')
# ax1.set_ylabel('Reliability')
# ax1.legend()
# plt.savefig(f'upcost_reliability_pareto')

# # Plot total cost on the x-axis, reliability on the y-axis
# fig2, ax2 = plt.subplots()
# ax2.plot('total_cost', 'reliability', 'bo', data=objs_3, label='dh > 3 ft')
# ax2.plot('total_cost', 'reliability', 'ro', data=objs_0, label='dh = 0 ft')
# ax2.set_xlabel('Total cost [$]')
# ax2.set_ylabel('Reliability')
# ax2.legend()
# plt.savefig(f'totcost_reliability_pareto')

# # Plot NPV of total costs for different heightening strategies
# fig3, ax3 = plt.subplots()
# ax3.set_xlim(0,14)
# ax3.plot('dh', 'total_cost', 'bo', data=objs_3, label='dh > 3 ft')
# ax3.plot('dh', 'total_cost', 'ro', data=objs_0, label='dh = 0 ft')
# ax3.set_ylabel('Total cost [$]')
# ax3.set_xlabel('Elevation height')
# ax3.legend()
# plt.savefig(f'height_totcost')

## ------------------------------------------------------------------
## PLOT HEIGHTS AT DIFFERENT TIME INTERVALS
## ------------------------------------------------------------------

# # 1. Load and merge the data
# # File definitions with descriptive labels
# files = {
#     'data/objectives_evHV_evGEV_0.csv': 'Time-indexed house value and GEV parameters',
#     'data/objectives_evHV.csv': 'Time-indexed house value',
#     'data/objectives_evGEV.csv': 'Time-indexed GEV parameters',
#     'data/objectives.csv': 'No time-indexed uncertainties'
# }
# num_scen = 4

# dfs = []
# for f, label in files.items():
#     df = pd.read_csv(f)
#     df['Scenario'] = label
#     dfs.append(df)

# df_all = pd.concat(dfs)

# # 2. Filter: dh == 0 OR dh >= 3
# df_filtered = df_all[(df_all['dh'] == 0) | (df_all['dh'] >= 3)].copy()
# df_filtered['Point Type'] = np.where(df_filtered['dh'] == 0, 'Baseline (dh=0)', 'Elevated (dh>=3)')

# # 3. Define the visual distinction for the baseline (dh=0)
# df_filtered['Point Type'] = np.where(df_filtered['dh'] == 0, 'Baseline (dh=0)', 'Elevated (dh>=3)')

# sns.set_theme(style="whitegrid")
# palette = sns.color_palette("husl", num_scen)

# # GRAPH 1: dh vs total_cost
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_filtered, x='dh', y='total_cost', hue='Scenario', 
#                 style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
#                 s=100, palette=palette)
# plt.title('Total Cost vs Heightening Strategy')
# plt.xlabel('Elevation Height [ft]')
# plt.ylabel('Total Cost [$]')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('figures/total_cost_vs_dh_comp.png')

# # GRAPH 2: upfront_cost vs reliability
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_filtered, x='upfront_cost', y='reliability', hue='Scenario', 
#                 style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
#                 s=100, palette=palette)
# plt.title('Reliability vs Upfront Construction Cost')
# plt.xlabel('Upfront Cost [$]')
# plt.ylabel('Lifetime Reliability')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('figures/reliability_vs_upfront_comp.png')

# # GRAPH 3: total_cost vs reliability
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_filtered, x='total_cost', y='reliability', hue='Scenario', 
#                 style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
#                 s=100, palette=palette)
# plt.title('Reliability vs Total Cost')
# plt.xlabel('Total Cost [$]')
# plt.ylabel('Lifetime Reliability')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('figures/reliability_vs_total_cost_comp.png')

## ------------------------------------------------------------------
## PLOT GEV FUNCTION
## ------------------------------------------------------------------

t = 30
b1 = 0.02
b2 = 0.001

# Define parameters
mus = pd.read_csv('mu_chain.csv')
loc = np.mean(mus)
loc_ev = loc + t*b1

sigmas = pd.read_csv('sigma_chain.csv')
scale = np.mean(sigmas)
scale_ev = np.exp(np.log(scale) + t*b2)

xis = pd.read_csv('xi_chain.csv')
shape = -np.mean(xis)

# Generate data points
x = np.linspace(genextreme.ppf(0.01, shape, loc, scale),
                genextreme.ppf(0.99, shape, loc, scale), 100)
x_ev = np.linspace(genextreme.ppf(0.01, shape, loc_ev, scale_ev),
                   genextreme.ppf(0.99, shape, loc_ev, scale_ev), 100)

# Calculate PDF and CDF
pdf = genextreme.pdf(x, shape, loc, scale)
pdf_ev = genextreme.pdf(x, shape, loc_ev, scale_ev)

# Create the plot
fig, ax1 = plt.subplots()

ax1.plot(x, pdf, 'r-', label='GEV initial')
ax1.plot(x_ev, pdf_ev, 'b-', label=f'GEV after {t} years')
ax1.set_ylabel('Density')
ax1.tick_params(axis='y')

plt.legend()
plt.title(f'Initial GEV Distribution (shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f})')
plt.show()