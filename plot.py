# -*- coding: utf-8 -*-
"""
Filename: plot.py
Author: Lindsey Lu
Created: 2026-04-08
Version: 1.0
Description: Plot the GEV function after 30 years.
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
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
## PLOT DIFFERENT EVOLVING PARAMETERS
## ------------------------------------------------------------------

# 1. Load and merge the data
# File definitions with descriptive labels
# NOTE: need to change/filter for the just year 0 in df1
files = {
    'data/objectives_stat.csv': 'Stationary GEV, static house value',
    'data/objectives_evGEV_coeff2.csv': 'Nonstationary GEV, static house value',
    'data/objectives_evHVonly.csv': 'Stationary GEV, increasing house value',
    'data/objectives_coeff2.csv': 'Nonstationary GEV, increasing house value'
}
num_scen = 4

dfs = []
sorted_dfs = []
for f, label in files.items():
    df = pd.read_csv(f)
    df['Scenario'] = label
    filtered_df = df[df['yr_elev'] == 0]    # ensure only elevation @now is considered
    # NOTE: insert sorted dfs here
    sorted_df = filtered_df.sort_values(by=['total_cost'], ascending=[True])
    dfs.append(filtered_df)
    sorted_dfs.append(sorted_df)

df_all = pd.concat(dfs)
df_all['total_cost_M'] = df_all['total_cost']/1e6

# 2. Filter: dh == 0 OR dh >= 3
df_filtered = df_all[(df_all['dh'] >= 3)].copy()
df_h0 = df_all[(df_all['dh'] == 0)].copy()

# --- GET MIN COST ---
# 1. Concatenate the list of sorted dataframes into ONE big DataFrame
df_all_sorted = pd.concat(sorted_dfs)

# 2. Scale the cost for the minimum points so it matches your Y-axis scale
df_all_sorted['total_cost_M'] = df_all_sorted['total_cost'] / 1e6

# 3. Now drop duplicates safely on the combined DataFrame
df_min_costs = df_all_sorted.drop_duplicates(subset=['Scenario'], keep='first')
# --------------------

sns.set_theme(style="ticks")
palette = ['k', '#DC267F', '#FE6100', '#785EF0']

# GRAPH 1: dh vs total_cost
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_filtered, x='dh', y='total_cost_M', hue='Scenario', 
             palette=palette, linewidth=1.5)
rect = patches.Rectangle(xy=(0,0), width=3, height=10, facecolor='lightgrey')
plt.gca().add_patch(rect)
sns.scatterplot(data=df_h0, x='dh', y='total_cost_M', hue='Scenario',
                palette=palette, legend=False, clip_on=False, zorder=10)
sns.scatterplot(data=df_min_costs, x='dh', y='total_cost_M', hue='Scenario',
                palette=palette, legend=False, clip_on=False, zorder=10, 
                marker='s', s=60)
plt.scatter([], [], color='black', label='Minimum', marker='s')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}M'))
plt.legend(fontsize=14)
plt.xlim(0,14)
plt.ylim(0,5.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Elevation Height [ft]', fontsize=16)
plt.ylabel('Total Cost [$]', fontsize=16)
plt.tight_layout()
plt.savefig('figures/total_cost_vs_dh_comp.png', dpi=300)

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

# t1 = 50
# t2 = 100
# b1 = np.array([0.01, 0.02, 0.03])
# b2 = np.array([0.001, 0.003, 0.005])

# # Define parameters
# mus = pd.read_csv('mu_chain.csv')
# loc = np.mean(mus)
# loc_30 = loc + t1*b1
# loc_100 = loc + t2*b1

# sigmas = pd.read_csv('sigma_chain.csv')
# scale = np.mean(sigmas)
# scale_30 = np.exp(np.log(scale) + t1*b2)
# scale_100 = np.exp(np.log(scale) + t2*b2)

# xis = pd.read_csv('xi_chain.csv')
# shape = -np.mean(xis)

# # Generate data points
# x = np.linspace(genextreme.ppf(0.01, shape, loc, scale),
#                 genextreme.ppf(0.99, shape, loc, scale), 100)
# x_30 = np.linspace(genextreme.ppf(0.01, shape, loc_30, scale_30),
#                    genextreme.ppf(0.99, shape, loc_30, scale_30), 100)

# print(x_30.shape)
# # x_100 = np.linspace(genextreme.ppf(0.01, shape, loc_100, scale_100),
# #                     genextreme.ppf(0.99, shape, loc_100, scale_100), 100)

# # Calculate PDF and CDF
# pdf = genextreme.pdf(x, shape, loc, scale)
# # pdf_30 = genextreme.pdf(x, shape, loc_30, scale_30)
# # pdf_100 = genextreme.pdf(x, shape, loc_100, scale_100)

# # Create the plot
# fig, ax1 = plt.subplots()

# # Plot baseline
# ax1.plot(x, pdf, 'k-', label='Stationary GEV')

# # Loop through each scenario (0, 1, 2) to fix indexing dynamically
# colors = ['#648FFF', '#DC267F', '#FFB000']
# for i in range(len(b1)):
#     # Calculate the PDF dynamically for column i
#     pdf_scenario = genextreme.pdf(x_30[:, i], shape, loc_30[i], scale_30[i])
    
#     # Plot column i of x_30 against its corresponding pdf
#     ax1.plot(x_30[:, i], pdf_scenario, color=colors[i], linestyle='-',
#              label=rf'$\beta_1$={b1[i]}, $\beta_2$={b2[i]}')

# ax1.set_ylabel('Density')
# ax1.set_xlabel('Annual Maximum Water Level (ft)')
# ax1.legend()
# ax1.tick_params(axis='y')

# plt.savefig('figures/nonstationary_gev.png', dpi=300)