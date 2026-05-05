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

# 1. Load and merge the data
# File definitions with descriptive labels
files = {
    'objectives_yr0.csv': 'Year 0',
    'objectives_yr10.csv': 'Year 10',
    'objectives_yr20.csv': 'Year 20',
    'objectives_yr10_hv3.csv': 'Year 10 (3% Appreciation)'
}

dfs = []
for f, label in files.items():
    df = pd.read_csv(f)
    df['Scenario'] = label
    dfs.append(df)

df_all = pd.concat(dfs)

# 2. Filter: dh == 0 OR dh >= 3
df_filtered = df_all[(df_all['dh'] == 0) | (df_all['dh'] >= 3)].copy()
df_filtered['Point Type'] = np.where(df_filtered['dh'] == 0, 'Baseline (dh=0)', 'Elevated (dh>=3)')

# 3. Define the visual distinction for the baseline (dh=0)
df_filtered['Point Type'] = np.where(df_filtered['dh'] == 0, 'Baseline (dh=0)', 'Elevated (dh>=3)')

sns.set_theme(style="whitegrid")
palette = sns.color_palette("husl", 4)

# GRAPH 1: dh vs total_cost
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='dh', y='total_cost', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Total Cost vs Heightening Strategy')
plt.xlabel('Elevation Height [ft]')
plt.ylabel('Total Cost [$]')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('total_cost_vs_dh.png')

# GRAPH 2: upfront_cost vs reliability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='upfront_cost', y='reliability', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Reliability vs Upfront Construction Cost')
plt.xlabel('Upfront Cost [$]')
plt.ylabel('Lifetime Reliability')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('reliability_vs_upfront.png')

# GRAPH 3: total_cost vs reliability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='total_cost', y='reliability', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Reliability vs Total Cost')
plt.xlabel('Total Cost [$]')
plt.ylabel('Lifetime Reliability')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('reliability_vs_total_cost.png')