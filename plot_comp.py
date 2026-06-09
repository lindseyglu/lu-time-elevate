# -*- coding: utf-8 -*-
"""
Filename: plot_comp.py
Author: Lindsey Lu
Created: 2026-04-08
Version: 1.0
Description: Plot comparisons with additional evolving parameters.
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from scipy.stats import genextreme

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
plt.savefig('figures/reliability_vs_upfront_comp.png')

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
plt.savefig('figures/reliability_vs_total_cost_comp.png')

## ------------------------------------------------------------------
## PLOT DIFFERENT B1 AND B2 PARAMETERS
## ------------------------------------------------------------------

# 1. Load and merge the data
# File definitions with values for coefficients
files = {
    # '/Volumes/keller-lab/Lindsey_Lu/single_house/data/objectives_coeff0.csv': r'$\beta$1 = 0.005, $\beta$2 = 0.0005',
    '/Volumes/keller-lab/Lindsey_Lu/single_house/data/objectives_evGEV_coeff1.csv': r'$\beta$1 = 0.01, $\beta$2 = 0.001',
    '/Volumes/keller-lab/Lindsey_Lu/single_house/data/objectives_evGEV_coeff2.csv': r'$\beta$1 = 0.02, $\beta$2 = 0.003',
    '/Volumes/keller-lab/Lindsey_Lu/single_house/data/objectives_evGEV_coeff3.csv': r'$\beta$1 = 0.03, $\beta$2 = 0.005',
    '/Volumes/keller-lab/Lindsey_Lu/single_house/data/objectives_stat.csv': 'Stationary GEV'
}
num_scen = 5

dfs = []
for f, label in files.items():
    df = pd.read_csv(f)
    df['Scenario'] = label
    dfs.append(df)

df_all = pd.concat(dfs)
df_all['total_cost_k'] = df_all['total_cost']/1e3

# 2. Filter: dh >= 3 AND dh <=14
df_filtered = df_all[(df_all['dh'] >= 3) & (df_all['dh'] <= 14)].copy()
df_h0 = df_all[df_all['dh'] == 0].copy()

# Separate into year 0, 10, and 20
df_0 = df_filtered[df_filtered['yr_elev'] == 0]
df_h0_0 = df_h0[df_h0['yr_elev'] == 0]
df_10 = df_filtered[df_filtered['yr_elev'] == 10]
df_20 = df_filtered[df_filtered['yr_elev'] == 20]

# Sort the dataframe by scenario, and then by cost in ascending order
df_min_costs0 = df_0.sort_values(by=['Scenario', 'total_cost'], ascending=[True, True])
df_min_costs10 = df_10.sort_values(by=['Scenario', 'total_cost'], ascending=[True, True])
df_min_costs20 = df_20.sort_values(by=['Scenario', 'total_cost'], ascending=[True, True])

# Drop duplicates based on the 'Scenario' column, keeping only the first occurrence
# Since it's sorted by cost, the first occurrence is guaranteed to be the minimum cost
df_min_costs0 = df_min_costs0.drop_duplicates(subset=['Scenario'], keep='first')
df_min_costs10 = df_min_costs10.drop_duplicates(subset=['Scenario'], keep='first')
df_min_costs20 = df_min_costs20.drop_duplicates(subset=['Scenario'], keep='first')

print(df_min_costs0)

# Set theme
sns.set_theme(style="ticks")
palette = ['#648FFF', '#DC267F', '#FFB000', 'k']

# GRAPH 1: Plot year 0 for the three different scenarios
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_0, x='dh', y='total_cost_k', hue='Scenario', 
             palette=palette)
rect = patches.Rectangle(xy=(0,0), width=3, height=500, facecolor='lightgrey')
plt.gca().add_patch(rect)
sns.scatterplot(data=df_min_costs0, x='dh', y='total_cost_k', hue='Scenario',
                palette=palette, legend=False, clip_on=False, zorder=10, marker='s')
sns.scatterplot(data=df_h0_0, x='dh', y='total_cost_k', hue='Scenario',
                palette=palette, legend=False, 
                clip_on=False, zorder=10)
plt.scatter([], [], color='black', label='Minimum', marker='s')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}k'))
plt.xlabel('Elevation Height [ft]', fontsize=16)
plt.ylabel('Total Cost [$]', fontsize=16)
plt.legend(fontsize=14)
plt.xlim(0,14)
plt.ylim(0,500)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('figures/total_cost_vs_dh_coeffs_noHV.png', dpi=300)