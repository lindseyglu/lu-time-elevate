# -*- coding: utf-8 -*-
"""
Filename: plot.py
Author: Lindsey Lu
Created: 2026-04-08
Version: 2.0
Description: Plot Pareto fronts and single objective optimization 
             for home elevation height and implementation timing strategies
             using unified multi-dimensional simulation outputs.
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the unified data file
file_path = '/Volumes/keller-lab/Lindsey_Lu/single_house/data/objectives_5e6_3-20.csv'
df_all = pd.read_csv(file_path)

# Chronologically sort by year so that legends and palettes align perfectly
df_all = df_all.sort_values(by='yr_elev')

# Dynamically construct scenario strings from the single column layout
df_all['Scenario'] = 'Year ' + df_all['yr_elev'].astype(str)
unique_scenarios = df_all['Scenario'].unique()
num_scen = len(unique_scenarios)

# 2. Filter: dh == 0 OR dh >= 3
df_filtered = df_all[(df_all['dh'] >= 3)].copy()
df_filtered['Point Type'] = np.where(df_filtered['dh'] == 0, 'Baseline (dh=0)', 'Elevated (dh>=3)')

# Set up plotting style and palette dynamically based on the numbers of years evaluated
sns.set_theme(style="whitegrid")
palette = sns.color_palette("husl", num_scen)

# GRAPH 1: dh vs total_cost
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='dh', y='total_cost', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Total Cost vs Heightening Strategy')
plt.xlabel('Elevation Height [ft]')
plt.ylabel('Total Cost [$]')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/total_cost_vs_dh_multyrs_5e6_20.png')

# GRAPH 2: upfront_cost vs reliability
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='upfront_cost', y='reliability', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Reliability vs Upfront Construction Cost')
plt.xlabel('Upfront Cost [$]')
plt.ylabel('Lifetime Reliability')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/reliability_vs_upfront_multyrs_5e6_20.png')

# GRAPH 3: total_cost vs reliability
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='total_cost', y='reliability', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Reliability vs Total Cost')
plt.xlabel('Total Cost [$]')
plt.ylabel('Lifetime Reliability')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/reliability_vs_total_cost_multyrs_5e6_20.png')

# GRAPH 4: dh vs upfront_cost
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='dh', y='upfront_cost', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Upfront Cost vs Heightening Strategy')
plt.xlabel('Elevation Height [ft]')
plt.ylabel('Upfront Cost [$]')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/upfront_cost_vs_dh_multyrs_5e6_20.png')

# GRAPH 5: dh vs damages
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='dh', y='damages', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Lifetime Damages vs Heightening Strategy')
plt.xlabel('Elevation Height [ft]')
plt.ylabel('Lifetime Damages [$]')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/damages_vs_dh_multyrs_5e6_20.png')

# GRAPH 6: upfront vs damages
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='upfront_cost', y='damages', hue='Scenario', 
                style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
                s=100, palette=palette)
plt.title('Upfront Cost vs Lifetime Damages')
plt.xlabel('Upfront Cost [$]')
plt.ylabel('Lifetime Damages [$]')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('figures/upcost_vs_damages_multyrs_5e6_20.png')