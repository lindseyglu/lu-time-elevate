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
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

# 1. Load the unified data file
file_path = '/Volumes/keller-lab/Lindsey_Lu/single_house/data/objectives_coeff2.csv'
df_all = pd.read_csv(file_path)

# Chronologically sort by year so that legends and palettes align perfectly
df_all = df_all.sort_values(by='yr_elev')

# Dynamically construct scenario strings from the single column layout
df_all['Strategy'] = 'Year ' + df_all['yr_elev'].astype(str)
unique_scenarios = df_all['Strategy'].unique()
num_scen = len(unique_scenarios)

# Create a column for the total_cost in millions (M)
# This is for clearer axes when plotting total_cost
df_all['total_cost_M'] = df_all['total_cost']/1e6
df_all['upfront_cost_k'] = df_all['upfront_cost']/1e3

# 2. Filter: dh >= 3
df_filtered = df_all[(df_all['dh'] >= 3)].copy()
df_0 = df_all[(df_all['dh'] == 0)].copy()

# Sort the dataframe by scenario, and then by cost in ascending order
df_sort = df_all.sort_values(by=['Strategy', 'total_cost'], ascending=[True, True])
# Drop duplicates based on the 'Strategy' column, keeping only the first occurrence
# Since it's sorted by cost, the first occurrence is guaranteed to be the minimum cost
df_min_cost = df_sort.drop_duplicates(subset=['Strategy'], keep='first')

# Set up plotting style and palette dynamically based on the numbers of years evaluated
sns.set_theme(style="ticks")
palette = ['#785EF0', '#59d27e', '#e6616b']

# GRAPH 1: dh vs total_cost
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_filtered, x='dh', y='total_cost_M', hue='Strategy', 
             palette=palette)
rect = patches.Rectangle(xy=(0,0), width=3, height=10, facecolor='lightgrey')
plt.gca().add_patch(rect)
sns.scatterplot(data=df_min_cost, x='dh', y='total_cost_M', hue='Strategy',
                palette=palette, s = 50, legend=False, clip_on=False, 
                marker='s', zorder=10)
sns.scatterplot(data=df_0, x='dh', y='total_cost_M', color='red',
                marker = 'x', s = 100, clip_on=False, zorder=10, 
                label='No heightening')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}M'))
plt.scatter([], [], color='black', marker='s', label='Minimum')
plt.xlabel('Elevation Height [ft]', fontsize=16)
plt.ylabel('Total Cost [$]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,14)
plt.ylim(0,5.5)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('figures/total_cost_vs_dh_coeff2.png', dpi=300)

# Read in csv file
# This file has yrs 0 to 20, evaluated at dh=14 (where min cost occurs)
df_contyrs = pd.read_csv('data/objectives_contyrs_coeff2.csv')
df_contyrs['total_cost_M'] = df_contyrs['total_cost']/1e6

df_contyrs0 = df_contyrs[df_contyrs['yr_elev']==0]

# GRAPH 2: year of elevation vs minimum cost
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_contyrs, x='yr_elev', y='total_cost_M',
             label=r'$\Delta$h = 14', color='#785EF0')
sns.scatterplot(data=df_contyrs0, x='yr_elev', y='total_cost_M',
                s=80, clip_on=False, zorder=10, label='Minimum',
                color='#785EF0', marker='s')
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}M'))
plt.xlabel('Year of Elevation', fontsize=16)
plt.ylabel('Total Cost [$]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0,20)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('figures/total_cost_vs_year_coeff2.png', dpi=300)

# GRAPH 3: upfront_cost vs reliability
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_filtered, x='upfront_cost_k', y='reliability', 
             hue='Strategy', palette=palette, linewidth=2.5)
plt.scatter(x=0, y=1, label='Ideal point', color='goldenrod', marker='*',
            zorder=10, clip_on=False, s=200)
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}k'))
plt.xlabel('Upfront Cost [$]', fontsize=16)
plt.ylabel('Lifetime Reliability', fontsize=16)
plt.xlim(0,300)
plt.ylim(0,1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('figures/reliability_vs_upfront_coeff2.png', dpi=300)

# GRAPH 4: total_cost vs reliability
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_filtered, x='total_cost_M', y='reliability', 
             hue='Strategy', palette=palette, linewidth=2.5)
plt.scatter(x=0, y=1, label='Ideal point', color='goldenrod', marker='*',
            zorder=10, clip_on=False, s=200)
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}M'))
plt.xlabel('Total Cost [$]', fontsize=16)
plt.ylabel('Lifetime Reliability', fontsize=16)
plt.xlim(0,3.5)
plt.ylim(0,1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('figures/reliability_vs_total_coeff2.png', dpi=300)

# # GRAPH 4: dh vs upfront_cost
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_filtered, x='dh', y='upfront_cost', hue='Scenario', 
#                 style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
#                 s=100, palette=palette)
# plt.title('Upfront Cost vs Heightening Strategy')
# plt.xlabel('Elevation Height [ft]')
# plt.ylabel('Upfront Cost [$]')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('figures/upfront_cost_vs_dh_coeff2.png')

# # GRAPH 5: dh vs damages
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_filtered, x='dh', y='damages', hue='Scenario', 
#                 style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
#                 s=100, palette=palette)
# plt.title('Lifetime Damages vs Heightening Strategy')
# plt.xlabel('Elevation Height [ft]')
# plt.ylabel('Lifetime Damages [$]')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('figures/damages_vs_dh_coeff2.png')

# # GRAPH 6: upfront vs damages
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_filtered, x='upfront_cost', y='damages', hue='Scenario', 
#                 style='Point Type', markers={'Baseline (dh=0)': 'X', 'Elevated (dh>=3)': 'o'}, 
#                 s=100, palette=palette)
# plt.title('Upfront Cost vs Lifetime Damages')
# plt.xlabel('Upfront Cost [$]')
# plt.ylabel('Lifetime Damages [$]')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('figures/upcost_vs_damages_coeff2.png')