# -*- coding: utf-8 -*-
"""
Filename: plot_sa.py
Author: Lindsey Lu
Created: 2026-05-24
Version: 2.0
Description: Plots sensitivity analysis first-, second-,
             and total-order in a spider plot.
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

problem = {
    'num_vars': 9,
    'names': ['mu_u', 'sigma_u', 'xi_u', 'dr_u', 'lt_u', 'dd_err', 'hv_rate', 'b1', 'b2'],
    'bounds': [[0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [-30, 30],
               [-0.025, 0.095],
               [-0.01, 0.1],
               [-0.003, 0.005]]
}

# Read in csv values
Si = pd.read_csv('sobol_main_effects.csv')
Si2 = pd.read_csv('sobol_second_order_interactions.csv')

# =============================================================================
# REPLICATING THE RADIAL NETWORK PLOT FROM R SCRIPT GUIDELINES
# =============================================================================

# 1. Define parameter groupings and metadata to match R code color/discipline logic
param_info = {
    'mu_u':      {'display_name': 'Location\nparameter',    'color': 'darkgreen', 'discipline': 'Earth sciences'},
    'sigma_u':   {'display_name': 'Scale\nparameter',       'color': 'darkgreen', 'discipline': 'Earth sciences'},
    'xi_u':      {'display_name': 'Shape\nparameter',       'color': 'darkgreen', 'discipline': 'Earth sciences'},
    'b1':        {'display_name': 'Location\ncoefficient',  'color': 'darkgreen', 'discipline': 'Earth sciences'},
    'b2':        {'display_name': 'Scale\ncoefficient',     'color': 'darkgreen', 'discipline': 'Earth sciences'},
    'dd_err':    {'display_name': 'Depth-\ndamage',         'color': 'darkred',   'discipline': 'Engineering'},
    'hv_rate':   {'display_name': 'House\nvalue rate',      'color': 'purple',    'discipline': 'Social sciences'},
    'dr_u':      {'display_name': 'Discount rate',          'color': 'purple',    'discipline': 'Social sciences'},
    'lt_u':      {'display_name': 'Lifetime',               'color': 'purple',    'discipline': 'Social sciences'}
}

# 2. Sort parameters to keep disciplines contiguous around the perimeter
ordered_keys = ['mu_u', 'sigma_u', 'xi_u', 'b1', 'b2', 'dd_err', 'hv_rate', 'dr_u', 'lt_u']
raw_names = problem['names']
num_vars = len(ordered_keys)

# Helper function for safe min-max scaling to prevent zero-division errors
def scale_value(val, val_min, val_max, out_min, out_max):
    if val_max == val_min:
        return out_min
    return out_min + ((val - val_min) / (val_max - val_min)) * (out_max - out_min)

# 3. Extract 1D indices and evaluate statistical significance (val - conf > 0)
s1_vals, st_vals = [], []
s1_sig, st_sig = [], []

for name in ordered_keys:
    idx = raw_names.index(name)
    s1_v = np.atleast_1d(Si['S1'][idx])[0]
    st_v = np.atleast_1d(Si['ST'][idx])[0]
    s1_c = np.atleast_1d(Si['S1_conf'][idx])[0]
    st_c = np.atleast_1d(Si['ST_conf'][idx])[0]
    
    s1_vals.append(s1_v)
    st_vals.append(st_v)
    # R Code rule: significant if zero is excluded from the 95% confidence interval
    s1_sig.append(1 if (s1_v - s1_c) > 0 else 0)
    st_sig.append(1 if (st_v - st_c) > 0 else 0)

s1_vals, st_vals = np.array(s1_vals), np.array(st_vals)
s1_sig, st_sig = np.array(s1_sig), np.array(st_sig)

# 4. Extract 2D Second-Order interactions and evaluate significance
s2_matrix = np.zeros((num_vars, num_vars))
s2_sig = np.zeros((num_vars, num_vars))

if 'S2' in Si2.columns:
    # Create a fast lookup dictionary using the parameter names as keys
    # We store both (p1, p2) and (p2, p1) so order doesn't matter during the lookup
    s2_lookup = {}
    for _, row in Si2.iterrows():
        p1, p2 = row['Parameter_1'], row['Parameter_2']
        s2_lookup[(p1, p2)] = (row['S2'], row['S2_conf'])
        s2_lookup[(p2, p1)] = (row['S2'], row['S2_conf'])

    # Populate the plotting matrices using our ordered keys
    for i in range(num_vars):
        for j in range(num_vars):
            if i != j:
                p1 = ordered_keys[i]
                p2 = ordered_keys[j]
                
                # Check if the pair exists in our CSV data
                if (p1, p2) in s2_lookup:
                    val, conf = s2_lookup[(p1, p2)]
                    
                    if not np.isnan(val):
                        s2_matrix[i, j] = val
                        # Significant if zero is excluded from the 95% confidence interval
                        s2_sig[i, j] = 1 if (val - conf) > 0 else 0

# 5. Set up Radial Geometry Coordinates (Matches R script settings exactly)
cent_x, cent_y = 0.0, 0.2
radi = 0.6
node_radi = radi - 0.2 * radi  # 0.48
angles = [i * (2 * np.pi / num_vars) for i in range(num_vars)]

node_coords = []
for angle in angles:
    node_coords.append((cent_x + np.cos(angle) * node_radi, cent_y + np.sin(angle) * node_radi))

# 6. Initialize Canvas
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')

# Draw grounding grey background circle
bg_circle = plt.Circle((cent_x, cent_y), 0.5, color='gray', alpha=0.12, zorder=0)
ax.add_patch(bg_circle)

# 7. Layer 1: Draw Second-Order Interaction Lines (Dark Blue)
sig_s2_mask = s2_matrix * s2_sig
max_s2 = np.max(sig_s2_mask) if np.any(sig_s2_mask > 0) else 1.0
min_s2 = np.min(sig_s2_mask[sig_s2_mask > 0]) if np.any(sig_s2_mask > 0) else 0.0

for i in range(num_vars):
    for j in range(i + 1, num_vars):
        if s2_sig[i, j] == 1:
            x1, y1 = node_coords[i]
            x2, y2 = node_coords[j]
            # Scales line width between 0.5 and 5.0 exactly like the R pipeline
            lwd = scale_value(s2_matrix[i, j], min_s2, max_s2, 0.5, 5.0)
            ax.plot([x1, x2], [y1, y2], color='darkblue', linewidth=lwd, zorder=1)

# 8. Layer 2 & 3: Draw Overlapping Total-Order (Black) and First-Order (Salmon) Nodes
sig_st = st_vals[st_sig >= 1]
max_st = np.max(sig_st) if len(sig_st) > 0 else 1.0
min_st = np.min(sig_st) if len(sig_st) > 0 else 0.0

sig_s1 = s1_vals[s1_sig >= 1]
max_s1 = np.max(sig_s1) if len(sig_s1) > 0 else 1.0
min_s1 = np.min(sig_s1) if len(sig_s1) > 0 else 0.0

for i, key in enumerate(ordered_keys):
    x, y = node_coords[i]
    
    # Total-Order Outer Boundary Anchor (Radius scaled 0.02 to 0.1)
    if st_sig[i] == 1:
        r_st = scale_value(st_vals[i], min_st, max_st, 0.02, 0.1)
        ax.add_patch(plt.Circle((x, y), r_st, color='black', zorder=2))
    else:
        # Non-significant fallback dot to keep layout structured
        ax.add_patch(plt.Circle((x, y), 0.015, color='gray', zorder=2))
        
    # First-Order Center Core (Radius scaled 0.005 to 0.08)
    if s1_sig[i] == 1:
        r_s1 = scale_value(s1_vals[i], min_s1, max_s1, 0.005, 0.08)
        ax.add_patch(plt.Circle((x, y), r_s1, color='#FF6666', zorder=3))

# 9. Add Perimeter Parameter Text Labels
for i, key in enumerate(ordered_keys):
    angle = angles[i]
    cosa, sina = np.cos(angle), np.sin(angle)
    info = param_info[key]
    
    tx = cent_x + cosa * (radi + 0.04)
    ty = cent_y + sina * (radi + 0.04)
    
    ha = 'left' if cosa >= 0 else 'right'
    va = 'center'
    if abs(cosa) < 0.1:
        ha = 'center'
        va = 'bottom' if sina > 0 else 'top'
        
    ax.text(tx, ty, info['display_name'], color=info['color'], 
            fontsize=9, weight='bold', ha=ha, va=va)

# 10. Add Regional Discipline Structural Grouping Headings
ax.text(0, 0.90, 'Earth sciences', fontsize=13, color='darkgreen', weight='bold', ha='right')
ax.text(-1.25, 0.15, 'Engineering', fontsize=13, color='darkred', weight='bold', ha='left')
ax.text(0.95, -0.35, 'Social sciences', fontsize=13, color='purple', weight='bold', ha='center')

# 11. Build Authentic Custom Legended Scale Box at Bottom (Matches R Output)
y_box = -0.85
# First Order Legend Elements
ax.add_patch(plt.Circle((-0.7, y_box), 0.06, color='#FF6666', zorder=2))
ax.add_patch(plt.Circle((-0.5, y_box), 0.015, color='#FF6666', zorder=2))
ax.text((-0.7), y_box + 0.09, f"{round(100*max_s1)}%", ha='center', fontsize=9, weight='bold')
ax.text((-0.5), y_box + 0.09, f"{round(100*min_s1)}%", ha='center', fontsize=9, weight='bold')
ax.text((-0.6), y_box + 0.16, 'First-order', ha='center', fontsize=10, color='#FF6666', weight='bold')

# Total Order Legend Elements
ax.add_patch(plt.Circle((-0.1, y_box), 0.07, color='black', zorder=2))
ax.add_patch(plt.Circle((0.1, y_box), 0.025, color='black', zorder=2))
ax.text((-0.1), y_box + 0.09, f"{round(100*max_st)}%", ha='center', fontsize=9, weight='bold')
ax.text((0.1), y_box + 0.09, f"{round(100*min_st)}%", ha='center', fontsize=9, weight='bold')
ax.text((0.0), y_box + 0.16, 'Total-order', ha='center', fontsize=10, color='black', weight='bold')

# Second Order Interaction Line Weight Elements
ax.plot([0.45, 0.55], [y_box, y_box], color='darkblue', linewidth=5.0)
ax.plot([0.65, 0.75], [y_box, y_box], color='darkblue', linewidth=0.5)
ax.text(0.50, y_box + 0.09, f"{round(100*max_s2)}%", ha='center', fontsize=9, weight='bold')
ax.text(0.70, y_box + 0.09, f"{round(100*min_s2)}%", ha='center', fontsize=9, weight='bold')
ax.text(0.60, y_box + 0.16, 'Second-order', ha='center', fontsize=10, color='darkblue', weight='bold')

# Save Plot
plt.savefig('S29_SA_RadialPlot_mostlikely_16.png', dpi=300, bbox_inches='tight')
print("Radial plot successfully updated and saved as 'S29_SA_RadialPlot_mostlikely_16.png'")