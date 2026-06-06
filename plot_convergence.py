# import libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                   # Use the 'Agg' backend to avoid the Qt/DBus error while plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# PLOT CONVERGENCE TESTING
# ------------------------------------------------------------------
def plot_convergence(csv_filepath, heights_to_plot=[0, 3, 9, 14]):
    # Read in data from the CSV
    df = pd.read_csv(csv_filepath)
    
    # Filter for specific nsow values
    nsow_targets = [1000, 10000, 100000, 500000]
    df = df[df['nsow'].isin(nsow_targets)]

    metrics = ['total_cost', 'bcr', 'reliability']
    y_labels = ['Total Cost [$]', 'Benefit-cost Ratio', 'Lifetime Reliability']
    
    for height in heights_to_plot:
        data_subset = df[df['dh'] == height]
        
        # Added sharex=True to clean up the shared x-axis ticks
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        fig.suptitle(f'Convergence Analysis for Heightening Strategy dh = {height}ft', fontsize=16)
        
        for idx, metric in enumerate(metrics):
            sns.stripplot(ax=axes[idx], data=data_subset, x='nsow', y=metric, 
                            jitter=0.2, alpha=0.6, palette="viridis")
            
            sns.pointplot(ax=axes[idx], data=data_subset, x='nsow', y=metric, 
                            color='black', markers='D', scale=0.5)
            
            # Set custom y-axis labels
            axes[idx].set_ylabel(y_labels[idx], fontsize=14)
            
            # INCREASE TICK LABEL SIZE
            axes[idx].tick_params(axis='both', which='major', labelsize=12)
            
            # ONLY SHOW X-AXIS LABEL ON THE BOTTOM PLOT
            if idx == len(metrics) - 1:
                axes[idx].set_xlabel('Number of Samples', fontsize=14)
            else:
                axes[idx].set_xlabel("") # Prevent seaborn from defaulting to 'nsow'
                
            axes[idx].set_title("") 
            axes[idx].grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'convergence_evGEV_evHV_5e6_{height}.png', dpi=300)
        plt.close(fig) # Close the figure to free up memory

# Run the plotting function using the saved CSV
plot_convergence('convergence_evHV_evGEV_5e6.csv')

# ------------------------------------------------------------------
# CALCULATE & EXPORT SUMMARY STATISTICS (STD & % of MEAN)
# ------------------------------------------------------------------
print("\nCalculating summary statistics...")

df_convergence = pd.read_csv('convergence_evHV_evGEV_5e6.csv')

# Group by elevation height and sample size
stats_df = df_convergence.groupby(['dh', 'nsow']).agg(
    tc_mean=('total_cost', 'mean'),
    tc_std=('total_cost', 'std'),          
    bcr_mean=('bcr', 'mean'),              
    bcr_std=('bcr', 'std'),                
    rel_mean=('reliability', 'mean'),
    rel_std=('reliability', 'std')
).reset_index()

# Calculate standard deviation as a percentage of the mean
stats_df['tc_std_pct'] = (stats_df['tc_std'] / stats_df['tc_mean']) * 100
stats_df['bcr_std_pct'] = (stats_df['bcr_std'] / stats_df['bcr_mean']) * 100
stats_df['rel_std_pct'] = (stats_df['rel_std'] / stats_df['rel_mean']) * 100

# Reorder columns for readability
stats_df = stats_df[[
    'dh', 'nsow', 
    'tc_mean', 'tc_std', 'tc_std_pct', 
    'bcr_mean', 'bcr_std', 'bcr_std_pct', 
    'rel_mean', 'rel_std', 'rel_std_pct'
]]

# Save to CSV
stats_csv_name = 'convergence_summary_stats.csv'
stats_df.to_csv(stats_csv_name, index=False)
print(f"Summary statistics saved to '{stats_csv_name}'")