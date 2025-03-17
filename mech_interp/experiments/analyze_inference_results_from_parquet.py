#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# load file from parquet
n = 512
df = pd.read_parquet(f'/workspace/removing-layer-norm/mech_interp/data/inference_logs/dataset_apollo-pile_samples_10000_seqlen_{n}/inference_results.parquet')
if 'input_sequence_id' not in df.columns:
    df['input_sequence_id'] = df.index//(n-1)
    

#%% compare JSD vs Top50 JSD
# Baseline vs Finetuned
counts, xedges, yedges, im = plt.hist2d(df['jsd_baseline_vs_finetuned'], df['topk_jsd_baseline_vs_finetuned'], 
                                       bins=50, cmap='viridis', norm=LogNorm())
plt.colorbar(im, label='Count', format='%.0e')
plt.xlabel('JSD (baseline vs finetuned)')
plt.ylabel('Top50 JSD (baseline vs finetuned)') 
plt.show()

# Baseline vs NoLN
counts, xedges, yedges, im = plt.hist2d(df['jsd_baseline_vs_noLN'], df['topk_jsd_baseline_vs_noLN'],
                                       bins=50, cmap='viridis', norm=LogNorm())
plt.colorbar(im, label='Count', format='%.0e')
plt.xlabel('JSD (baseline vs noLN)')
plt.ylabel('Top50 JSD (baseline vs noLN)')
plt.show()

# Finetuned vs NoLN 
counts, xedges, yedges, im = plt.hist2d(df['jsd_finetuned_vs_noLN'], df['topk_jsd_finetuned_vs_noLN'],
                                       bins=50, cmap='viridis', norm=LogNorm())
plt.colorbar(im, label='Count', format='%.0e')
plt.xlabel('JSD (finetuned vs noLN)')
plt.ylabel('Top50 JSD (finetuned vs noLN)')
plt.show()

# %% ###### functions to plot distributions ######
def plot_ce_distribution(df, bins = np.arange(0, 90, 1), log_scale = True):
    plt.hist(df['ce_baseline'], bins=bins, histtype='step', label='baseline')
    plt.hist(df['ce_finetuned'], bins=bins, histtype='step', label='finetuned')
    plt.hist(df['ce_noLN'], bins=bins, histtype='step', label='noLN')
    plt.legend()
    if log_scale:
        plt.yscale('log')
    plt.title('CE Loss Distribution')
    plt.xlabel('CE Loss')
    plt.ylabel('Frequency')
    plt.show()

def plot_jsd_distribution(df, bins = np.arange(0, 1, 0.01), log_scale = True):
    plt.hist(df['jsd_baseline_vs_finetuned'], bins=bins, histtype='step', label='baseline vs finetuned')
    plt.hist(df['jsd_baseline_vs_noLN'], bins=bins, histtype='step', label='baseline vs noLN')
    plt.hist(df['jsd_finetuned_vs_noLN'], bins=bins, histtype='step', label='finetuned vs noLN')
    plt.legend()
    if log_scale:
        plt.yscale('log')
    plt.title('JSD Distribution')
    plt.xlabel('JSD')
    plt.ylabel('Frequency')
    plt.show()

def plot_ce_diff_distribution(df, bins = np.arange(-90, 90, 1), log_scale = True):
    plt.hist(df['ce_diff_baseline_vs_finetuned'], bins=bins, histtype='step', label='baseline vs finetuned')
    plt.hist(df['ce_diff_baseline_vs_noLN'], bins=bins, histtype='step', label='baseline vs noLN')
    plt.hist(df['ce_diff_finetuned_vs_noLN'], bins=bins, histtype='step', label='finetuned vs noLN')
    plt.legend()
    if log_scale:
        plt.yscale('log')
    plt.title('CE Loss Difference Distribution')
    plt.xlabel('CE Loss Difference')
    plt.ylabel('Frequency')
    plt.show()


def plot_ce_hist2d_comparison(df, max_ce=90, bins=100, cmap='viridis'):
    """
    Create 2D histogram plots comparing CE losses between models.
    
    Args:
        df: DataFrame containing CE loss columns
        max_ce: Maximum CE value to display on axes
        bins: Number of bins for the histogram (or tuple for different x/y bins)
        cmap: Colormap to use for the histogram
    """
    # Find max value across all CE columns if not specified
    if max_ce is None:
        max_ce = max(df['ce_baseline'].max(), df['ce_finetuned'].max(), df['ce_noLN'].max())
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Baseline vs Finetuned
    h1 = ax1.hist2d(df['ce_baseline'], df['ce_finetuned'], 
                   bins=bins, range=[[0, max_ce], [0, max_ce]], 
                   cmap=cmap, norm=plt.cm.colors.LogNorm())
    fig.colorbar(h1[3], ax=ax1, label='Frequency')
    ax1.set_xlabel('Baseline CE')
    ax1.set_ylabel('Finetuned CE')
    ax1.set_aspect('equal')
    ax1.set_title('CE Loss: Baseline vs Finetuned')
    
    # Plot 2: Baseline vs NoLN
    h2 = ax2.hist2d(df['ce_baseline'], df['ce_noLN'], 
                   bins=bins, range=[[0, max_ce], [0, max_ce]], 
                   cmap=cmap, norm=plt.cm.colors.LogNorm())
    fig.colorbar(h2[3], ax=ax2, label='Frequency')
    ax2.set_xlabel('Baseline CE')
    ax2.set_ylabel('NoLN CE')
    ax2.set_aspect('equal')
    ax2.set_title('CE Loss: Baseline vs NoLN')
    
    # Plot 3: Finetuned vs NoLN
    h3 = ax3.hist2d(df['ce_finetuned'], df['ce_noLN'], 
                   bins=bins, range=[[0, max_ce], [0, max_ce]], 
                   cmap=cmap, norm=plt.cm.colors.LogNorm())
    fig.colorbar(h3[3], ax=ax3, label='Frequency')
    ax3.set_xlabel('Finetuned CE')
    ax3.set_ylabel('NoLN CE')
    ax3.set_aspect('equal')
    ax3.set_title('CE Loss: Finetuned vs NoLN')
    
    # Add diagonal reference line to each plot
    for ax in [ax1, ax2, ax3]:
        ax.plot([0, max_ce], [0, max_ce], 'k--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2, ax3)

def plot_ce_scatter_comparison(df, max_ce = 90):
    # Find max value across all CE columns
    if max_ce is None:
        max_ce = max(df['ce_baseline'].max(), df['ce_finetuned'].max(), df['ce_noLN'].max())
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Baseline vs Finetuned, colored by NoLN
    scatter1 = ax1.scatter(df['ce_baseline'], df['ce_finetuned'], c=df['ce_noLN'],
                          cmap='viridis', alpha=0.5, s=1)
    fig.colorbar(scatter1, ax=ax1, label='NoLN CE')
    ax1.set_xlabel('Baseline CE')
    ax1.set_ylabel('Finetuned CE')
    ax1.set_xlim(0, max_ce)
    ax1.set_ylim(0, max_ce)
    ax1.set_aspect('equal')
    ax1.set_title('CE Loss: Baseline vs Finetuned')

    # Plot 2: Baseline vs NoLN, colored by Finetuned
    scatter2 = ax2.scatter(df['ce_baseline'], df['ce_noLN'], c=df['ce_finetuned'],
                          cmap='viridis', alpha=0.5, s=1)
    fig.colorbar(scatter2, ax=ax2, label='Finetuned CE')
    ax2.set_xlabel('Baseline CE')
    ax2.set_ylabel('NoLN CE')
    ax2.set_xlim(0, max_ce)
    ax2.set_ylim(0, max_ce)
    ax2.set_aspect('equal')
    ax2.set_title('CE Loss: Baseline vs NoLN')

    # Plot 3: Finetuned vs NoLN, colored by Baseline
    scatter3 = ax3.scatter(df['ce_finetuned'], df['ce_noLN'], c=df['ce_baseline'],
                          cmap='viridis', alpha=0.5, s=1)
    fig.colorbar(scatter3, ax=ax3, label='Baseline CE')
    ax3.set_xlabel('Finetuned CE')
    ax3.set_ylabel('NoLN CE')
    ax3.set_xlim(0, max_ce)
    ax3.set_ylim(0, max_ce)
    ax3.set_aspect('equal')
    ax3.set_title('CE Loss: Finetuned vs NoLN')

    plt.tight_layout()
    plt.show()

def plot_jsd_scatter_comparison(df, max_jsd = 1.0):
    # Find max value across all JSD columns if not specified
    if max_jsd is None:
        max_jsd = max(df['jsd_baseline_vs_finetuned'].max(), 
                     df['jsd_baseline_vs_noLN'].max(), 
                     df['jsd_finetuned_vs_noLN'].max())
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Baseline vs Finetuned, colored by NoLN
    scatter1 = ax1.scatter(df['jsd_baseline_vs_finetuned'], df['jsd_baseline_vs_noLN'], 
                          c=df['jsd_finetuned_vs_noLN'],
                          cmap='viridis', alpha=0.5, s=1)
    fig.colorbar(scatter1, ax=ax1, label='Finetuned vs NoLN JSD')
    ax1.set_xlabel('Baseline vs Finetuned JSD')
    ax1.set_ylabel('Baseline vs NoLN JSD')
    ax1.set_xlim(0, max_jsd)
    ax1.set_ylim(0, max_jsd)
    ax1.set_aspect('equal')
    ax1.set_title('JSD: Baseline vs Finetuned/NoLN')

    # Plot 2: Baseline vs NoLN, colored by Finetuned
    scatter2 = ax2.scatter(df['jsd_baseline_vs_noLN'], df['jsd_finetuned_vs_noLN'], 
                          c=df['jsd_baseline_vs_finetuned'],
                          cmap='viridis', alpha=0.5, s=1)
    fig.colorbar(scatter2, ax=ax2, label='Baseline vs Finetuned JSD')
    ax2.set_xlabel('Baseline vs NoLN JSD')
    ax2.set_ylabel('Finetuned vs NoLN JSD')
    ax2.set_xlim(0, max_jsd)
    ax2.set_ylim(0, max_jsd)
    ax2.set_aspect('equal')
    ax2.set_title('JSD: NoLN Comparisons')

    # Plot 3: Finetuned vs NoLN, colored by Baseline
    scatter3 = ax3.scatter(df['jsd_finetuned_vs_noLN'], df['jsd_baseline_vs_finetuned'], 
                          c=df['jsd_baseline_vs_noLN'],
                          cmap='viridis', alpha=0.5, s=1)
    fig.colorbar(scatter3, ax=ax3, label='Baseline vs NoLN JSD')
    ax3.set_xlabel('Finetuned vs NoLN JSD')
    ax3.set_ylabel('Baseline vs Finetuned JSD')
    ax3.set_xlim(0, max_jsd)
    ax3.set_ylim(0, max_jsd)
    ax3.set_aspect('equal')
    ax3.set_title('JSD: Model Comparisons')

    plt.tight_layout()
    plt.show()

#%% ###### filter dataset ######
# Token IDs for special characters: [('\r', 201), ('\t', 197), ('\x0c', 200), ('\n', 198), ('\n\n', 628)
df_filtered = df[~df['following_token'].isin([201, 197, 200])]
df_filtered = df_filtered[~df_filtered['last_token'].isin([198, 628])]
df_filtered = df_filtered[(df_filtered['context_length'] >1) & (df_filtered['context_length'] < 100)]
df_filtered = df_filtered[df_filtered['input_sequence_id'] != 3784]
df_filtered = df_filtered[df_filtered['input_sequence_id'] != 6634]
df_filtered = df_filtered[df_filtered['input_sequence_id'] != 8859]
# galvin's area
df_LN_specific_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] < 0.05) & (df_filtered['jsd_baseline_vs_noLN'] > 0.3)] 
# luca's area
df_superfiltered = df_LN_specific_diff[(df_LN_specific_diff['ce_finetuned'] < 1) & (df_LN_specific_diff['ce_noLN'] > 2.5)]

#%%
plot_ce_distribution(df)
plot_jsd_distribution(df)
plot_ce_diff_distribution(df)
plot_ce_hist2d_comparison(df, max_ce=90)
plot_ce_hist2d_comparison(df_filtered, max_ce=90)
plot_ce_hist2d_comparison(df_LN_specific_diff, max_ce=90)
plot_ce_hist2d_comparison(df_superfiltered, max_ce=20)
 #%%

#%% plot showing galvin's area
df_LN_specific_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] < 0.05) & (df_filtered['jsd_baseline_vs_noLN'] > 0.3)]
plt.scatter(df_LN_specific_diff['jsd_baseline_vs_finetuned'], df_LN_specific_diff['jsd_baseline_vs_noLN'], s=0.1, label='LN specific diff')
df_both_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] > 0.3) & (df_filtered['jsd_baseline_vs_noLN'] > 0.3)]
plt.scatter(df_both_diff['jsd_baseline_vs_finetuned'], df_both_diff['jsd_baseline_vs_noLN'], s=0.1, label='both diff')
df_no_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] < 0.05) & (df_filtered['jsd_baseline_vs_noLN'] < 0.05)]
plt.scatter(df_no_diff['jsd_baseline_vs_finetuned'], df_no_diff['jsd_baseline_vs_noLN'], s=0.1, label='no diff')
df_finetuned_specific_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] > 0.3) & (df_filtered['jsd_baseline_vs_noLN'] < 0.05)]
plt.scatter(df_finetuned_specific_diff['jsd_baseline_vs_finetuned'], df_finetuned_specific_diff['jsd_baseline_vs_noLN'], s=0.1, label='finetuned specific diff')
plt.xlim(0, 0.7)
plt.ylim(0, 0.7)
plt.legend()
plt.vlines([0.05, 0.3], 0, 0.7, color='grey', linestyle='--')
plt.hlines([0.05, 0.3], 0, 0.7, color='grey', linestyle='--')
plt.xlabel('JSD (baseline vs finetuned)')
plt.ylabel('JSD (baseline vs noLN)')
plt.show()

# Calculate mean context lengths
mean_lengths = {
    'LN specific diff': df_LN_specific_diff['context_length'].mean(),
    'both diff': df_both_diff['context_length'].mean(),
    'no diff': df_no_diff['context_length'].mean(),
    'finetuned specific diff': df_finetuned_specific_diff['context_length'].mean()
}

# Create bar plot
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Default matplotlib colors
plt.bar(mean_lengths.keys(), mean_lengths.values(), color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Context Length')
plt.title('Average Context Length by Group')
plt.tight_layout()
plt.show()
