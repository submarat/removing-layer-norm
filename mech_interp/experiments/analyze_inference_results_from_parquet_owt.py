#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_parquet('/workspace/removing-layer-norm/mech_interp/data/inference_logs/dataset_apollo-owt_samples_10000_seqlen_100/inference_results.parquet')

print(df.columns)

# %%
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

plot_ce_distribution(df)
plot_jsd_distribution(df)
plot_ce_diff_distribution(df)
plot_ce_scatter_comparison(df, 40)
#%% plot the scatter plot of jsd div before filtering
df_LN_specific_diff = df[(df['jsd_baseline_vs_finetuned'] < 0.05) & (df['jsd_baseline_vs_noLN'] > 0.3)]
plt.scatter(df_LN_specific_diff['jsd_baseline_vs_finetuned'], df_LN_specific_diff['jsd_baseline_vs_noLN'], s=0.1, label='LN specific diff')
df_both_diff = df[(df['jsd_baseline_vs_finetuned'] > 0.3) & (df['jsd_baseline_vs_noLN'] > 0.3)]
plt.scatter(df_both_diff['jsd_baseline_vs_finetuned'], df_both_diff['jsd_baseline_vs_noLN'], s=0.1, label='both diff')
df_no_diff = df[(df['jsd_baseline_vs_finetuned'] < 0.05) & (df['jsd_baseline_vs_noLN'] < 0.05)]
plt.scatter(df_no_diff['jsd_baseline_vs_finetuned'], df_no_diff['jsd_baseline_vs_noLN'], s=0.1, label='no diff')
df_finetuned_specific_diff = df[(df['jsd_baseline_vs_finetuned'] > 0.3) & (df['jsd_baseline_vs_noLN'] < 0.05)]
plt.scatter(df_finetuned_specific_diff['jsd_baseline_vs_finetuned'], df_finetuned_specific_diff['jsd_baseline_vs_noLN'], s=0.1, label='finetuned specific diff')
plt.xlim(0, 0.7)
plt.ylim(0, 0.7)
plt.legend()
plt.xlabel('JSD (baseline vs finetuned)')
plt.ylabel('JSD (baseline vs noLN)')
plt.show()


#%%
df_LN_specific_morethen10 = df_LN_specific_diff[df_LN_specific_diff['context_length'] > 10]
print(df_LN_specific_morethen10.sort_values(by='jsd_baseline_vs_noLN', ascending=False)['full_context_text'].head(10))
print('\n\n')
print(df_LN_specific_morethen10.sort_values(by='jsd_baseline_vs_noLN', ascending=False)['last_token_text'].head(10))
print('\n\n')
print(df_LN_specific_morethen10.sort_values(by='jsd_baseline_vs_noLN', ascending=False)['following_token_text'].head(10))

# # %%  filter out next tokens being \r, \t, \f
# df = pd.read_parquet('/workspace/removing-layer-norm/mech_interp/data/inference_logs/dataset_apollo-pile_samples_10000_seqlen_100/inference_results.parquet')
# df_filtered = df[~df['following_token_text'].isin(['\r', '\t', '\f'])]
# plot_ce_distribution(df_filtered)
# plot_jsd_distribution(df_filtered)

# #%% filter out context lenght > 1
# df_filtered = df[~df['following_token_text'].isin(['\r', '\t', '\f'])]
# df_filtered = df_filtered[df_filtered['context_length'] > 1]
# plot_ce_distribution(df_filtered)
# plot_jsd_distribution(df_filtered)
# plot_ce_diff_distribution(df_filtered)
# plot_ce_diff_distribution(df_filtered, log_scale=False)

# #%% now do a scatter plot of jsd div 
# plt.scatter(df_filtered['jsd_baseline_vs_finetuned'], df_filtered['jsd_baseline_vs_noLN'])
# plt.xlabel('JSD (baseline vs finetuned)')
# plt.ylabel('JSD (baseline vs noLN)')
# plt.xlabel('JSD (baseline vs finetuned)')
# plt.ylabel('JSD (baseline vs noLN)')
# plt.show()

# #%%
# df_LN_specific_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] < 0.05) & (df_filtered['jsd_baseline_vs_noLN'] > 0.3)]
# plt.scatter(df_LN_specific_diff['jsd_baseline_vs_finetuned'], df_LN_specific_diff['jsd_baseline_vs_noLN'], s=0.1, label='LN specific diff')
# df_both_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] > 0.3) & (df_filtered['jsd_baseline_vs_noLN'] > 0.3)]
# plt.scatter(df_both_diff['jsd_baseline_vs_finetuned'], df_both_diff['jsd_baseline_vs_noLN'], s=0.1, label='both diff')
# df_no_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] < 0.05) & (df_filtered['jsd_baseline_vs_noLN'] < 0.05)]
# plt.scatter(df_no_diff['jsd_baseline_vs_finetuned'], df_no_diff['jsd_baseline_vs_noLN'], s=0.1, label='no diff')
# df_finetuned_specific_diff = df_filtered[(df_filtered['jsd_baseline_vs_finetuned'] > 0.3) & (df_filtered['jsd_baseline_vs_noLN'] < 0.05)]
# plt.scatter(df_finetuned_specific_diff['jsd_baseline_vs_finetuned'], df_finetuned_specific_diff['jsd_baseline_vs_noLN'], s=0.1, label='finetuned specific diff')
# plt.xlim(0, 0.7)
# plt.ylim(0, 0.7)
# plt.legend()
# plt.xlabel('JSD (baseline vs finetuned)')
# plt.ylabel('JSD (baseline vs noLN)')
# plt.show()
#%%
df