# %%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

# %%
df = pd.read_parquet('metrics_comparison.parquet')

# %%
baseline_losses = df['ce_baseline']
finetuned_losses = df['ce_finetuned']
noln_losses = df['ce_noln']

plt.style.use('seaborn-v0_8-colorblind')
plt.figure(figsize=(10, 6))

# Plot histograms with transparency
plt.hist(baseline_losses, bins=50, alpha=0.5, 
        label=f'Baseline (avg={baseline_losses.mean():.2f} ± {baseline_losses.std():.2f})')
plt.hist(finetuned_losses, bins=50, alpha=0.5,
        label=f'Finetuned (avg={finetuned_losses.mean():.2f} ± {finetuned_losses.std():.2f})')
plt.hist(noln_losses, bins=50, alpha=0.5,
        label=f'No LayerNorm (avg={noln_losses.mean():.2f} ± {noln_losses.std():.2f})')


plt.xlabel('Cross Entropy Loss')
plt.ylabel('Counts (log scale)')
plt.yscale('log')
plt.title('Distribution of Token-Level Cross Entropy Losses')
plt.xlim(0, 25)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/ce_loss_per_model.png', dpi=300)
plt.show()

# %%
plt.style.use('seaborn-v0_8-colorblind')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Define sequence length bins
bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

for i, (start, end) in enumerate(bins):
    # Get mask for this length bin
    mask = (df.sequence_length >= start) & (df.sequence_length < end)
    
    # Skip if no samples in this bin
    if not mask.any():
        continue
        
    # Plot histograms for this sequence length bin
    axes[i].hist(baseline_losses[mask], bins=50, alpha=0.5,
                 label=f'Baseline (avg={baseline_losses[mask].mean():.2f} ± {baseline_losses[mask].std():.2f})')
    axes[i].hist(finetuned_losses[mask], bins=50, alpha=0.5,
                 label=f'Finetuned (avg={finetuned_losses[mask].mean():.2f} ± {finetuned_losses[mask].std():.2f})')
    axes[i].hist(noln_losses[mask], bins=50, alpha=0.5,
                 label=f'No LayerNorm (avg={noln_losses[mask].mean():.2f} ± {noln_losses[mask].std():.2f})')
    
    axes[i].set_xlabel('Cross Entropy Loss')
    axes[i].set_ylabel('Counts (log scale)')
    axes[i].set_yscale('log')
    axes[i].set_title(f'Sequence Length {start}-{end}\n(n={mask.sum()})')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Remove empty subplot if we have one
if len(bins) < 6:
    fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('figures/ce_loss_per_model_by_seqlen.png', dpi=300)
plt.show()


# %%
baseline_vs_finetuned = df['ce_diff_baseline_finetuned']
baseline_vs_noln = df['ce_diff_baseline_noln']
finetuned_vs_noln = df['ce_diff_finetuned_noln']

# %%
plt.style.use('seaborn-v0_8-colorblind')
plt.figure(figsize=(10, 6))

plt.hist(baseline_vs_finetuned, bins=50, alpha=0.5,
         label=f'Baseline vs Finetuned (avg={baseline_vs_finetuned.mean():.2f} ± {baseline_vs_finetuned.std():.2f})')
plt.hist(baseline_vs_noln, bins=50, alpha=0.5,
         label=f'Baseline vs NoLN (avg={baseline_vs_noln.mean():.2f} ± {baseline_vs_noln.std():.2f})')
plt.hist(finetuned_vs_noln, bins=50, alpha=0.5,
         label=f'Finetuned vs NoLN (avg={finetuned_vs_noln.mean():.2f} ± {finetuned_vs_noln.std():.2f})')

plt.xlabel('Absolute Difference in Cross Entropy Loss')
plt.yscale('log')
plt.ylabel('Counts (log scale)')
plt.title('Distribution of Pairwise Loss Differences')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/delta_ce_per_model.png', dpi=300)
plt.show()

# %%
plt.style.use('seaborn-v0_8-colorblind')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Define sequence length bins with finer granularity for shorter sequences
bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

for i, (start, end) in enumerate(bins):
   # Get mask for this length bin
   mask = (df.sequence_length >= start) & (df.sequence_length < end)
   
   # Skip if no samples in this bin
   if not mask.any():
       continue
       
   # Plot histograms for this sequence length bin
   axes[i].hist(baseline_vs_finetuned[mask], bins=50, alpha=0.5,
                label=f'Baseline vs Finetuned (avg={baseline_vs_finetuned[mask].mean():.2f} ± {baseline_vs_finetuned[mask].std():.2f})')
   axes[i].hist(baseline_vs_noln[mask], bins=50, alpha=0.5,
                label=f'Baseline vs NoLN (avg={baseline_vs_noln[mask].mean():.2f} ± {baseline_vs_noln[mask].std():.2f})')
   axes[i].hist(finetuned_vs_noln[mask], bins=50, alpha=0.5,
                label=f'Finetuned vs NoLN (avg={finetuned_vs_noln[mask].mean():.2f} ± {finetuned_vs_noln[mask].std():.2f})')
   
   axes[i].set_yscale('log')
   axes[i].set_xlabel('Absolute Difference in Cross Entropy Loss')
   axes[i].set_ylabel('Counts (log scale)')
   axes[i].set_title(f'Sequence Length {start}-{end}\n(n={mask.sum():,})')
   axes[i].legend()
   axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/delta_ce_per_model_by_seqlen.png', dpi=300)
plt.show()

# %%
js_baseline_finetuned = df['js_baseline_finetuned']
js_baseline_noln = df['js_baseline_noln']
js_finetuned_noln = df['js_finetuned_noln']

# %%
# Plot histogram of JS divergences
plt.style.use('seaborn-v0_8-colorblind')
plt.figure(figsize=(10, 6))

plt.hist(js_baseline_finetuned, bins=50, alpha=0.5,
         label=f'Baseline vs Finetuned (avg={js_baseline_finetuned.mean():.3f} ± {js_baseline_finetuned.std():.3f})')
plt.hist(js_baseline_noln, bins=50, alpha=0.5,
         label=f'Baseline vs NoLN (avg={js_baseline_noln.mean():.3f} ± {js_baseline_noln.std():.3f})')
plt.hist(js_finetuned_noln, bins=50, alpha=0.5,
         label=f'Finetuned vs NoLN (avg={js_finetuned_noln.mean():.3f} ± {js_finetuned_noln.std():.3f})')

plt.yscale('log')
plt.xlabel('Jensen-Shannon Divergence')
plt.ylabel('Counts (log scale)')
plt.title('Distribution of Pairwise JS Divergences')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/js_per_model.png', dpi=300)
plt.show()

# %%
# Plot by sequence length bins
plt.style.use('seaborn-v0_8-colorblind')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 400), (400, 500)]

for i, (start, end) in enumerate(bins):
    mask = (df.sequence_length >= start) & (df.sequence_length < end)
    
    if not mask.any():
        continue
        
    axes[i].hist(js_baseline_finetuned[mask], bins=50, alpha=0.5,
                 label=f'Baseline vs Finetuned (avg={js_baseline_finetuned[mask].mean():.3f} ± {js_baseline_finetuned[mask].std():.3f})')
    axes[i].hist(js_baseline_noln[mask], bins=50, alpha=0.5,
                 label=f'Baseline vs NoLN (avg={js_baseline_noln[mask].mean():.3f} ± {js_baseline_noln[mask].std():.3f})')
    axes[i].hist(js_finetuned_noln[mask], bins=50, alpha=0.5,
                 label=f'Finetuned vs NoLN (avg={js_finetuned_noln[mask].mean():.3f} ± {js_finetuned_noln[mask].std():.3f})')
    
    axes[i].set_yscale('log')
    axes[i].set_xlabel('JS Divergence')
    axes[i].set_ylabel('Count (log scale)')
    axes[i].set_title(f'Sequence Length {start}-{end}\n(n={mask.sum():,})')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/js_per_model_by_seqlen.png', dpi=300)
plt.show()

# %%
plt.style.use('seaborn-v0_8-colorblind')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot for each pair of models
pairs = [
    (baseline_vs_finetuned, js_baseline_finetuned, 'Baseline vs Finetuned'),
    (baseline_vs_noln, js_baseline_noln, 'Baseline vs NoLN'),
    (finetuned_vs_noln, js_finetuned_noln, 'Finetuned vs NoLN')
]

for ax, (ce_diff, js_div, title) in zip(axes, pairs):
    # Calculate correlation
    correlation = np.corrcoef(ce_diff, js_div)[0,1]
    
    # Create scatter plot
    ax.scatter(ce_diff, js_div, alpha=0.1, s=1)
    ax.set_xlabel('Absolute CE Difference')
    ax.set_ylabel('JS Divergence')
    ax.set_title(f'{title}\nr = {correlation:.2f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/js_delta_ce_corr.png', dpi=300)
plt.show()

# %%
plt.style.use('seaborn-v0_8-colorblind')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.ravel()

mask = df.sequence_length == 1
# First plot CE losses
axes[0].hist(baseline_losses[mask], bins=50, alpha=0.5,
             label=f'Baseline (avg={baseline_losses[mask].mean():.2f} ± {baseline_losses[mask].std():.2f})')
axes[0].hist(finetuned_losses[mask], bins=50, alpha=0.5,
             label=f'Finetuned (avg={finetuned_losses[mask].mean():.2f} ± {finetuned_losses[mask].std():.2f})')
axes[0].hist(noln_losses[mask], bins=50, alpha=0.5,
             label=f'No LayerNorm (avg={noln_losses[mask].mean():.2f} ± {noln_losses[mask].std():.2f})')   
axes[0].set_xlabel('Cross Entropy Loss')
axes[0].set_ylabel('Counts (log scale)')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
    
# Next abs(Delta CE)
axes[1].hist(baseline_vs_finetuned[mask], bins=50, alpha=0.5,
             label=f'Baseline vs Finetuned (avg={baseline_vs_finetuned[mask].mean():.2f} ± {baseline_vs_finetuned[mask].std():.2f})')
axes[1].hist(baseline_vs_noln[mask], bins=50, alpha=0.5,
             label=f'Baseline vs NoLN (avg={baseline_vs_noln[mask].mean():.2f} ± {baseline_vs_noln[mask].std():.2f})')
axes[1].hist(finetuned_vs_noln[mask], bins=50, alpha=0.5,
             label=f'Finetuned vs NoLN (avg={finetuned_vs_noln[mask].mean():.2f} ± {finetuned_vs_noln[mask].std():.2f})')
axes[1].set_xlabel('Absolute Difference in Cross Entropy Loss')
axes[1].set_ylabel('Counts (log scale)')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)


all_js_data = np.concatenate([
    js_baseline_finetuned[mask], 
    js_baseline_noln[mask], 
    js_finetuned_noln[mask]
])
min_val, max_val = np.min(all_js_data), np.max(all_js_data)
bin_edges = np.linspace(min_val, max_val, 30)  # Use 30 points = 29 bins

# Finally JS divergence   
axes[2].hist(js_baseline_finetuned[mask], bins=bin_edges, alpha=0.5,
             label=f'Baseline vs Finetuned (avg={js_baseline_finetuned[mask].mean():.3f} ± {js_baseline_finetuned[mask].std():.3f})')
axes[2].hist(js_baseline_noln[mask], bins=bin_edges, alpha=0.5,
             label=f'Baseline vs NoLN (avg={js_baseline_noln[mask].mean():.3f} ± {js_baseline_noln[mask].std():.3f})')
axes[2].hist(js_finetuned_noln[mask], bins=bin_edges, alpha=0.5,
             label=f'Finetuned vs NoLN (avg={js_finetuned_noln[mask].mean():.3f} ± {js_finetuned_noln[mask].std():.3f})')
axes[2].set_xlabel('Jensen-Shannon Divergence')
axes[2].set_ylabel('Counts (log scale)')
axes[2].set_yscale('log')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 0.7)
axes[2].set_ylim(1, 1e4)

plt.tight_layout()
plt.savefig('figures/metrics_on_BOS.png', dpi=300)
plt.show()