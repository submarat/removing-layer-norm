# %%
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
import matplotlib.pylab as plt
import seaborn as sns

# %%
df = pd.read_parquet('../data/pile_sub_l256-512_s16.parquet')
baseline = np.load('../experiments/softmax_probabilities_baseline.npy')
finetuned = np.load('../experiments/softmax_probabilities_finetuned.npy')
noLN = np.load('../experiments/softmax_probabilities_nln.npy')

# %%
# Ensure each model's probabilities sum to one for each sample
assert np.allclose(baseline.sum(axis=1), 1.0, rtol=1e-5)
assert np.allclose(finetuned.sum(axis=1), 1.0, rtol=1e-5)
assert np.allclose(noLN.sum(axis=1), 1.0, rtol=1e-5)

# %%
eps = 1e-10  # small epsilon to prevent overflow from log(0)

baseline_losses = -np.log(baseline[np.arange(len(df.target_token)), df.target_token] + eps)
finetuned_losses = -np.log(finetuned[np.arange(len(df.target_token)), df.target_token] + eps)
noln_losses = -np.log(noLN[np.arange(len(df.target_token)), df.target_token] + eps)

# %%
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
plt.savefig('ce_loss_per_model.png', dpi=300)
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
plt.savefig('ce_loss_per_model_by_seqlen.png', dpi=300)
plt.show()

# Print summary statistics
print("\nSummary of samples in each length bin:")
for start, end in bins:
    mask = (df.sequence_length >= start) & (df.sequence_length < end)
    print(f"Length {start}-{end}: {mask.sum()} samples")
    
    
# %%
def ce_loss_difference(losses1, losses2):
    """
    Calculate absolute difference in CE loss between two models
    
    Args:
        losses1: array of CE losses from first model
        losses2: array of CE losses from second model
    Returns:
        Array of absolute differences in loss
    """
    return np.abs(losses1 - losses2)

# Calculate pairwise differences
baseline_vs_finetuned = ce_loss_difference(baseline_losses, finetuned_losses)
baseline_vs_noln = ce_loss_difference(baseline_losses, noln_losses)
finetuned_vs_noln = ce_loss_difference(finetuned_losses, noln_losses)

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
plt.savefig('delta_ce_per_model.png', dpi=300)
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
plt.savefig('delta_ce_per_model_by_seqlen.png', dpi=300)
plt.show()

# Print summary statistics
print("\nSummary of samples in each length bin:")
for start, end in bins:
   mask = (df.sequence_length >= start) & (df.sequence_length < end)
   print(f"Length {start}-{end}: {mask.sum():,} samples")

# %%
def js_divergence_batched(p, q, batch_size=1000, eps=1e-10):
    """
    Compute Jensen-Shannon divergence between two probability distributions in batches
    
    Args:
        p: array of shape (n_samples, d_vocab) containing softmax probabilities
        q: array of shape (n_samples, d_vocab) containing softmax probabilities
        batch_size: number of samples to process at once
        eps: small value to prevent log(0)
    
    Returns:
        Array of shape (n_samples,) containing JS divergence for each sample
    """
    n_samples = len(p)
    js_divs = np.zeros(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        p_batch = p[i:batch_end]
        q_batch = q[i:batch_end]
        
        # Calculate midpoint distribution
        m = 0.5 * (p_batch + q_batch)
        
        # Calculate KL(p||m) and KL(q||m)
        kl_p_m = np.sum(p_batch * np.log(p_batch / (m + eps) + eps), axis=1)
        kl_q_m = np.sum(q_batch * np.log(q_batch / (m + eps) + eps), axis=1)
        
        # Average the KL divergences
        js_divs[i:batch_end] = 0.5 * (kl_p_m + kl_q_m)
    
    return js_divs

# Calculate pairwise JS divergences in batches
js_baseline_finetuned = js_divergence_batched(baseline, finetuned)
js_baseline_noln = js_divergence_batched(baseline, noLN)
js_finetuned_noln = js_divergence_batched(finetuned, noLN)

# %%
# Plot histogram of JS divergences
plt.style.use('seaborn-v0_8-colorblind')
plt.figure(figsize=(10, 6))

plt.hist(js_baseline_finetuned, bins=50, alpha=0.5,
         label=f'Baseline vs Finetuned (avg={js_baseline_finetuned.mean():.2f} ± {js_baseline_finetuned.std():.2f})')
plt.hist(js_baseline_noln, bins=50, alpha=0.5,
         label=f'Baseline vs NoLN (avg={js_baseline_noln.mean():.2f} ± {js_baseline_noln.std():.2f})')
plt.hist(js_finetuned_noln, bins=50, alpha=0.5,
         label=f'Finetuned vs NoLN (avg={js_finetuned_noln.mean():.2f} ± {js_finetuned_noln.std():.2f})')

plt.yscale('log')
plt.xlabel('Jensen-Shannon Divergence')
plt.ylabel('Counts (log scale)')
plt.title('Distribution of Pairwise JS Divergences')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('js_per_model.png', dpi=300)
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
plt.savefig('js_delta_ce_corr.png', dpi=300)
plt.show()

# %%
raw_df = pd.read_parquet('../data/raw_pile10k.parquet')
n_examples = 300
top_fn_idx = set(np.argsort(js_finetuned_noln)[-n_examples:])

# Find overlaps
all_three = top_fn_idx

print(f"Sequences in all three sets: {len(all_three)}")
if len(all_three) > 0:
    print("\nDetails for sequences that all models disagree on most:")
    for idx in all_three:
        orig_idx = df.iloc[idx].original_sequence_id
        print(f"\n{'='*80}")
        print(f"Index {idx} (Original ID: {orig_idx})")
        print(f"Sequence length: {df.sequence_length.iloc[idx]}")
        print(f"JS divergences:")
        print(f"  Baseline vs Finetuned: {js_baseline_finetuned[idx]:.2f}")
        print(f"  Baseline vs NoLN: {js_baseline_noln[idx]:.2f}")
        print(f"  Finetuned vs NoLN: {js_finetuned_noln[idx]:.2f}")
        print(f"\nOriginal text:")
        print(raw_df.iloc[orig_idx].text)
# %%
