# %%
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
import matplotlib.pylab as plt
import seaborn as sns

# %%
df = pd.read_parquet('../../data/pile_sub_l256-512_s16.parquet')
baseline = np.load('../../experiments/softmax_probabilities_baseline.npy')
finetuned = np.load('../../experiments/softmax_probabilities_finetuned.npy')
noLN = np.load('../../experiments/softmax_probabilities_nln.npy')

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
plt.savefig('js_per_model.png', dpi=300)
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
plt.savefig('js_per_model_by_seqlen.png', dpi=300)
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
def compute_calibration_error(probs, targets, n_bins=10):
    """
    Compute Expected Calibration Error with equally spaced bins
    """
    # Get confidence (max probability) and predictions
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == targets
    
    # Create equally spaced bins
    bin_edges = np.linspace(0, 1, n_bins + 1)  # creates [0, 0.1, 0.2, ..., 0.9, 1.0]
    bin_indices = np.digitize(confidences, bin_edges) - 1
    
    # Initialize arrays for results
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)
    
    # Calculate calibration for each bin
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.any():
            bin_sizes[i] = mask.sum()
            bin_accuracies[i] = accuracies[mask].mean()
            bin_confidences[i] = confidences[mask].mean()
    
    # Calculate ECE
    ece = np.sum(np.abs(bin_accuracies - bin_confidences) * (bin_sizes / len(confidences)))
    
    return ece, bin_confidences, bin_accuracies, bin_sizes

# Calculate calibration for each model
baseline_cal = compute_calibration_error(baseline, df.target_token)
finetuned_cal = compute_calibration_error(finetuned, df.target_token)
noln_cal = compute_calibration_error(noLN, df.target_token)


# %%
# Plot calibration curves
plt.style.use('seaborn-v0_8-colorblind')
plt.figure(figsize=(10, 6))

# Plot perfect calibration line
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

# Plot each model's calibration
plt.plot(baseline_cal[1], baseline_cal[2], 'o-', label=f'Baseline (ECE={baseline_cal[0]:.4f})')
plt.plot(finetuned_cal[1], finetuned_cal[2], 'o-', label=f'Finetuned (ECE={finetuned_cal[0]:.4f})')
plt.plot(noln_cal[1], noln_cal[2], 'o-', label=f'No LayerNorm (ECE={noln_cal[0]:.4f})')

plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('Calibration Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("calibration_per_model.png", dpi=300)
plt.show()

# Print bin information
print("\nBin Information:")
print(f"{'Bin Range':<12} {'Baseline':>12} {'Finetuned':>12} {'NoLN':>12}")
print("-" * 50)
for i in range(10):
    bin_start = i/10
    bin_end = (i+1)/10
    print(f"{f'{bin_start:.1f}-{bin_end:.1f}':<12} {baseline_cal[3][i]:>12,.0f} {finetuned_cal[3][i]:>12,.0f} {noln_cal[3][i]:>12,.0f}")
# %%
