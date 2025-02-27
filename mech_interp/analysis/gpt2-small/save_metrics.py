# %%
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
import matplotlib.pylab as plt
import seaborn as sns
from transformers import GPT2TokenizerFast

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
df['ce_baseline'] = baseline_losses
df['ce_finetuned'] = finetuned_losses
df['ce_noln'] = noln_losses

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
df['ce_diff_baseline_finetuned'] = baseline_vs_finetuned 
df['ce_diff_baseline_noln'] = baseline_vs_noln 
df['ce_diff_finetuned_noln'] =  finetuned_vs_noln

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
df['js_baseline_finetuned'] = js_baseline_finetuned
df['js_baseline_noln'] = js_baseline_noln
df['js_finetuned_noln'] = js_finetuned_noln

# %%
df['pred_baseline'] = np.argmax(baseline, axis=1)
df['pred_finetuned'] = np.argmax(finetuned, axis=1)
df['pred_noln'] = np.argmax(noLN, axis=1)

# &&

# %%
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.decode(df.input_sequence.iloc[0])

# %%
df['input_sequence'] = df['input_sequence'].apply(lambda indices: tokenizer.decode(indices))
df['target_token'] = df['target_token'].apply(lambda indices: tokenizer.decode(indices))
df['pred_baseline'] = df['pred_baseline'].apply(lambda indices: tokenizer.decode(indices))
df['pred_finetuned'] = df['pred_finetuned'].apply(lambda indices: tokenizer.decode(indices))
df['pred_noln'] = df['pred_noln'].apply(lambda indices: tokenizer.decode(indices))

# %%
df.to_parquet('metrics_comparison.parquet', index=False)
# %%