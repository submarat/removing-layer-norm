# %%
import sys
import os
import numpy as np
import pandas as pd
import torch as t
import matplotlib.pylab as plt
import einops
from datasets import load_dataset

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_models import ModelFactory

# %%
num_neurons = 7
save_path = 'figures/medium'
model_size = 'medium'
os.makedirs(save_path, exist_ok=True)

# %%
# Initialize model factory with all models
model_names = ['baseline', 'finetuned', 'noLN']
model_factory = ModelFactory(model_names,
                             model_dir='../models/',
                             model_size=model_size)

# Set up color schemes for consistent visualization
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
model_colors = {
    'baseline': colors[0],
    'finetuned': colors[1],
    'noLN': colors[2]
}

# Dictionary to store results for each model
results = {}

# Iterate through all models
for model_name, model in model_factory.models.items():
    print(f"Processing model: {model_name}")
    
    # Set model to eval mode and freeze parameters
    model = model
    for param in model.parameters():
        param.requires_grad_(False)

    # Extract weights
    residual_stream_dim = model.cfg.d_model
    Wout = model.blocks[-1].mlp.W_out
    Wu = model.unembed.W_U
    L2_Wout = t.norm(Wout, dim=1, keepdim=True)
    L2_Wu = t.norm(Wu, dim=0, keepdim=True)
    norm_Wout = Wout / L2_Wout
    norm_Wu = Wu / L2_Wu
    
    print(f'{model_name} - Wout: {Wout.shape}, Wu: {Wu.shape}')
    
    # Calculate cosine similarities
    norm_Wu_Wout = einops.einsum(norm_Wu, norm_Wout,
                                'dmodel dvocab, dmlp dmodel -> dmlp dvocab')
    # Find variance across entire vocab
    var = t.var(norm_Wu_Wout, axis=1).cpu().numpy()
    L2_Wout_array = t.squeeze(L2_Wout).cpu().numpy()
    
    # Find entropy neuron candidates (high norm, low variance)
    num_entropy_neurons = 100
    ratio = L2_Wout_array / var
    top_indices = np.argsort(-ratio)[:num_entropy_neurons]
    
    # Store results for this model
    results[model_name] = {
        'Wout': Wout,
        'Wu': Wu,
        'L2_Wout': L2_Wout_array,
        'var': var,
        'ratio': ratio,
        'top_indices': top_indices
    }
    
    # Calculate SVD for unembedding matrix
    U, S, Vt = t.svd(Wu)
    norm_S = S / t.max(S)
    results[model_name]['norm_S'] = norm_S
    results[model_name]['U'] = U
    results[model_name]['S'] = S
    results[model_name]['Vt'] = Vt

# %%
# Create individual plots for each model's normal vs entropy neurons
plt.figure(figsize=(15, 5))

for i, (model_name, model_results) in enumerate(results.items()):
    plt.subplot(1, len(results), i+1)
    
    L2_Wout = model_results['L2_Wout']
    var = model_results['var']
    top_indices = model_results['top_indices']
    
    # Plot normal neurons
    plt.scatter(L2_Wout, var, alpha=0.5, label='Normal')
    
    # Plot top entropy neurons for this model
    plt.scatter(L2_Wout[top_indices[:num_neurons]], var[top_indices[:num_neurons]], 
                color='red', s=30, label='Entropy')
    
    plt.xlabel('||w_out||')
    plt.ylabel('LogitVar(w_out)')
    plt.yscale('log')
    plt.title(f'{model_name}')
    plt.legend()

plt.tight_layout()
plt.savefig(f'{save_path}/all_models_neurons.png', dpi=300)
plt.show()

# %%
# Create SVD plots with all models on the same plot - two columns: full and zoomed
plt.figure(figsize=(15, 6))

# Plot 1: Full SVD for all models
plt.subplot(1, 2, 1)

for model_name, model_results in results.items():
    norm_S = model_results['norm_S']
    x_vals = t.arange(norm_S.shape[0]).cpu().numpy()
    
    plt.plot(x_vals, norm_S.cpu().numpy(), 
             lw=2, label=f'{model_name}', 
             color=model_colors[model_name])

plt.ylim(0, 0.5)
plt.ylabel('Normalised Singular Values')
plt.xlabel('Singular Vector Index')
plt.title('SVD Comparison - All Models')
plt.legend(loc='upper right')

# Plot 2: Zoomed SVD for all models
plt.subplot(1, 2, 2)

for model_name, model_results in results.items():
    S = model_results['S']
    norm_S = S / t.sum(S)
    x_vals = t.arange(norm_S.shape[0]).cpu().numpy()
    
    plt.plot(x_vals, norm_S.cpu().numpy(), 
             lw=2, label=f'{model_name}', 
             color=model_colors[model_name])

plt.ylim(0, 0.001)
plt.xlim(residual_stream_dim - 15, residual_stream_dim - 1)
plt.ylabel('Normalised Singular Values')
plt.xlabel('Singular Vector Index')
plt.title('SVD Comparison - Zoomed')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f'{save_path}/all_models_SVD_comparison.png', dpi=300)
plt.show()

# %%
print(results['baseline']['top_indices'][:10])
