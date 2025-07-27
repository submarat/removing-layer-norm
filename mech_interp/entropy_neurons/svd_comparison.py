# %%
import sys
import os
import numpy as np
import pandas as pd
import torch as t
import matplotlib.pylab as plt
import seaborn as sns
import einops

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_models import ModelFactory


# %%
num_neurons = 7
model_size = 'medium'
save_path = f'figures/{model_size}'
os.makedirs(save_path, exist_ok=True)

# %%
# Initialize model factory with all models
model_names = ['baseline', 'finetuned', 'noLN']
model_factory = ModelFactory(model_names,
                             model_dir='../models/',
                             model_size=model_size)

# Set up color schemes for consistent visualization
colors = sns.color_palette("colorblind")

model_type = model_size.capitalize()
model_labels = {
            'baseline': f'GPT2-{model_type} original',
            'finetuned': f'GPT2-{model_type} vanilla FT',
            'noLN': f'GPT2-{model_type} LN-free FT'
        }

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
    bout = model.blocks[-1].mlp.b_out
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
    L2_Wu_array = t.squeeze(L2_Wu).cpu().numpy()
    
    # Find entropy neuron candidates (high norm, low variance)
    num_entropy_neurons = 100
    ratio = L2_Wout_array / var
    top_indices = np.argsort(-ratio)[:num_entropy_neurons]
    
    # Store results for this model
    results[model_name] = {
        'Wout': Wout.cpu().numpy(),
        'bout': bout.cpu().numpy(),
        'Wu': Wu,
        'L2_Wout': L2_Wout_array,
        'L2_Wu': L2_Wu_array,
        'var': var,
        'ratio': ratio,
        'top_indices': top_indices
    }
    
    # Calculate SVD for unembedding matrix
    U, S, Vt = t.svd(Wu)
    norm_S = S / t.max(S)
    
    # Calculate Cosine Sim of first entropy neuron with singular vectors
    first_entropy_neuron_idx = top_indices[0]
    Wout_idx = Wout[first_entropy_neuron_idx, :]
    dots = t.matmul(U.T, Wout_idx)
    # Calculate norms
    wout_norm = t.norm(Wout_idx)
    u_norms = t.norm(U, dim=0)
    # Calculate cosine similarity
    cos_sim = (dots / (wout_norm * u_norms)).abs().cpu().numpy()
    squared_cos_sim = cos_sim ** 2
    cum_squared_cos_sim = np.cumsum(squared_cos_sim)
 
    results[model_name]['norm_S'] = norm_S
    results[model_name]['U'] = U
    results[model_name]['S'] = S
    results[model_name]['Vt'] = Vt
    results[model_name]['cos_sim'] = cos_sim
    results[model_name]['squared_cos_sim'] = squared_cos_sim
    results[model_name]['cum_squared_cos_sim'] = cum_squared_cos_sim

    
# %%
for model_name in model_names:
    # Get the Wout and bout values
    Wout = results[model_name]['Wout']
    bout = results[model_name]['bout']
    neuron_indices = results[model_name]['top_indices'][0:1]
    
    # Print the bias for each neuron
    print("Printing 'effective' bias for each entropy neuron")
    for i, idx in enumerate(neuron_indices):
        neuron_weights = Wout[idx, :]
        norm_weights = neuron_weights / np.linalg.norm(neuron_weights)
        effective_bias = np.dot(norm_weights, bout)
        # Print the result
        print(f"{model_name:<10} : neuron {idx}\t{effective_bias:.6f}")    

        
# %%
# Combined plot with all models' W_out norms on a single histogram
plt.figure(figsize=(10, 6))

for model_name, model_results in results.items():
    L2_Wout = model_results['L2_Wout']
    plt.hist(L2_Wout, bins=50, 
             label=f'{model_labels[model_name]}',
             color=model_colors[model_name],
             histtype='step', linewidth=3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('||w_out|| Norm', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.legend(loc='upper right', fontsize=16)
plt.tight_layout()
plt.savefig(f'{save_path}/all_models_Wout_norm.png', dpi=300)
plt.show()


# %%
# Combined plot with all models' W_u norms on a single histogram
plt.figure(figsize=(10, 6))

for model_name, model_results in results.items():
    L2_Wu = model_results['L2_Wu']
    L2_Wu_normalized = (L2_Wu - np.min(L2_Wu)) / (np.max(L2_Wu) - np.min(L2_Wu))
    plt.hist(L2_Wu_normalized, bins=60, 
             label=f'{model_labels[model_name]}',
             color=model_colors[model_name],
             histtype='step', linewidth=3)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Normalised ||W_u|| Norm', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.legend(loc='upper right', fontsize=16)
plt.tight_layout()
plt.savefig(f'{save_path}/all_models_Wu_norm.png', dpi=300)
plt.show()

# %%
# Create individual plots for each model's normal vs entropy neurons
plt.figure(figsize=(15, 5))

for i, (model_name, model_results) in enumerate(results.items()):
    plt.subplot(1, len(results), i+1)
    
    L2_Wout = model_results['L2_Wout']
    var = model_results['var']
    top_indices = model_results['top_indices']
    
    # Plot normal neurons
    plt.scatter(L2_Wout, var, alpha=0.5, label='Normal', s=30)
    
    # Plot top entropy neurons for this model
    plt.scatter(L2_Wout[top_indices[:num_neurons]], var[top_indices[:num_neurons]], 
                color='red', s=40, label='Entropy')
    
    plt.xlabel('||w_out||', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if i == 0:
        plt.ylabel('LogitVar(w_out)', fontsize=16)
    plt.yscale('log')
    plt.title(f'{model_labels[model_name]}', fontsize=14)
    plt.legend(loc='upper right', fontsize=16)

plt.tight_layout()
plt.savefig(f'{save_path}/all_models_neurons.png', dpi=300)
plt.show()

# %%
# Create SVD plots with all models on the same plot - two columns: full and zoomed
plt.figure(figsize=(8, 8))

# Zoomed SVD for all models
plt.subplot(1, 1, 1)

for model_name, model_results in results.items():
    norm_S = model_results['norm_S']
    x_vals = t.arange(norm_S.shape[0]).cpu().numpy()
    
    # Plot lower singular values to visualise nullspace
    plt.plot(x_vals, norm_S.cpu().numpy(), 
             lw=3, label=f'{model_labels[model_name]}', 
             color=model_colors[model_name])
    
    # Plot entropy neuron overlap with singular vectors
    overlaps = results[model_name]['cos_sim']
    plt.plot(x_vals, overlaps, color=model_colors[model_name], linestyle='--', lw=2)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(residual_stream_dim - 50, residual_stream_dim - 1)
plt.ylim(0, 0.2)
plt.ylabel('Normalised Singular Values', fontsize=16)
plt.xlabel('Singular Vector Index', fontsize=16)
#plt.title('SVD Comparison - Nullspace')
plt.legend(loc='upper right', fontsize=16)

plt.tight_layout()
plt.savefig(f'{save_path}/all_models_SVD_comparison.png', dpi=300)
plt.show()

# %%
arr = results['baseline']['top_indices'][:50]
print(str(arr.tolist()).replace(' ', ''))