# %%
import sys
import copy
import numpy as np
import pandas as pd
import torch as t
import matplotlib.pylab as plt
from transformer_lens import HookedTransformer
import einops
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_models import StandardModelLoader


# %%
os.makedirs('figures', exist_ok=True)
os.makedirs('parquets', exist_ok=True)


# %%
model = StandardModelLoader(model_dir='../models/',
                            model_subdir="gpt2_baseline",
                            repo_id="main").load(fold_ln=True,
                                                 center_writing_weights=True,
                                                 center_unembed=True)
model = model.eval()
for param in model.parameters():
    param.requires_grad_(False)  # Using the in-place version


# %%
Wout = model.blocks[-1].mlp.W_out
Wu = model.unembed.W_U
L2_Wout = t.norm(Wout, dim=1, keepdim=True)
L2_Wu = t.norm(Wu, dim=0, keepdim=True)
norm_Wout = Wout / L2_Wout
norm_Wu = Wu / L2_Wu

# %%
print('Wout :', Wout.shape, 'Wu : ', Wu.shape)
norm_Wu_Wout = einops.einsum(norm_Wu, norm_Wout,
                             'dmodel dvocab, dmlp dmodel -> dmlp dvocab')
# Find variance across entire vocab
var = t.var(norm_Wu_Wout, axis=1).cpu().numpy()
L2_Wout = t.squeeze(L2_Wout).cpu().numpy()

# %%
# Find entropy neuron candidates (high norm, low variance)
num_entropy_neurons = 100
ratio = L2_Wout / var
top_indices = np.argsort(-ratio)[:num_entropy_neurons]  # Get top 100 to make random better
ex_indices = [584, 2123, 2378, 2910, 2870, 1611] # Good examples for further analysis

# %%
# Plot normal neurons
plt.scatter(L2_Wout, var, alpha=0.5, label='Normal')

# Plot entropy neurons
plt.scatter(L2_Wout[ex_indices], var[ex_indices], 
            color='red', s=30, label='Entropy')

# Add labels
for idx in ex_indices:
    plt.annotate(str(idx), (L2_Wout[idx], var[idx]))

# Add labels
plt.xlabel('||w_out||')
plt.ylabel('LogitVar(w_out)')
plt.yscale('log')
plt.title('Neuron Type: Normal vs Entropy')
plt.legend()
plt.savefig('figures/baseline_variance.png', dpi=200)
plt.show()

# %%
U, S, Vt = t.svd(Wu)


# %%
print(U.shape, S.shape, Vt.shape)
norm_S = S / t.max(S)

# %%
plt.plot(t.arange(norm_S.shape[0]).cpu().numpy(),
        norm_S.cpu().numpy(),
        lw=2, label='Singular Values') 

for idx in ex_indices:
    Wout_idx = Wout[idx, :]
    # Calculate dot products
    dots = t.matmul(U.T, Wout_idx)
    # Calculate norms
    wout_norm = t.norm(Wout_idx)
    u_norms = t.norm(U, dim=0)
    # Calculate cosine similarity
    cos_sim_idx = (dots / (wout_norm * u_norms)).abs().cpu().numpy() 
    plt.plot(cos_sim_idx, linewidth=1.5, label=f'{idx}')

plt.xlim(0, 767)
plt.ylim(0, 1)
plt.ylabel('Normed Singular Values')
plt.xlabel('Left Singular Vectors')
plt.legend(loc='upper right')
plt.savefig('figures/baseline_SVD.png', dpi=300)
plt.show()

# %%
plt.plot(t.arange(norm_S.shape[0]).cpu().numpy(),
        norm_S.cpu().numpy(),
        lw=2, label='Singular Values') 

for idx in ex_indices:
    Wout_idx = Wout[idx, :]
    # Calculate dot products
    dots = t.matmul(U.T, Wout_idx)
    # Calculate norms
    wout_norm = t.norm(Wout_idx)
    u_norms = t.norm(U, dim=0)
    # Calculate cosine similarity
    cos_sim_idx = (dots / (wout_norm * u_norms)).abs().cpu().numpy() 
    plt.plot(cos_sim_idx, linewidth=1.5, label=f"{idx}")

plt.xlim(730, 767)
plt.ylim(0, 0.3)
plt.ylabel('Normalised Singular Values')
plt.xlabel('Left Singular Vectors')
plt.legend(loc='upper left')
plt.savefig('figures/baseline_SVD_zoomed.png', dpi=300)
plt.show()


# %%
def collate_fn(examples) -> t.Tensor:
    sequences = []
    for ex in examples:
        sequences.append(ex['input_ids'][:512])
    return t.tensor(sequences)

dataset_path = 'lucabaroni/apollo-pile-filtered-10k'
dataset = load_dataset(dataset_path, streaming=True, split="train")
dataset = dataset.shuffle(seed=42).take(1000)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

# %%
hook_names = [
    "blocks.11.hook_resid_pre",
    "blocks.11.hook_attn_out",
    "blocks.11.mlp.hook_post",
    "blocks.11.hook_mlp_out"
]

targets = []
resid_inp = []
attn_out = []
mlp_mid = []
gt_loss = []

with t.no_grad():
    for batch in dataloader:
        device = next(model.parameters()).device 
        batch = batch.to(device)
        
        logits, cache = model.run_with_cache(
        batch,
        names_filter=lambda name: name in hook_names
        )

        # Crop to match target token positions
        target_token = batch[:, 1:].reshape(-1)
        
        # Extract the residual stream, attention out, and middle of mlp final
        resid_pre = cache["blocks.11.hook_resid_pre"][:, :-1, :]
        attn = cache["blocks.11.hook_attn_out"][:, :-1, :]
        mid_mlp = cache["blocks.11.mlp.hook_post"][:, :-1, :]
        logits = logits[:, :-1, :]
        
        # Reshape to (batch * seq_len, ...)
        resid_pre = resid_pre.reshape(-1, resid_pre.size(-1))
        attn = attn.reshape(-1, attn.size(-1))
        mid_mlp = mid_mlp.reshape(-1, mid_mlp.size(-1))
        logits = logits.reshape(-1, logits.size(-1))
        loss = t.nn.functional.cross_entropy(
            logits,
            target_token,
            reduce=False
        )

        # Store for later
        targets.append(target_token)
        resid_inp.append(resid_pre)
        attn_out.append(attn)
        mlp_mid.append(mid_mlp)
        gt_loss.append(loss)

# Concatenate across all batches
targets = t.cat(targets, dim=0)
resid_inp = t.cat(resid_inp, dim=0)
attn_out = t.cat(attn_out, dim=0)
mlp_mid = t.cat(mlp_mid, dim=0)
ref_loss = t.cat(gt_loss, dim=0)

# Save important weights for ablation analysis
Wout = model.blocks[-1].mlp.W_out.clone()
b_out = model.blocks[-11].mlp.b_out.clone()
unembed = copy.deepcopy(model.unembed)

# %%
# Delete model and empty cache to free up gpu memory
del model
t.cuda.empty_cache()

# %%
# Calculate original statistics batch wise to avoid OOM issues
batch_size = 512
num_samples = mlp_mid.shape[0]
all_losses = []
all_entropies = []
all_max_probs = []
all_max_indices = []
all_entropy_activations = []
all_variances = []

# Process in batches
for start_idx in range(0, num_samples, batch_size):
    end_idx = min(start_idx + batch_size, num_samples)
    
    # Get batch slices
    batch_mlp_mid = mlp_mid[start_idx:end_idx]
    batch_resid_inp = resid_inp[start_idx:end_idx]
    batch_attn_out = attn_out[start_idx:end_idx]
    batch_targets = targets[start_idx:end_idx]
    
    # Calculate for this batch
    batch_mlp_out = einops.einsum(Wout, batch_mlp_mid, 'dmlp dmodel, seq dmlp -> seq dmodel') + b_out
    batch_x_orig = batch_resid_inp + batch_attn_out + batch_mlp_out
    batch_x_mean = batch_x_orig.mean(dim=-1, keepdim=True)
    batch_x_var = (batch_x_orig - batch_x_mean).pow(2).mean(dim=-1, keepdim=True)
    batch_x_centred = (batch_x_orig - batch_x_mean) / (batch_x_var + 1e-12).sqrt()
    batch_logits = unembed(batch_x_centred)
    
    # Calculate loss
    batch_loss = t.nn.functional.cross_entropy(
        batch_logits,
        batch_targets,
        reduction='none'
    )
    
    # Calculate softmax probabilities
    batch_probs = t.softmax(batch_logits, dim=-1)
    
    # Calculate entropy: -sum(p * log(p))
    batch_entropy = -t.sum(batch_probs * t.log(batch_probs + 1e-10), dim=-1)
    
    # Get maximum probability and its index
    batch_max_probs, batch_max_indices = t.max(batch_probs, dim=-1)
   
    # Extract the activations for the entropy neurons
    batch_entropy_neurons = batch_mlp_mid[:, ex_indices]
    all_entropy_activations.append(batch_entropy_neurons)
    
    # Store results
    all_losses.append(batch_loss)
    all_entropies.append(batch_entropy)
    all_max_probs.append(batch_max_probs)
    all_max_indices.append(batch_max_indices)
    all_variances.append(batch_x_var)
    
    # Free up memory
    del batch_mlp_mid, batch_resid_inp, batch_attn_out
    del batch_mlp_out, batch_x_orig, batch_logits, batch_probs
    t.cuda.empty_cache()


# %%
# Combine all results
loss = t.cat(all_losses)
entropy = t.cat(all_entropies)
max_probs = t.cat(all_max_probs)
max_indices = t.cat(all_max_indices)
entropy_activations = t.cat(all_entropy_activations, dim=0)  # Combine along the first dimension
variances = t.cat(all_variances, dim=0)

# %%
# Check that loss from original model is close to our 'simulation'
print(ref_loss.mean().item(), loss.mean().item())


# %%
df = pd.DataFrame({'ce_loss' : loss.cpu().detach().numpy(),
                   'entropy': entropy.cpu().detach().numpy(),
                   'target_token': targets.cpu().detach().numpy(),
                   'y_pred': max_probs.cpu().detach().numpy(),
                   'y_hat': max_indices.cpu().detach().numpy(),
                       })
# Add each neuron's activation as a separate column
for i in range(entropy_activations.shape[1]):
    df[f'entropy_neuron_{top_indices[i]}'] = entropy_activations[:, i].cpu().detach().numpy()
df.to_parquet('parquets/baseline_entropy.parquet')

# %%
mean_mlp_act = mlp_mid.mean(dim=0, keepdim=True)

results = {}
for idx in ex_indices:
    # Clone to avoid modifying original
    ablated_mid = mlp_mid.clone()
    ablated_mid[:, idx] = mean_mlp_act[:, idx]
    # Save total effect metrics
    total_loss_ablated = []
    entropy_ablated = []
    max_probs_ablated = []
    max_pred_ablated = []
    # Save direct effect metrics 
    direct_loss_ablated = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        # Get correct batch indices
        batch_targets = targets[start_idx:end_idx]
        batch_resid_inp = resid_inp[start_idx:end_idx]
        batch_attn_out = attn_out[start_idx:end_idx]
        batch_ablated_mlp_mid = ablated_mid[start_idx:end_idx]
        
        # Compute logits 
        batch_ablated_mlp_out = einops.einsum(Wout, batch_ablated_mlp_mid,
                                              'dmlp dmodel, seq dmlp -> seq dmodel') + b_out
        batch_x_ablated = batch_resid_inp + batch_attn_out + batch_ablated_mlp_out
        
        # Calculate Total effect
        batch_x_ablated_mean = batch_x_ablated.mean(dim=-1, keepdim=True)
        batch_x_ablated_var = (batch_x_ablated - batch_x_ablated_mean).pow(2).mean(dim=-1, keepdim=True)
        batch_x_ablated_centred = (batch_x_ablated - batch_x_ablated_mean) / (batch_x_ablated_var + 1e-12).sqrt()
        batch_logits_ablated = unembed(batch_x_ablated_centred)
        batch_loss = t.nn.functional.cross_entropy(batch_logits_ablated, batch_targets,
                                                   reduce=False)
        batch_probs = t.softmax(batch_logits_ablated, dim=-1)
        batch_entropy = -t.sum(batch_probs * t.log(batch_probs + 1e-10), dim=-1)
        batch_max_probs, batch_max_indices = t.max(batch_probs, dim=-1)
        
        total_loss_ablated.append(batch_loss)
        entropy_ablated.append(batch_entropy)
        max_probs_ablated.append(batch_max_probs)
        max_pred_ablated.append(batch_max_indices)
        
        # Calculate Direct effect
        batch_var_orig = variances[start_idx:end_idx] 
        batch_x_ablated_centred = (batch_x_ablated - batch_x_ablated_mean) / (batch_var_orig + 1e-12).sqrt()
        batch_logits_ablated = unembed(batch_x_ablated_centred)
        batch_loss = t.nn.functional.cross_entropy(batch_logits_ablated, batch_targets,
                                                   reduce=False)
        direct_loss_ablated.append(batch_loss)
        
        
    total_loss_ablated = t.cat(total_loss_ablated, dim=0)
    total_entropy_ablated = t.cat(entropy_ablated, dim=0) 
    total_max_probs_ablated = t.cat(max_probs_ablated)
    total_max_pred_ablated = t.cat(max_pred_ablated)
    
    direct_loss_ablated = t.cat(direct_loss_ablated, dim=0)
    total_effect = (loss - total_loss_ablated).abs().cpu().numpy()
    direct_effect = (loss - direct_loss_ablated).abs().cpu().numpy()
    results[idx] = {
        'total_effect': total_effect,
        'direct_effect': direct_effect
    }

    df = pd.DataFrame({'ce_loss' : total_loss_ablated.cpu().detach().numpy(),
                       'entropy': total_entropy_ablated.cpu().detach().numpy(),
                       'target_token': targets.cpu().detach().numpy(),
                       'y_pred': total_max_probs_ablated.cpu().detach().numpy(),
                       'y_hat': total_max_pred_ablated.cpu().detach().numpy(),
                           })
    df.to_parquet(f'parquets/baseline_entropy_{idx}_ablated.parquet')

# %%
# Do the same for random neurons
random_neurons = [i for i in range(Wout.shape[0]) if i not in top_indices]
np.random.seed(666)
random_selection = np.random.choice(random_neurons, size=100, replace=False)

# %%
random_results = {}
for idx in random_selection:
    # Clone to avoid modifying original
    ablated_mid = mlp_mid.clone()
    ablated_mid[:, idx] = mean_mlp_act[:, idx]
    total_loss_ablated = []
    direct_loss_ablated = []
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        # Get correct batch indices
        batch_targets = targets[start_idx:end_idx]
        batch_resid_inp = resid_inp[start_idx:end_idx]
        batch_attn_out = attn_out[start_idx:end_idx]
        batch_ablated_mlp_mid = ablated_mid[start_idx:end_idx]
        
        # Compute logits 
        batch_ablated_mlp_out = einops.einsum(Wout, batch_ablated_mlp_mid,
                                              'dmlp dmodel, seq dmlp -> seq dmodel') + b_out
        batch_x_ablated = batch_resid_inp + batch_attn_out + batch_ablated_mlp_out
        
        # Calculate Total effect
        batch_x_ablated_mean = batch_x_ablated.mean(dim=-1, keepdim=True)
        batch_x_ablated_var = (batch_x_ablated - batch_x_ablated_mean).pow(2).mean(dim=-1, keepdim=True)
        batch_x_ablated_centred = (batch_x_ablated - batch_x_ablated_mean) / (batch_x_ablated_var + 1e-12).sqrt()
        batch_logits_ablated = unembed(batch_x_ablated_centred)
        batch_loss = t.nn.functional.cross_entropy(batch_logits_ablated, batch_targets,
                                                   reduce=False)
        total_loss_ablated.append(batch_loss)
        
        # Calculate Direct effect
        batch_var_orig = variances[start_idx:end_idx] 
        batch_x_ablated_centred = (batch_x_ablated - batch_x_ablated_mean) / (batch_var_orig + 1e-12).sqrt()
        batch_logits_ablated = unembed(batch_x_ablated_centred)
        batch_loss = t.nn.functional.cross_entropy(batch_logits_ablated, batch_targets,
                                                   reduce=False)
        direct_loss_ablated.append(batch_loss)
        
        
    total_loss_ablated = t.cat(total_loss_ablated, dim=0)
    direct_loss_ablated = t.cat(direct_loss_ablated, dim=0)
    total_effect = (loss - total_loss_ablated).abs().cpu().numpy()
    direct_effect = (loss - direct_loss_ablated).abs().cpu().numpy()
    random_results[idx] = {
        'total_effect': total_effect,
        'direct_effect': direct_effect
    }

    
# %%
def plot_effects_boxplots(results, random_results):
    """
    Create box plots comparing total and direct effects for entropy neurons vs random neurons.
     
    Args:
        results: Dictionary with entropy neuron results where keys are neuron indices
        random_results: Dictionary with random neuron results
    """
    # Get indices of entropy neurons from the results dict
    entropy_indices = list(results.keys())
     
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))
     
    # Data to plot
    data = []
    
    # Add entropy neuron data
    for idx in entropy_indices:
        data.append(results[idx]['total_effect'])
        data.append(results[idx]['direct_effect'])
    
    # Concatenate random neuron data
    random_total = np.concatenate([random_results[idx]['total_effect'] for idx in random_results.keys()])
    random_direct = np.concatenate([random_results[idx]['direct_effect'] for idx in random_results.keys()])
    data.append(random_total)
    data.append(random_direct)
    
    # Create box plot
    positions = []
    for i in range(len(entropy_indices)):
        positions.append(i*2 + 1)      # total effect
        positions.append(i*2 + 1.5)    # direct effect
    positions.append(len(entropy_indices)*2 + 1)    # random total
    positions.append(len(entropy_indices)*2 + 1.5)  # random direct
    
    bp = ax.boxplot(data, positions=positions, widths=0.3, patch_artist=True, showfliers=False)    
    
    # Set colors for the boxes
    colors = []
    for i in range(len(entropy_indices)):
        colors.extend(['lightblue', 'lightgreen'])
    colors.extend(['lightblue', 'lightgreen'])
    
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color)
    
    # Set the x-tick labels
    positions = []
    tick_labels = []
    for i, idx in enumerate(entropy_indices):
        positions.append(i*2 + 1.25)  # Center between the pair
        tick_labels.append(f"{idx}")
    positions.append(len(entropy_indices)*2 + 1.25)
    tick_labels.append("Random")
    
    plt.xticks(positions, tick_labels)
    
    # Add labels
    plt.ylabel('|ΔLoss|')
    plt.xlabel('Neuron Name')
    plt.ylim(-0.001, 0.09)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Total'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Direct')
    ]
    ax.legend(handles=legend_elements, title="Effect Type")
    
    # Add vertical line before random neurons
    plt.axvline(x=len(entropy_indices)*2 + 0.5, color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/baseline_total_and_direct.png', dpi=300)
    plt.show()

plot_effects_boxplots(results, random_results)