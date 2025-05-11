#%%
# Imports and setup
import sys
import os
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import load_nln_model, load_finetuned_model
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from mech_interp.load_dataset import DataManager
#%%
# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
n_samples = 100
# idxs_list = torch.randint(0, 768, (10, 50))
idxs_list = [torch.randperm(768)[:50] for _ in range(n_samples)]

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_finetuned = load_finetuned_model()
model_nln = load_nln_model()
model_finetuned.eval()
model_nln.eval()

dm = DataManager(dataset_name='luca-pile', num_samples=100, batch_size=100, max_context=50)
dataloader = dm.create_dataloader()
#%%
prompts = next(iter(dataloader))

# Helper functions
def get_logit_diff(logits_and_answer_token_indices):
    """Calculate difference between correct and incorrect logits"""
    logits, answer_token_indices = logits_and_answer_token_indices
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

def get_mean_activations(model, tokens, position):
    """Get mean activations for a given position"""
    activations = {}
    def forward_hook(act, hook):
        activations[hook.name] = act.detach()
    
    model.reset_hooks()
    model.add_hook(position, forward_hook, "fwd")
    _ = model(tokens)
    model.reset_hooks()
    
    return activations[position].mean(dim=(0,1))

def ablate_neuron(model, tokens, idxs, position, activation_values):
    """Ablate specific neurons by replacing with mean values"""
    def ablation_hook(act, hook):
        modified_act = act.clone()
        modified_act[:, :, idxs] = activation_values[idxs].view(1, 1, -1)
        return modified_act
    
    model.reset_hooks()
    model.add_hook(position, ablation_hook)
    ablated_logits = model(tokens)
    model.reset_hooks()
    return ablated_logits

def run_attribution_patching(model, clean_tokens, token_indeces, position, mean_activations):
    """Run attribution patching for a given model and position"""
    activations = {}
    gradients = {}
    def forward_hook(act, hook):
        activations[hook.name] = act.detach()
        
    def backward_hook(grad, hook):
        gradients[hook.name] = grad.detach()
    
    # Run forward and backward passes
    model.reset_hooks()
    model.add_hook(position, forward_hook, "fwd")
    model.add_hook(position, backward_hook, "bwd")
    logits = model(clean_tokens)
    logit_diff = get_logit_diff([logits, token_indeces])
    logit_diff.backward()
    model.reset_hooks()
    
    # Calculate attributions
    mlp_activations = activations[position]
    mlp_activations_diff = copy.deepcopy(mlp_activations) 
    mlp_activations_diff -= mean_activations.view(1, 1, -1)
    mlp_gradients = gradients[position]
    neuron_attribution_effects = (mlp_activations_diff * mlp_gradients).sum(dim=(0, 1))
    return neuron_attribution_effects


#%%
# Setup experiment positions
positions = []
for i in range(12):
    positions.extend([
        f'blocks.{i}.hook_mlp_out',
        f'blocks.{i}.hook_resid_post', 
        f'blocks.{i}.hook_attn_out'
    ])

positions = [positions[-3]] # only use the last MLP output
print(positions)
# models_list = [("finetuned", model_finetuned), ("nln", model_nln)]
models_list = [("nln", model_nln)]
# Run experiments
results = {}
for position in tqdm(positions):
    results[position] = {}
    for model_name, model in models_list:
        with torch.no_grad():
            clean_tokens = next(iter(dataloader))
            # Get baseline results
            clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            answer_token_indices_ablation_baseline = torch.topk(clean_logits[:, -1, :], k=2, dim=-1).indices

            # list of metrics
            
            clean_logit_diff = get_logit_diff([clean_logits, answer_token_indices_ablation_baseline]).item()
        
            
            ##### ABLATION EXPERIMENT #####
            mean_activations = get_mean_activations(model, clean_tokens, position)
            neuron_ablation_effects = []
            for idxs in idxs_list:
                ablated_logits = ablate_neuron(model, clean_tokens, idxs, position, mean_activations)
                ablated_logit_diff = get_logit_diff([ablated_logits, answer_token_indices_ablation_baseline]).item()
                effect = clean_logit_diff - ablated_logit_diff
                neuron_ablation_effects.append(effect)
        
        # Run attribution patching
        neuron_attribution_effects = run_attribution_patching(model, clean_tokens, answer_token_indices_ablation_baseline, position, mean_activations)
        # Group attributions (this should be possible because we taking a linear approximation)
        with torch.no_grad():
            grouped_attribution_effects = []
            for idxs in idxs_list:
                group_score = neuron_attribution_effects[idxs].sum()
                grouped_attribution_effects.append(group_score.item())
            
            # Store results
            results[position][model_name] = {
                "neuron_ablation_effects": np.array(neuron_ablation_effects),
                "neuron_attribution_effects": neuron_attribution_effects.detach().cpu().numpy(),
                "grouped_attribution_effects": np.array(grouped_attribution_effects),
                'attribution_ablation_correlation': np.corrcoef(
                    np.array(neuron_ablation_effects),
                    np.array(grouped_attribution_effects)
                )[0,1]
            }

#%%

print(type(results[positions[-1]]['nln']['grouped_attribution_effects']))
print(type(results[positions[-1]]['nln']['neuron_ablation_effects']))
plt.scatter(results[positions[-1]]['nln']['grouped_attribution_effects'], results[positions[-1]]['nln']['neuron_ablation_effects'])
plt.xlabel("neuron attribution effect")
plt.ylabel("neuron ablation effect")
plt.show()


# %% plots
results = torch.load( f"results_group_size_top2clean_diff_mean_ablation_vs_attribution.pt")

positions = list(results.keys())

# Create figures directory if it doesn't exist
figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

fig, axes = plt.subplots(12, 3, figsize=(12,48))
for position, ax in zip(positions, axes.flatten()):
    for model_name in ['finetuned', 'nln']:
        ax.scatter(torch.tensor(results[position][model_name]['neuron_ablation_effects']).cpu().numpy(), torch.tensor(results[position][model_name]['neuron_ablation_effects']).cpu().numpy()- torch.tensor(results[position] [model_name]['grouped_attribution_effects']).cpu().numpy(), label=model_name)
    ax.axhline(0, color='black', linewidth=0.5)
    maximum = np.max([
        np.abs(results[position]['finetuned']['neuron_ablation_effects']), 
        np.abs(results[position]['nln']['neuron_ablation_effects'])
        ])*1.05
    minimum = - maximum
    ax.set_xlim(minimum, maximum)
    ax.set_xlabel("ablation effect - attribution effect")
    ax.set_ylabel("attribution effect")
    ax.set_title(f"{position}")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "Apollo_ablation_vs_attrpatch_mean_ablate1.png"))
plt.show()

fig, axes = plt.subplots(12, 3, figsize=(12,48))
for position, ax in zip(positions, axes.flatten()):
    for model_name in ['finetuned', 'nln']:
        ax.scatter(torch.tensor(results[position][model_name]['neuron_ablation_effects']).cpu().numpy(),  torch.tensor(results[position] [model_name]['grouped_attribution_effects']).cpu().numpy(), label=model_name)
    ax.axhline(0, color='black', linewidth=0.5)
    maximum = np.max([
        np.abs(results[position]['finetuned']['neuron_ablation_effects']), 
        np.abs(results[position]['nln']['neuron_ablation_effects'])
        ])*1.05
    minimum = - maximum
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.plot()
    ax.set_xlabel("ablation effect")
    ax.set_ylabel("attribution effect")
    ax.set_title(f"{position}")
    ax.legend()
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(figures_dir, "Apollo_ablation_vs_attrpatch_mean_ablate2.png"))
plt.show()

fig, axes = plt.subplots(1,3, figsize=(12,4))
fig.suptitle("attribution ablation correlation")

# Plot correlations by position type
for i, ax in enumerate(axes):
    positions_subset = positions[i::3]
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions_subset)))
    
    for position, color in zip(positions_subset, colors):
        ax.plot([0,1], 
                [results[position]['finetuned']['attribution_ablation_correlation'],
                 results[position]['nln']['attribution_ablation_correlation']], 
                '-*', label='layer ' + position.split('.')[-2], color=color)
    
    ax.legend()
    ax.set_xlim(-1,2)
    ax.set_ylim(-1,1.05)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["finetuned", "No LayerNorm"])
    ax.set_ylabel("attribution ablation correlation")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f"{positions_subset[0].split('.')[-1]}")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "Apollo_ablation_vs_attrpatch_mean_ablate3.png"))
plt.show()

# Plot correlations by layer
fig, axes = plt.subplots(1,3, figsize=(12,4))
fig.suptitle("attribution ablation correlation")

for i, ax in enumerate(axes):
    positions_subset = positions[i::3]
    corrs_finetuned = [results[position]['finetuned']['attribution_ablation_correlation'] for position in positions_subset]
    corrs_nln = [results[position]['nln']['attribution_ablation_correlation'] for position in positions_subset]
    
    ax.plot(np.arange(len(corrs_finetuned)), corrs_finetuned, label='finetuned')
    ax.plot(np.arange(len(corrs_nln)), corrs_nln, label='No LayerNorm')
    ax.legend()
    ax.set_ylim(-1,1.05)
    ax.set_xlabel("layer")
    ax.set_ylabel("attribution ablation correlation")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f"{positions_subset[0].split('.')[-1]}")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "Apollo_ablation_vs_attrpatch_mean_ablate4.png"))
plt.show()
