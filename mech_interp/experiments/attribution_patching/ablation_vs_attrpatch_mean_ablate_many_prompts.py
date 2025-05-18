#%%
import numpy as np
import pickle
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
from torch.nn import functional as F
from mech_interp.experiments.attribution_patching.utils import get_mean_activations

#%%
# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model_finetuned = load_finetuned_model()
model_nln = load_nln_model()
model_name = "nln"

# Setup experiment positions
positions = []
for i in range(12):
    positions.extend([
        f'blocks.{i}.hook_mlp_out',
        f'blocks.{i}.hook_resid_post', 
        f'blocks.{i}.hook_attn_out'
    ])
dir = "mean_activations"
if not os.path.exists(f"{dir}/mean_activations_{model_name}.pkl"):
    mean_activations = get_mean_activations(model_nln, positions) 
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}/mean_activations_{model_name}.pkl", "wb") as f:
        pickle.dump(mean_activations, f)
else:
    with open(f"{dir}/mean_activations_{model_name}.pkl", "rb") as f:
        mean_activations = pickle.load(f)


#%%
def get_logit_diff(logits_and_answer_token_indices):
    """Calculate difference between correct and incorrect logits"""
    logits, answer_token_indices = logits_and_answer_token_indices
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

def ablate_neuron(model, tokens, idxs, position, activation_values):
    """Ablate specific neurons by replacing with mean values"""
    def ablation_hook(act, hook):
        # Create a new tensor to avoid in-place modification
        modified_act = act.clone()
        modified_act[:, :, idxs] = activation_values[idxs].view(1, 1, -1)
        return modified_act
    
    model.reset_hooks()
    model.add_hook(position, ablation_hook, "fwd")
    ablated_logits = model(tokens)
    model.reset_hooks()
    return ablated_logits

def run_attribution_patching(model, clean_tokens, position, mean_activations, metric, answer_token_indices=None, correct_token=None, top_logit_idx=None, clean_logits=None):
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

    if metric == 'logit_diff':
        f = get_logit_diff([logits, answer_token_indices])
    if metric == 'top_logit':
        f = logits[np.arange(logits.shape[0]), -1, top_logit_idx].diagonal().mean()
    if metric == 'top_logit_log_prop':
        f = torch.log(torch.softmax(logits[:, -1, :], dim=-1))[:, top_logit_idx].diagonal().mean()
    if metric == 'kl_div_with_clean':
        f = compute_kl_divergence(logits[:, -1, :], clean_logits[:, -1, :]).mean(dim=0)
    if metric == 'log_props_correct':
        f = torch.log(torch.softmax(logits[:, -2,  :], dim=-1))[:, correct_token].diagonal().mean()
    if metric == 'CE':
        f = compute_CE(torch.log(torch.softmax(logits[:, -2,  :], dim=-1)), correct_token).mean(dim=0)
    f.backward()
    model.reset_hooks()

    # Calculate attributions
    mlp_activations = activations[position]
    mlp_activations_diff = copy.deepcopy(mlp_activations)
    mlp_activations_diff -= mean_activations.view(1, 1, -1)
    mlp_gradients = gradients[position]
    neuron_attribution_effects = (mlp_activations_diff * mlp_gradients).sum(dim=(0, 1))
    return neuron_attribution_effects


def compute_kl_divergence(logits_p, logits_q, eps=1e-8):
    """
    Compute KL divergence between two distributions represented by logits.
    
    Args:
        logits_p: Tensor of shape [batch_size, seq_len, vocab_size] - first distribution
        logits_q: Tensor of shape [batch_size, seq_len, vocab_size] - second distribution
        eps: Small epsilon value for numerical stability
        
    Returns:
        KL divergence tensor of shape [batch_size, seq_len]
    """
    # Convert logits to probabilities
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    
    # Add small epsilon for numerical stability
    p = p + eps
    q = q + eps
    
    # Renormalize
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    # Compute KL divergence: sum(p * log(p/q))
    kl_div = p * (torch.log(p) - torch.log(q))
    
    # Sum over vocabulary dimension
    return kl_div.sum(dim=-1)

def compute_CE(log_props, correct_token):
    return -torch.gather(log_props, dim=-1, index=correct_token.unsqueeze(-1)).squeeze(-1).contiguous()

# Setup experiment positions
positions = []
for i in range(12):
    positions.extend([
        f'blocks.{i}.hook_mlp_out',
        f'blocks.{i}.hook_resid_post', 
        f'blocks.{i}.hook_attn_out'
    ])


TYPE_OF_PROMPTS = 'apollo'
if TYPE_OF_PROMPTS == 'ioi':
    # Test data
    prompts = [
        "When John and Mary went to the shops, John gave the bag to",
        "When John and Mary went to the shops, Mary gave the bag to",
        "When Tom and James went to the park, James gave the ball to",
        "When Tom and James went to the park, Tom gave the ball to",
        "When Dan and Sid went to the shops, Sid gave an apple to",
        "When Dan and Sid went to the shops, Dan gave an apple to",
        "After Martin and Amy went to the park, Amy gave a drink to",
        "After Martin and Amy went to the park, Martin gave a drink to",
        ]
    answers = [
        (" Mary", " John"),
        (" John", " Mary"),
        (" Tom", " James"),
        (" James", " Tom"),
        (" Dan", " Sid"),
        (" Sid", " Dan"),
        (" Martin", " Amy"),
        (" Amy", " Martin"),
    ]
elif TYPE_OF_PROMPTS == 'apollo':
    dm = DataManager(dataset_name='luca-pile', num_samples=100, batch_size=100, max_context=50)
    dataloader = dm.create_dataloader()
    clean_tokens = next(iter(dataloader)).to(model_nln.cfg.device)
else:
    raise ValueError(f"Invalid type of prompts: {TYPE_OF_PROMPTS}")

metrics = ['logit_diff', 'top_logit', 'top_logit_log_prop', 'kl_div_with_clean', 'log_props_correct', 'CE']

#%%

# indices to ablate
n_samples = 100
idxs_list = [torch.randperm(768)[:50] for _ in range(n_samples)]

# Setup experiment positions
positions = []
for i in range(12):
    positions.extend([
        f'blocks.{i}.hook_mlp_out',
        f'blocks.{i}.hook_resid_post', 
        f'blocks.{i}.hook_attn_out'
    ])
# positions = [positions[-3]] # only use the last MLP output
print(positions)
models_list = [("nln", model_nln), ("finetuned", model_finetuned)]
#%%
# Run experiments
results = {}
for model_name, model in models_list:
    results[model_name] = {}
    for position in tqdm(positions):
        with torch.no_grad():››
            results[model_name][position] = {}
            if TYPE_OF_PROMPTS == 'ioi':
                clean_tokens = model.to_tokens(prompts)
                answer_token_indices = torch.tensor(
                    [[model.to_single_token(answers[i][j]) for j in range(2)]
                    for i in range(len(answers))],
                    device=model.cfg.device,
                    dtype=torch.int64
                )
                clean_logits, clean_cache = model.run_with_cache(clean_tokens)
            elif TYPE_OF_PROMPTS == 'apollo':
                clean_logits, clean_cache = model.run_with_cache(clean_tokens)
                answer_token_indices = torch.topk(clean_logits[:, -1, :], k=2, dim=-1).indices

            
            top_logit_idx = torch.topk(clean_logits[:, -1, :], k=1, dim=-1).indices

            correct_token = clean_tokens[:, -1]
            # list of metrics for clean logits
            clean_logit_diff = get_logit_diff([clean_logits, answer_token_indices])
            clean_top_logit = clean_logits[:, -1, top_logit_idx]
            clean_log_props = torch.log(torch.softmax(clean_logits[:, -1, :], dim=-1))
            clean_top_log_prop = clean_log_props[:, top_logit_idx]
            clean_kl_div_with_clean = compute_kl_divergence(clean_logits[:, -1, :], clean_logits[:, -1, :])
            clean_log_props_preceeding = torch.log(torch.softmax(clean_logits[:, -2, :], dim=-1))
            clean_log_props_correct = clean_log_props_preceeding[:, correct_token]
            clean_CE = compute_CE(clean_log_props_preceeding, correct_token)

            # Run ablation experiment
            neuron_ablation_effects = {metric: [] for metric in metrics}
            
            mean_activations = get_mean_activations(model, clean_tokens, position)
            for idxs in idxs_list:
                ablated_logits = ablate_neuron(model, clean_tokens, idxs, position, mean_activations)
                # list of metrics for ablated logits
                ablated_logit_diff = get_logit_diff([ablated_logits, answer_token_indices])
                ablated_top_logit = ablated_logits[:, -1, top_logit_idx]
                ablated_log_props = torch.log(torch.softmax(ablated_logits[:, -1, :], dim=-1))
                ablated_top_log_prop = ablated_log_props[:, top_logit_idx]
                ablated_kl_div_with_clean = compute_kl_divergence(ablated_logits[:, -1, :], clean_logits[:, -1, :])
                ablated_log_props_preceeding = torch.log(torch.softmax(ablated_logits[:, -2, :], dim=-1))
                ablated_log_props_correct = ablated_log_props_preceeding[:, correct_token]
                ablated_CE = compute_CE(ablated_log_props_preceeding, correct_token)

                # compute the effects
                ablation_effects = {}
                ablation_effects['logit_diff'] = clean_logit_diff - ablated_logit_diff
                ablation_effects['top_logit'] = (clean_top_logit - ablated_top_logit).squeeze().diagonal().mean(dim=0)
                ablation_effects['top_logit_log_prop'] = (clean_top_log_prop - ablated_top_log_prop).squeeze().diagonal().mean(dim=0)
                ablation_effects['kl_div_with_clean'] = (clean_kl_div_with_clean - ablated_kl_div_with_clean).mean(dim=0)
                ablation_effects['log_props_correct'] = (clean_log_props_correct - ablated_log_props_correct).squeeze().diagonal().mean(dim=0)
                ablation_effects['CE'] = (clean_CE - ablated_CE).mean(dim=0)

                for metric in metrics:
                    neuron_ablation_effects[metric].append(ablation_effects[metric].detach().cpu().numpy())
            
        # Run attribution patching
        neuron_attribution_effects = {}
        for metric in metrics:
            neuron_attribution_effects[metric] = run_attribution_patching(
                model=model, 
                clean_tokens=clean_tokens, 
                answer_token_indices=answer_token_indices, 
                position=position, 
                mean_activations=mean_activations, 
                metric=metric, 
                clean_logits=clean_logits, 
                correct_token=correct_token, 
                top_logit_idx=top_logit_idx).detach().cpu().numpy()

        # Group attributions
        # this should be possible because we are linearizing the model
        with torch.no_grad():
            grouped_attribution_effects = {}
            for metric in metrics:
                grouped_attribution_effects[metric] = []
                for idxs in idxs_list:
                    group_score = neuron_attribution_effects[metric][idxs].sum()
                    grouped_attribution_effects[metric].append(group_score.item())

            # Store results
            results[model_name][position] = {}
            for metric in metrics:
                results[model_name][position][metric] = {
                    "neuron_ablation_effects": np.array(neuron_ablation_effects[metric]),
                    "neuron_attribution_effects": np.array(neuron_attribution_effects[metric]),
                    "grouped_attribution_effects": np.array(grouped_attribution_effects[metric]),
                    'attribution_ablation_correlation': np.corrcoef(
                        np.array(neuron_ablation_effects[metric]),
                        np.array(grouped_attribution_effects[metric])
                    )[0,1]
                }
import pickle
with open("results_ablation_vs_attrpatch_mean_ablate.pt", "wb") as f:
    pickle.dump(results, f)

#%%
with open("results_ablation_vs_attrpatch_mean_ablate.pt", "rb") as f:
    results = pickle.load(f)

figures_dir = "figures/ablation_vs_attrpatch_mean_ablate"
os.makedirs(figures_dir, exist_ok=True)

for metric in metrics:
    fig, axes = plt.subplots(12, 3, figsize=(12, 48))
    fig.suptitle(f"ablation vs (ablation - attribution) - {metric}")
    for position, ax in zip(positions, axes.flatten()):
        # Check if data exists for this position
        for model_name in ['finetuned', 'nln']:
            ax.scatter(
                torch.tensor(results[model_name][position][metric]['neuron_ablation_effects']).cpu().numpy(), 
                torch.tensor(results[model_name][position][metric]['neuron_ablation_effects']).cpu().numpy() - 
                torch.tensor(results[model_name][position][metric]['grouped_attribution_effects']).cpu().numpy(), 
                label=model_name
            )
            ax.axhline(0, color='black', linewidth=0.5)
            maximum = np.max([
                np.abs(results[model_name][position][metric]['neuron_ablation_effects']), 
                np.abs(results[model_name][position][metric]['neuron_ablation_effects'])
            ])*1.05
            minimum = -maximum
            ax.set_xlim(minimum, maximum)
            ax.set_xlabel("ablation effect")
            ax.set_ylabel("ablation effect - attribution effect")
            ax.set_title(f"{position}")
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(figures_dir, f"ablation_vs_attrpatch_mean_ablate1_{metric}.png"))
    plt.tight_layout()
    plt.show()

#%%
for position in ['blocks.11.hook_resid_post', 'blocks.3.hook_resid_post']:

    fig, ax = plt.subplots(1, len(metrics), figsize=(20, 4))
    for i, metric in enumerate(metrics):
        ax[i].scatter(results['finetuned'][position][metric]['neuron_ablation_effects'], results['finetuned'][position][metric]['grouped_attribution_effects'], label='finetuned')
        ax[i].scatter(results['nln'][position][metric]['neuron_ablation_effects'], results['nln'][position][metric]['grouped_attribution_effects'], label='nln')
        ax[i].legend()
        ax[i].set_xlabel("ablation effect")
        ax[i].set_ylabel("attribution effect")
        maximum = np.max([
                    np.abs(results[model_name][position][metric]['neuron_ablation_effects']), 
                    np.abs(results[model_name][position][metric]['neuron_ablation_effects'])
                ])*1.05
        ax[i].axvline(0, color='black', linewidth=0.5)
        ax[i].axhline(0, color='black', linewidth=0.5)
        minimum = -maximum
        ax[i].set_xlim(minimum, maximum)
        ax[i].set_ylim(minimum, maximum)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_title(metric)
    fig.suptitle(f"position: {position}", fontsize=20)
    plt.tight_layout()
    plt.show()

#%%
# Plot 2: Ablation effect vs Attribution effect

for metric in metrics:
    fig, axes = plt.subplots(12, 3, figsize=(12, 48))
    fig.suptitle(f"ablation vs (ablation - attribution) - {metric}")
    for position, ax in zip(positions, axes.flatten()):
        # Check if data exists for this position
        for model_name in ['finetuned', 'nln']:
            ax.scatter(
                torch.tensor(results[model_name][position][metric]['neuron_ablation_effects']).cpu().numpy(), 
                torch.tensor(results[model_name][position][metric]['grouped_attribution_effects']).cpu().numpy(), 
                label=model_name
            )
            ax.axhline(0, color='black', linewidth=0.5)
            maximum = np.max([
                np.abs(results[model_name][position][metric]['neuron_ablation_effects']), 
                np.abs(results[model_name][position][metric]['neuron_ablation_effects'])
            ])*1.05
            minimum = -maximum
            ax.set_xlim(minimum, maximum)
            ax.set_xlabel("ablation effect")
            ax.set_ylabel("attribution effect")
            ax.set_title(f"{position}")
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    fig.suptitle(f"ablation vs attribution - {metric}")
    plt.savefig(os.path.join(figures_dir, f"ablation_vs_attrpatch_mean_ablate1_{metric}.png"))
    plt.tight_layout()
    plt.show()
#%%
# Plot 3: Attribution ablation correlation by position type
for metric in metrics:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Attribution ablation correlation - {metric}")
    for i, ax in enumerate(axes):
        positions_subset = [p for p in positions[i::3] if p in results['finetuned'] and p in results['nln'] and 
                           all(metric in results['finetuned'][p] and metric in results['nln'][p] for model_name in ['finetuned', 'nln'])]
       
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions_subset)))
        
        for position, color in zip(positions_subset, colors):
            ax.plot([0, 1], 
                    [results['finetuned'][position][metric]['attribution_ablation_correlation'],
                        results['nln'][position][metric]['attribution_ablation_correlation']], 
                    '-*', label=position.split('.')[-2], color=color)
        ax.legend()
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["finetuned", "No LayerNorm"])
        ax.set_ylabel("attribution ablation correlation")
        ax.axhline(0, color='black', linewidth=0.5)
        if positions_subset:
            ax.set_title(f"{positions_subset[0].split('.')[-1]} - {metric}")

    plt.savefig(os.path.join(figures_dir, f"ablation_vs_attrpatch_mean_ablate3_{metric}.png"))
    plt.tight_layout()
    plt.show()


    # %%
# Plot 4: Attribution ablation l1 diff by position type
for metric in metrics:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Attribution ablation l1 diff - {metric}")
    for i, ax in enumerate(axes):   
        positions_subset = [p for p in positions[i::3] if p in results['finetuned'] and p in results['nln'] and 
                           all(metric in results['finetuned'][p] and metric in results['nln'][p] for model_name in ['finetuned', 'nln'])]
       
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions_subset)))
        
        for position, color in zip(positions_subset, colors):
            ax.plot([0, 1], 
                    [
                        np.mean(np.abs(results[model][position][metric]['neuron_ablation_effects']-
                                       results[model][position][metric]['grouped_attribution_effects'])) 
                                       for model in ['finetuned', 'nln']], '-*', label='' + position.split('.')[-2], color=color)
        ax.legend()
        ax.set_xlim(-1, 2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["finetuned", "No LayerNorm"])
        ax.set_ylabel("l1 diff")
        ax.axhline(0, color='black', linewidth=0.5)
        if positions_subset:
            ax.set_title(f"{positions_subset[0].split('.')[-1]} - {metric}")

    plt.savefig(os.path.join(figures_dir, f"ablation_vs_attrpatch_mean_ablate3_{metric}.png"))
    plt.tight_layout()
    plt.show()

#%%
for metric in metrics:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Attribution ablation l1 diff - {metric}")
    for i, ax in enumerate(axes):   
        positions_subset = [p for p in positions[i::3] if p in results['finetuned'] and p in results['nln'] and 
                           all(metric in results['finetuned'][p] and metric in results['nln'][p] for model_name in ['finetuned', 'nln'])]
       
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions_subset)))
        
        for position, color in zip(positions_subset, colors):
            ax.plot([0, 1], 
                    [
                        np.mean(np.abs(results[model][position][metric]['neuron_ablation_effects']-
                                       results[model][position][metric]['grouped_attribution_effects'])) 
                                       for model in ['finetuned', 'nln']], '-*', label='' + position.split('.')[-2], color=color)
        ax.legend()
        ax.set_xlim(-1, 2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["finetuned", "No LayerNorm"])
        ax.set_ylabel("l1 diff")
        ax.axhline(0, color='black', linewidth=0.5)
        if positions_subset:
            ax.set_title(f"{positions_subset[0].split('.')[-1]} - {metric}")

    plt.savefig(os.path.join(figures_dir, f"ablation_vs_attrpatch_mean_ablate3_{metric}.png"))
    plt.tight_layout()
    plt.show()

# %%
