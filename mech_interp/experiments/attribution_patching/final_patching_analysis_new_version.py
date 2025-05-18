#%%
import sys
sys.path.append("/workspace/removing-layer-norm/")
import torch
import numpy as np
from transformers import AutoTokenizer
import einops
import matplotlib.pyplot as plt
from neel_plotly import imshow, scatter
from ioi_utils import get_logit_diff, ioi_metric
from mech_interp.experiments.attribution_patching.attribution_patching_utils import get_cache_fwd_and_bwd, get_attr_patch_attn_head_all_pos_every, get_attr_patch_block_every 
from tqdm import tqdm
from transformer_lens import patching
from functools import partial
from neel_plotly import imshow
import pickle
import os
from transformer_lens import ActivationCache
import seaborn as sns
import pandas as pd

def pickleload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

prompts_set = 1
def l1(act, attr):
    return torch.abs(act - attr)

def get_results_dicts(model_name):
    results = {
        'resid_pre': {
            'attribution' : torch.cat(
                                        [pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{prompts_set}/new_resid_pre_attr_patch_result_{model_name}.pkl") 
                                        for prompts_set in range(0, 30)
                                        ], 
                                        dim=-1),
            'activation' : torch.cat([pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{prompts_set}/new_resid_pre_act_patch_result_{model_name}.pkl") 
                                        for prompts_set in range(0, 30)
                                        ], 
                                        dim=-1),
        },
    }   
    return results

#%%
baseline_results = get_results_dicts('baseline')
vanilla_results = get_results_dicts('vanilla')
noLN_results = get_results_dicts('noLN')
#%%
fig, axes = plt.subplots(1, 3, figsize=(12, 3)) # Increased figsize for colorbar
diff_baseline = baseline_results['resid_pre']['activation'].mean(dim=-1)[0, 11]
diff_vanilla = vanilla_results['resid_pre']['activation'].mean(dim=-1)[0, 11]
diff_noLN = noLN_results['resid_pre']['activation'].mean(dim=-1)[0, 11]
norm_act_patch_resid_pre_baseline = baseline_results['resid_pre']['activation'].mean(dim=-1)/(diff_baseline)
norm_act_patch_resid_pre_vanilla = vanilla_results['resid_pre']['activation'].mean(dim=-1)/(diff_vanilla)
norm_act_patch_resid_pre_noLN = noLN_results['resid_pre']['activation'].mean(dim=-1)/(diff_noLN)
axes[0].imshow(norm_act_patch_resid_pre_baseline.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[0].set_title('GPT-2 small: activation patching')
axes[1].imshow(norm_act_patch_resid_pre_vanilla.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[1].set_title('Vanilla finetuned model: activation patching')
axes[2].imshow(norm_act_patch_resid_pre_noLN.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[2].set_title('LN-free model: activation patching')
for ax in axes:
    ax.set_xlabel('Position')
    ax.set_ylabel('Layer')
    ax.set_xticks(np.arange(0, noLN_results['resid_pre']['activation'].mean(dim=-1).cpu().numpy().shape[1]))
    ax.set_yticks(np.arange(0, noLN_results['resid_pre']['activation'].mean(dim=-1).cpu().numpy().shape[0]))
plt.tight_layout() # Adjust layout to prevent overlap
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 3)) # Increased figsize for colorbar
norm_att_patch_resid_pre_baseline = baseline_results['resid_pre']['attribution'].mean(dim=-1)/(diff_baseline)
norm_att_patch_resid_pre_vanilla = vanilla_results['resid_pre']['attribution'].mean(dim=-1)/(diff_vanilla)
norm_att_patch_resid_pre_noLN = noLN_results['resid_pre']['attribution'].mean(dim=-1)/(diff_noLN)
axes[0].imshow(norm_att_patch_resid_pre_baseline.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[0].set_title('GPT-2 small: attribution patching')
axes[1].imshow(norm_att_patch_resid_pre_vanilla.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[1].set_title('Vanilla finetuned model: attribution patching')
axes[2].imshow(norm_att_patch_resid_pre_noLN.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[2].set_title('LN-free model: attribution patching')
for ax in axes:
    ax.set_xlabel('Position')
    ax.set_ylabel('Layer')
    ax.set_xticks(np.arange(0, noLN_results['resid_pre']['attribution'].mean(dim=-1).cpu().numpy().shape[1]))
    ax.set_yticks(np.arange(0, noLN_results['resid_pre']['attribution'].mean(dim=-1).cpu().numpy().shape[0]))
plt.tight_layout() # Adjust layout to prevent overlap
plt.show()

#%%
abs_diff_baseline = torch.abs(norm_act_patch_resid_pre_baseline - norm_att_patch_resid_pre_baseline)
abs_diff_vanilla = torch.abs(norm_act_patch_resid_pre_vanilla - norm_att_patch_resid_pre_vanilla)
abs_diff_noLN = torch.abs(norm_act_patch_resid_pre_noLN - norm_att_patch_resid_pre_noLN)
plt.plot(abs_diff_baseline.sum(dim=-1).cpu().numpy(), label='baseline')
plt.plot(abs_diff_vanilla.sum(dim=-1).cpu().numpy(), label='vanilla')
plt.plot(abs_diff_noLN.sum(dim=-1).cpu().numpy(), label='noLN')
plt.legend()
plt.show()

plt.imshow(abs_diff_baseline.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
plt.show()
plt.imshow(abs_diff_vanilla.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
plt.show()
plt.imshow(abs_diff_noLN.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
plt.show()

baseline_error = abs_diff_baseline.cpu().numpy().sum()
vanilla_error = abs_diff_vanilla.cpu().numpy().sum()
noLN_error = abs_diff_noLN.cpu().numpy().sum()

print(baseline_error, vanilla_error, noLN_error)

#%% now let's repeat the analysis to find the std across prompts
norm_factor_baseline = baseline_results['resid_pre']['activation'][0, 11]
norm_factor_vanilla = vanilla_results['resid_pre']['activation'][0, 11]
norm_factor_noLN = noLN_results['resid_pre']['activation'][0, 11]

act_patch_baseline = baseline_results['resid_pre']['activation']
act_patch_vanilla = vanilla_results['resid_pre']['activation']
act_patch_noLN = noLN_results['resid_pre']['activation']
norm_act_patch_baseline = act_patch_baseline/norm_factor_baseline
norm_act_patch_vanilla = act_patch_vanilla/norm_factor_vanilla
norm_act_patch_noLN = act_patch_noLN/norm_factor_noLN

attr_patch_baseline = baseline_results['resid_pre']['attribution']
attr_patch_vanilla = vanilla_results['resid_pre']['attribution']
attr_patch_noLN = noLN_results['resid_pre']['attribution']
norm_attr_patch_baseline = attr_patch_baseline/norm_factor_baseline
norm_attr_patch_vanilla = attr_patch_vanilla/norm_factor_vanilla
norm_attr_patch_noLN = attr_patch_noLN/norm_factor_noLN

#%%
prompts = pickleload('/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_1/prompts.pkl')
prompt = prompts['prompts'][0]

from transformers import AutoTokenizer
import sys
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import load_baseline_small

model = load_baseline_small()
prompt_tokens = model.to_tokens(prompt)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Display the full prompt
print("Full prompt:", tokenizer.decode(prompt_tokens[0]))

# Display individual tokens
print("\nIndividual tokens:")
sentence_tokens = []
for token_id in prompt_tokens[0]:
    token = tokenizer.decode([token_id])
    print(f"Token ID: {token_id}, Token: '{token}'")
    sentence_tokens.append(str(token))
#%%
norm_act_patch_baseline_mean = norm_act_patch_baseline.mean(dim=-1)
norm_act_patch_vanilla_mean = norm_act_patch_vanilla.mean(dim=-1)
norm_act_patch_noLN_mean = norm_act_patch_noLN.mean(dim=-1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3)) # Increased figsize for colorbar
fig.suptitle('Activation patching', fontsize=16, y=1.05)
axes[0].imshow(norm_act_patch_baseline_mean.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[0].set_title('GPT-2 small')
axes[1].imshow(norm_act_patch_vanilla_mean.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[1].set_title('Vanilla finetuned model')
axes[2].imshow(norm_act_patch_noLN_mean.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[2].set_title('LN-free model')
for ax in axes:
    ax.set_xlabel('Position')
    ax.set_ylabel('Layer')
    # ax.set_xticks(np.arange(0, noLN_results['resid_pre']['activation'].mean(dim=-1).cpu().numpy().shape[1]))
    for spline in ax.spines.values():
        spline.set_visible(False)
    ax.set_xticks(np.arange(0, noLN_results['resid_pre']['attribution'].mean(dim=-1).cpu().numpy().shape[1]))
    ax.set_xticklabels(sentence_tokens, rotation=90, fontsize=8, ha='center')
    ax.set_yticks(np.arange(0, noLN_results['resid_pre']['attribution'].mean(dim=-1).cpu().numpy().shape[0]))
plt.show()

norm_attr_patch_baseline_mean = norm_attr_patch_baseline.mean(dim=-1)
norm_attr_patch_vanilla_mean = norm_attr_patch_vanilla.mean(dim=-1)
norm_attr_patch_noLN_mean = norm_attr_patch_noLN.mean(dim=-1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3)) # Increased figsize for colorbar
fig.suptitle('Attribution patching', fontsize=16, y=1.05)
axes[0].imshow(norm_attr_patch_baseline_mean.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[0].set_title('GPT-2 small')
axes[1].imshow(norm_attr_patch_vanilla_mean.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[1].set_title('Vanilla finetuned model')
axes[2].imshow(norm_attr_patch_noLN_mean.cpu().numpy(), vmin=-1, vmax=1, cmap='RdBu')
axes[2].set_title('LN-free model')
for ax in axes:
    ax.set_xlabel('Position', labelpad=-10)  # Moved xlabel up by using negative labelpad
    ax.set_ylabel('Layer')
    for spline in ax.spines.values():
        spline.set_visible(False)
    ax.set_xticks(np.arange(0, noLN_results['resid_pre']['attribution'].mean(dim=-1).cpu().numpy().shape[1]))
    ax.set_xticklabels(sentence_tokens, rotation=90, fontsize=8, ha='center')
    ax.set_yticks(np.arange(0, noLN_results['resid_pre']['attribution'].mean(dim=-1).cpu().numpy().shape[0]))
plt.show()

#%%
# find the right scale

#%%
def plot_patching_heatmaps(norm_patch_baseline, norm_patch_vanilla, norm_patch_noLN, sentence_tokens, title, zmin=-1, zmax=1, save_path=None):
    """
    Creates a figure with three heatmaps for baseline, vanilla, and noLN models with a colorbar.
    
    Args:
        norm_patch_baseline: Normalized patching results for baseline model
        norm_patch_vanilla: Normalized patching results for vanilla model
        norm_patch_noLN: Normalized patching results for noLN model
        sentence_tokens: List of token strings for x-axis labels
        title: Title for the figure 
    """
    norm_patch_baseline_mean = norm_patch_baseline.mean(dim=-1)
    norm_patch_vanilla_mean = norm_patch_vanilla.mean(dim=-1)
    norm_patch_noLN_mean = norm_patch_noLN.mean(dim=-1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3))  # Increased figsize for colorbar
    fig.suptitle(title, fontsize=18, y=1.05)

    # Plot each heatmap
    axes[0].imshow(norm_patch_baseline_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    axes[0].set_title('GPT-2 small', fontsize=12)

    axes[1].imshow(norm_patch_vanilla_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    axes[1].set_title('Vanilla finetuned model', fontsize=12)

    # Store the image handle for the colorbar
    im = axes[2].imshow(norm_patch_noLN_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    axes[2].set_title('LN-free model', fontsize=12)

    # Format all axes
    for ax in axes:
        ax.set_xlabel('Position', labelpad=-20, fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        for spline in ax.spines.values():
            spline.set_visible(False)
        ax.set_xticks(np.arange(0, len(sentence_tokens)))
        ax.set_xticklabels(sentence_tokens, rotation=90, ha='center')
        ax.set_yticks(np.arange(0, norm_patch_baseline_mean.shape[0]))
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Add colorbar to the right
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation='vertical',
        fraction=0.02,
        pad=0.04,
        shrink=1.5,        # Shortens the colorbar
        aspect=10,         # Controls thickness (smaller = thicker)
    )
    cbar.ax.set_yticks([-1, 0, 1])
    cbar.ax.tick_params(axis='both', which='major', labelsize=10)
    cbar.ax.set_ylabel('normalized logit diff', fontsize=10)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True,  bbox_inches = "tight")
    plt.show()

# Plot activation patching results
plot_patching_heatmaps(
    norm_act_patch_baseline, 
    norm_act_patch_vanilla, 
    norm_act_patch_noLN, 
    sentence_tokens, 
    'Activation patching',
    save_path='activation_patching_heatmap_from-1to1.png'
)

# Plot attribution patching results
plot_patching_heatmaps(
    norm_attr_patch_baseline, 
    norm_attr_patch_vanilla, 
    norm_attr_patch_noLN, 
    sentence_tokens, 
    'Attribution patching',
    save_path='attribution_patching_heatmap_from-1to1.png'
)

minmax_value = torch.max(torch.abs(torch.stack([
    norm_act_patch_baseline.mean(dim=-1), 
    norm_attr_patch_baseline.mean(dim=-1),
    norm_act_patch_vanilla.mean(dim=-1), 
    norm_attr_patch_vanilla.mean(dim=-1),
    norm_act_patch_noLN.mean(dim=-1), 
    norm_attr_patch_noLN.mean(dim=-1),
])))

# Plot activation patching results
plot_patching_heatmaps(
    norm_act_patch_baseline, 
    norm_act_patch_vanilla, 
    norm_act_patch_noLN, 
    sentence_tokens, 
    'Activation patching',
    zmin=-minmax_value, 
    zmax=minmax_value,
    save_path='activation_patching_heatmap_fixed_minmax.png'
)

# Plot attribution patching results
plot_patching_heatmaps(
    norm_attr_patch_baseline, 
    norm_attr_patch_vanilla, 
    norm_attr_patch_noLN, 
    sentence_tokens, 
    'Attribution patching',
    zmin=-minmax_value, 
    zmax=minmax_value,  
    save_path='attribution_patching_heatmap_fixed_minmax.png'
)

#%%
def plot_patching_heatmaps_combined(norm_patch_baseline, norm_patch_vanilla, norm_patch_noLN, norm_attr_patch_baseline, norm_attr_patch_vanilla, norm_attr_patch_noLN, sentence_tokens, title, zmin=-1, zmax=1, save_path=None):
    """
    Creates a figure with six heatmaps arranged in 2 rows of 3 columns for baseline, vanilla, and noLN models with a colorbar.
    
    Args:
        norm_patch_baseline: Normalized patching results for baseline model
        norm_patch_vanilla: Normalized patching results for vanilla model
        norm_patch_noLN: Normalized patching results for noLN model
        sentence_tokens: List of token strings for x-axis labels
        title: Title for the figure
    """
    # Calculate means for activation patching
    norm_patch_baseline_mean = norm_patch_baseline.mean(dim=-1)
    norm_patch_vanilla_mean = norm_patch_vanilla.mean(dim=-1)
    norm_patch_noLN_mean = norm_patch_noLN.mean(dim=-1)
    norm_attr_patch_baseline_mean = norm_attr_patch_baseline.mean(dim=-1)
    norm_attr_patch_vanilla_mean = norm_attr_patch_vanilla.mean(dim=-1)
    norm_attr_patch_noLN_mean = norm_attr_patch_noLN.mean(dim=-1)

    # Create a figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)

    # Plot each heatmap in the first row (activation patching)
    axes[0, 0].imshow(norm_patch_baseline_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    # axes[0, 0].set_title('GPT-2 small', fontsize=12)

    axes[0, 1].imshow(norm_patch_vanilla_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    # axes[0, 1].set_title('Vanilla finetuned model', fontsize=12)

    im = axes[0, 2].imshow(norm_patch_noLN_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    # axes[0, 2].set_title('LN-free model', fontsize=12)

    # Plot each heatmap in the second row (attribution patching)
    axes[1, 0].imshow(norm_attr_patch_baseline_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    # axes[1, 0].set_title('GPT-2 small', fontsize=12)

    axes[1, 1].imshow(norm_attr_patch_vanilla_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    # axes[1, 1].set_title('Vanilla finetuned model', fontsize=12)

    im = axes[1, 2].imshow(norm_attr_patch_noLN_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    # axes[1, 2].set_title('LN-free model', fontsize=12)

    model_names = ['GPT-2 Small', 'GPT-2 Small vanilla finetuned', 'GPT-2 Small LN-free finetuned']
    # Format all axes
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == 1:
                ax.set_xlabel('Position', labelpad=-20, fontsize=14)
            if j == 0:
                ax.set_ylabel('Layer', fontsize=14)
            for spline in ax.spines.values():
                spline.set_visible(False)
            ax.set_xticks(np.arange(0, len(sentence_tokens)))
            ax.set_xticklabels(sentence_tokens, rotation=90, ha='center')
            ax.set_yticks(np.arange(0, norm_patch_baseline_mean.shape[0]))
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            if i == 0:
                ax.set_title(f'Activation patching on \n{model_names[j]}', fontsize=14)
            else:
                ax.set_title(f'Attribution patching on \n{model_names[j]}', fontsize=14)

    # Add row titles
    # fig.text(0.05, 0.65, 'Activation Patching', fontsize=14, rotation=90, ha='center')
    # fig.text(0.05, 0.15, 'Attribution Patching', fontsize=14, rotation=90, ha='center')

    # Add colorbar to the right
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation='vertical',
        fraction=0.020,
        pad=0.04,
        shrink=1,
        aspect=10,
    )
    cbar.ax.set_yticks([-1, 0, 1])
    cbar.ax.tick_params(axis='both', which='major', labelsize=14)
    cbar.ax.set_ylabel('normalized \nlogit difference', fontsize=14)
    
    # Adjust layout
    # plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


# Plot combined activation and attribution patching results
plot_patching_heatmaps_combined(
    norm_act_patch_baseline, 
    norm_act_patch_vanilla, 
    norm_act_patch_noLN, 
    norm_attr_patch_baseline, 
    norm_attr_patch_vanilla, 
    norm_attr_patch_noLN, 
    sentence_tokens, 
    'Activation and Attribution Patching',
    zmin=-minmax_value, 
    zmax=minmax_value,
    save_path='combined_patching_heatmap.png'
)
#%%
def plot_patching_heatmaps(norm_patch_baseline, norm_patch_vanilla, norm_patch_noLN, sentence_tokens, title, zmin=-1, zmax=1, save_path=None):
    """
    Creates a figure with three heatmaps for baseline, vanilla, and noLN models with a colorbar.
    
    Args:
        norm_patch_baseline: Normalized patching results for baseline model
        norm_patch_vanilla: Normalized patching results for vanilla model
        norm_patch_noLN: Normalized patching results for noLN model
        sentence_tokens: List of token strings for x-axis labels
        title: Title for the figure 
    """
    norm_patch_baseline_mean = norm_patch_baseline.mean(dim=-1)
    norm_patch_vanilla_mean = norm_patch_vanilla.mean(dim=-1)
    norm_patch_noLN_mean = norm_patch_noLN.mean(dim=-1)
    fontsize = 16

    fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharex=True, sharey=True)  # Increased figsize for colorbar
    fig.suptitle(title, fontsize=18, y=1.05)

    # Plot each heatmap
    axes[0].imshow(norm_patch_baseline_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    axes[0].set_title('GPT-2 Small', fontsize=fontsize)

    axes[1].imshow(norm_patch_vanilla_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    axes[1].set_title('GPSTVanilla finetuned model', fontsize=fontsize)

    # Store the image handle for the colorbar
    im = axes[2].imshow(norm_patch_noLN_mean.cpu().numpy(), vmin=zmin, vmax=zmax, cmap='RdBu')
    axes[2].set_title('LN-free model', fontsize=fontsize)

    # Format all axes
    for i, ax in enumerate(axes)    :
        ax.set_xlabel('Position', labelpad=-20, fontsize=fontsize)
        if i == 0:
            ax.set_ylabel('Layer', fontsize=fontsize)
        for spline in ax.spines.values():
            spline.set_visible(False)
        ax.set_xticks(np.arange(0, len(sentence_tokens)))
        ax.set_xticklabels(sentence_tokens, rotation=90, ha='center')
        ax.set_yticks(np.arange(0, norm_patch_baseline_mean.shape[0]))
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Add colorbar to the right
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation='vertical',
        fraction=0.02,
        pad=0.04,
        shrink=1.5,        # Shortens the colorbar
        aspect=10,         # Controls thickness (smaller = thicker)
    )
    cbar.ax.set_yticks([-1, 0, 1])
    cbar.ax.tick_params(axis='both', which='major', labelsize=10)
    cbar.ax.set_ylabel('normalized\n logit difference', fontsize=10)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True,  bbox_inches = "tight")
    plt.show()

# Plot activation patching results
plot_patching_heatmaps(
    norm_act_patch_baseline, 
    norm_act_patch_vanilla, 
    norm_act_patch_noLN, 
    sentence_tokens, 
    'Activation patching',
    zmin=-minmax_value, 
    zmax=minmax_value,
    save_path='activation_patching_heatmap_fixed_minmax.png'
)



#%% option 1 
norm_diff_baseline = torch.abs(norm_act_patch_baseline - norm_attr_patch_baseline)
norm_diff_vanilla = torch.abs(norm_act_patch_vanilla - norm_attr_patch_vanilla)
norm_diff_noLN = torch.abs(norm_act_patch_noLN - norm_attr_patch_noLN)
norm_diff_baseline_mean = norm_diff_baseline.sum(dim=1).mean(dim=-1)
norm_diff_vanilla_mean = norm_diff_vanilla.sum(dim=1).mean(dim=-1)
norm_diff_noLN_mean = norm_diff_noLN.sum(dim=1).mean(dim=-1)

norm_diff_baseline_std = norm_diff_baseline.std(dim=1).mean(dim=-1)
norm_diff_vanilla_std = norm_diff_vanilla.std(dim=-1).mean(dim=-1)
norm_diff_noLN_std = norm_diff_noLN.std(dim=-1).mean(dim=-1)

norm_diff_baseline_sem = norm_diff_baseline_std / torch.sqrt(torch.tensor(norm_diff_baseline.shape[0]))
norm_diff_vanilla_sem = norm_diff_vanilla_std / torch.sqrt(torch.tensor(norm_diff_vanilla.shape[0]))
norm_diff_noLN_sem = norm_diff_noLN_std / torch.sqrt(torch.tensor(norm_diff_noLN.shape[0]))

plt.plot(norm_diff_baseline_mean.cpu().numpy(), label='baseline')
plt.plot(norm_diff_vanilla_mean.cpu().numpy(), label='vanilla')
plt.plot(norm_diff_noLN_mean.cpu().numpy(), label='noLN')
plt.legend()
plt.show()

#%% option 2 
plt.imshow(norm_attr_patch_baseline.mean(dim=-1).cpu().numpy()/480, vmin=-1, vmax=1, cmap='RdBu')
plt.show()
plt.imshow(norm_attr_patch_vanilla.std(dim=-1).cpu().numpy()/480, vmin=-1, vmax=1, cmap='RdBu')
plt.show()
plt.imshow(norm_attr_patch_noLN.std(dim=-1).cpu().numpy()/480, vmin=-1, vmax=1, cmap='RdBu')
plt.show()


#%%
# def get_l1(results, average_over_prompts=False, sum_over_positions=False, return_index=None, return_variance=False, component='every_block'):
#     act_result = results[component]['activation']
#     attr_result = results[component]['attribution']
    
#     # Calculate L1 for each prompt
#     l1_results = l1(act_result, attr_result)
    
#     if sum_over_positions == True:
#         l1_results = l1_results.sum(dim=-1)
    
#     if average_over_prompts == True:
#         if return_variance:
#             # Calculate mean and variance across prompts
#             l1_mean = l1_results.mean(dim=0)
#             l1_var = l1_results.var(dim=0)
#             if return_index is not None:
#                 return l1_mean[return_index], l1_var[return_index]
#             return l1_mean, l1_var
#         else:
#             # Just calculate mean
#             l1_mean = l1_results.mean(dim=0)
#             if return_index is not None:
#                 return l1_mean[return_index]
#             return l1_mean
    
#     if return_index is not None:
#         return l1_results[return_index]
#     return l1_results


# finetuned_results = get_results_dicts('finetuned')
# nln_results = get_results_dicts('nln')
# baseline_results = get_results_dicts('baseline')
# noLN_results = get_results_dicts('noLN')
# vanilla_results = get_results_dicts('vanilla')
# #%% # residual stream
# index = 0
# average_over_prompts = True     
# sum_over_positions = False
# return_variance = True

# # Get both mean and variance
# baseline_l1, baseline_var = get_l1(baseline_results, average_over_prompts=average_over_prompts, 
#                                   sum_over_positions=sum_over_positions, return_index=index, 
#                                   return_variance=return_variance)
# finetuned_l1, finetuned_var = get_l1(finetuned_results, average_over_prompts=average_over_prompts, 
#                                     sum_over_positions=sum_over_positions, return_index=index, 
#                                     return_variance=return_variance)
# nln_l1, nln_var = get_l1(nln_results, average_over_prompts=average_over_prompts, 
#                         sum_over_positions=sum_over_positions, return_index=index, 
#                         return_variance=return_variance)

# noLN_l1, noLN_var = get_l1(noLN_results, average_over_prompts=average_over_prompts, 
#                         sum_over_positions=sum_over_positions, return_index=index, 
#                         return_variance=return_variance)
# vanilla_l1, vanilla_var = get_l1(vanilla_results, average_over_prompts=average_over_prompts, 
#                         sum_over_positions=sum_over_positions, return_index=index, 
#                         return_variance=return_variance)

# # Convert tensors to numpy arrays for plotting
# baseline_mean = baseline_l1.sum(dim=-1).cpu().numpy()
# baseline_std = torch.sqrt(baseline_var.sum(dim=-1)).cpu().numpy()
# baseline_sem = baseline_std / np.sqrt(baseline_results['every_block']['activation'].shape[0])
# finetuned_mean = finetuned_l1.sum(dim=-1).cpu().numpy()
# finetuned_std = torch.sqrt(finetuned_var.sum(dim=-1)).cpu().numpy()
# finetuned_sem = finetuned_std / np.sqrt(finetuned_results['every_block']['activation'].shape[0])
# nln_mean = nln_l1.sum(dim=-1).cpu().numpy()
# nln_std = torch.sqrt(nln_var.sum(dim=-1)).cpu().numpy()
# nln_sem = nln_std / np.sqrt(nln_results['every_block']['activation'].shape[0])
# noLN_mean = noLN_l1.sum(dim=-1).cpu().numpy()
# noLN_std = torch.sqrt(noLN_var.sum(dim=-1)).cpu().numpy()
# noLN_sem = noLN_std / np.sqrt(noLN_results['every_block']['activation'].shape[0])
# vanilla_mean = vanilla_l1.sum(dim=-1).cpu().numpy()
# vanilla_std = torch.sqrt(vanilla_var.sum(dim=-1)).cpu().numpy()
# vanilla_sem = vanilla_std / np.sqrt(vanilla_results['every_block']['activation'].shape[0])

# # Create x-axis values (layer indices)
# x = np.arange(len(baseline_mean))

# # Create data for each model type with std
# data_std = []
# for i in range(len(x)):
#     # Baseline data
#     data_std.append({'Layer': i, 'L1 Loss': baseline_mean[i], 
#                 'Lower': baseline_mean[i] - baseline_std[i],
#                 'Upper': baseline_mean[i] + baseline_std[i], 
#                 'Model': 'baseline'})
#     # Finetuned data
#     data_std.append({'Layer': i, 'L1 Loss': finetuned_mean[i], 
#                 'Lower': finetuned_mean[i] - finetuned_std[i],
#                 'Upper': finetuned_mean[i] + finetuned_std[i], 
#                 'Model': 'finetuned'})
#     # NLN data
#     data_std.append({'Layer': i, 'L1 Loss': nln_mean[i], 
#                 'Lower': nln_mean[i] - nln_std[i],
#                 'Upper': nln_mean[i] + nln_std[i], 
#                 'Model': 'nln'})
#     # noLN data
#     data_std.append({'Layer': i, 'L1 Loss': noLN_mean[i], 
#                 'Lower': noLN_mean[i] - noLN_std[i],
#                 'Upper': noLN_mean[i] + noLN_std[i], 
#                 'Model': 'noLN'})
#     # vanilla data
#     data_std.append({'Layer': i, 'L1 Loss': vanilla_mean[i], 
#                 'Lower': vanilla_mean[i] - vanilla_std[i],
#                 'Upper': vanilla_mean[i] + vanilla_std[i], 
#                 'Model': 'vanilla'})

# df_std = pd.DataFrame(data_std)

# # Plot using seaborn with area (fill_between) - STD version
# plt.figure(figsize=(10, 6))
# for model, color in zip(['baseline', 'finetuned', 'nln', 'noLN', 'vanilla'], ['blue', 'orange', 'green', 'red', 'purple']):
#     model_data = df_std[df_std['Model'] == model]
#     sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model, color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)

# plt.xlabel('Layer')
# plt.ylabel('L1 Loss (summed over positions)')
# plt.title('L1 Loss of Activation vs Attribution Patching (±1 std dev)')
# plt.legend(title='Model')
# plt.show()

# # Create data for each model type with SEM
# data_sem = []
# for i in range(len(x)):
#     # Baseline data
#     data_sem.append({'Layer': i, 'L1 Loss': baseline_mean[i], 
#                 'Lower': baseline_mean[i] - baseline_sem[i],
#                 'Upper': baseline_mean[i] + baseline_sem[i], 
#                 'Model': 'baseline'})
#     # Finetuned data
#     data_sem.append({'Layer': i, 'L1 Loss': finetuned_mean[i], 
#                 'Lower': finetuned_mean[i] - finetuned_sem[i],
#                 'Upper': finetuned_mean[i] + finetuned_sem[i], 
#                 'Model': 'finetuned'})
#     # NLN data
#     data_sem.append({'Layer': i, 'L1 Loss': nln_mean[i], 
#                 'Lower': nln_mean[i] - nln_sem[i],
#                 'Upper': nln_mean[i] + nln_sem[i], 
#                 'Model': 'nln'})
#     # noLN data
#     data_sem.append({'Layer': i, 'L1 Loss': noLN_mean[i], 
#                 'Lower': noLN_mean[i] - noLN_sem[i],
#                 'Upper': noLN_mean[i] + noLN_sem[i], 
#                 'Model': 'noLN'})
#     # vanilla data
#     data_sem.append({'Layer': i, 'L1 Loss': vanilla_mean[i], 
#                 'Lower': vanilla_mean[i] - vanilla_sem[i],
#                 'Upper': vanilla_mean[i] + vanilla_sem[i], 
#                 'Model': 'vanilla'})

# df_sem = pd.DataFrame(data_sem)
# # Plot using seaborn with area (fill_between) - SEM version
# plt.figure(figsize=(10, 6))
# for model, color in zip(['baseline', 'finetuned', 'nln', 'noLN', 'vanilla'], ['blue', 'orange', 'green', 'red', 'purple']):
#     model_data = df_sem[df_sem['Model'] == model]
#     sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model, color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)
    
# # Increase font size
# plt.xlabel('Layer', fontsize=14)
# plt.ylabel('Absolute difference (summed over positions)', fontsize=14)
# plt.title('Absolute difference between activation and attribution patching (±1 SEM)', fontsize=16)
# plt.legend(title='Model', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()


# plt.figure(figsize=(10, 6))
# for model, color in zip(['baseline', 'vanilla', 'noLN'], ['blue', 'orange', 'green' ]):
#     model_data = df_sem[df_sem['Model'] == model]
#     sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model, color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)
    
# # Increase font size
# plt.xlabel('Layer', fontsize=14)
# plt.ylabel('Absolute difference (summed over positions)', fontsize=14)
# plt.title('Absolute difference between activation and attribution patching (±1 SEM)', fontsize=16)
# plt.legend(title='Model', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# plt.figure(figsize=(10, 6))
# for model, color in zip(['baseline', 'finetuned', 'nln'], ['blue', 'orange', 'green']):
#     model_data = df_sem[df_sem['Model'] == model]
#     sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model, color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)
    
# # Increase font size
# plt.xlabel('Layer', fontsize=14)
# plt.ylabel('Absolute difference (summed over positions)', fontsize=14)
# plt.title('Absolute difference between activation and attribution patching (±1 SEM)', fontsize=16)
# plt.legend(title='Model', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# #%% # for the poster
# imshow(
#     [finetuned_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy(),
#     noLN_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["model with LN", "LN-free model"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1, 
#     title="Activation Patching (residual stream (pre block))",
#     )

# imshow(
#     [finetuned_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy(),
#     noLN_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["model with LN", "LN-free model"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1, 
#     title="Attribution Patching (residual stream (pre block))"
#     )

# plt.figure(figsize=(10, 6))
# for model, color in zip(['finetuned',  'noLN'], ['blue', 'red']):
#     model_data = df_sem[df_sem['Model'] == model]
#     model_name = 'LN-free model' if model == 'noLN' else 'Model with LN'
#     ax = sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model_name + ' (±1 SEM)', color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)
    
# # Increase font size
# ax.set_xlabel('Layer', fontsize=14)
# ax.set_ylabel('Absolute difference (summed over positions) ', fontsize=14)
# ax.set_title('Activation vs attribution patching \non the residual stream (resid_pre)', fontsize=16)
# ax.legend(fontsize=12)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.show()
# #%% find numbers for abstract
# diff_finetuned = finetuned_results['every_block']['activation'].mean(dim=0)[0] - finetuned_results['every_block']['attribution'].mean(dim=0)[0]
# diff_nln = nln_results['every_block']['activation'].mean(dim=0)[0] - nln_results['every_block']['attribution'].mean(dim=0)[0]


# print(torch.corrcoef(torch.stack([finetuned_results['every_block']['activation'].mean(dim=0)[0].flatten(), finetuned_results['every_block']['attribution'].mean(dim=0)[0].flatten()])))
# print(torch.corrcoef(torch.stack([nln_results['every_block']['activation'].mean(dim=0)[0].flatten(), nln_results['every_block']['attribution'].mean(dim=0)[0].flatten()])))

# print(torch.corrcoef(torch.stack([vanilla_results['every_block']['activation'].mean(dim=0)[0].flatten(), vanilla_results['every_block']['attribution'].mean(dim=0)[0].flatten()])))
# print(torch.corrcoef(torch.stack([noLN_results['every_block']['activation'].mean(dim=0)[0].flatten(), noLN_results['every_block']['attribution'].mean(dim=0)[0].flatten()])))


# # abs_diff_finetuned = torch.abs(finetuned_results['every_block']['activation'].mean(dim=0)[0] - finetuned_results['every_block']['attribution'].mean(dim=0)[0]).sum(dim=-1).cpu().numpy()
# # abs_diff_vanilla = torch.abs(vanilla_results['every_block']['activation'].mean(dim=0)[0] - vanilla_results['every_block']['attribution'].mean(dim=0)[0]).sum(dim=-1).cpu().numpy()
# # abs_diff_nln = torch.abs(nln_results['every_block']['activation'].mean(dim=0)[0] - nln_results['every_block']['attribution'].mean(dim=0)[0]).sum(dim=-1).cpu().numpy()
# # abs_diff_noLN = torch.abs(noLN_results['every_block']['activation'].mean(dim=0)[0] - noLN_results['every_block']['attribution'].mean(dim=0)[0]).sum(dim=-1).cpu().numpy()

# # old_abs_diff_distribution = abs_diff_finetuned-abs_diff_nln
# # new_abs_diff_distribution = abs_diff_vanilla-abs_diff_noLN

# # plt.hist(old_abs_diff_distribution, label='old', alpha=0.3)
# # plt.hist(new_abs_diff_distribution, label='new',  alpha=0.3)
# # plt.axvline(old_abs_diff_distribution.mean(), color='blue', linestyle='--', label='old mean')
# # plt.axvline(new_abs_diff_distribution.mean(), color='orange', linestyle='--', label='new mean')
# # plt.plot([old_abs_diff_distribution.mean() - old_abs_diff_distribution.std(), old_abs_diff_distribution.mean() + old_abs_diff_distribution.std()], [2.5, 2.5], color='blue', linewidth=1)
# # plt.plot([new_abs_diff_distribution.mean() - new_abs_diff_distribution.std(), new_abs_diff_distribution.mean() + new_abs_diff_distribution.std()], [2.8, 2.8], color='orange', linewidth=1)
# # plt.axvline(0, linestyle='-', color='black', linewidth=.5)
# # plt.legend()
# # plt.show()
# #%%
# diff_finetuned = finetuned_results['every_block']['activation'].mean(dim=0)[0] - finetuned_results['every_block']['attribution'].mean(dim=0)[0]
# diff_nln = nln_results['every_block']['activation'].mean(dim=0)[0] - nln_results['every_block']['attribution'].mean(dim=0)[0]

# abs_diff_finetuned_normalized = torch.abs(diff_finetuned).sum(dim=-1)/torch.abs(diff_finetuned).sum(dim=-1)
# abs_diff_nln_normalized = torch.abs(diff_nln).sum(dim=-1)/torch.abs(diff_finetuned).sum(dim=-1)
# relative_error_nln  = 1-abs_diff_nln_normalized
# mean_nln = relative_error_nln.mean().cpu().numpy()
# std_nln = relative_error_nln.std().cpu().numpy()
# plt.plot([mean_nln - std_nln, mean_nln + std_nln], [3.5, 3.5], color='blue', linewidth=1)
# plt.hist(relative_error_nln.cpu().numpy(), label='nln', alpha=0.3)
# plt.axvline(mean_nln, color='blue', linestyle='--', label='nln mean')


# diff_vanilla = vanilla_results['every_block']['activation'].mean(dim=0)[0] - vanilla_results['every_block']['attribution'].mean(dim=0)[0]
# diff_noLN = noLN_results['every_block']['activation'].mean(dim=0)[0] - noLN_results['every_block']['attribution'].mean(dim=0)[0]
# abs_diff_vanilla_normalized = torch.abs(diff_vanilla).sum(dim=-1)/torch.abs(diff_vanilla).sum(dim=-1)
# abs_diff_noLN_normalized = torch.abs(diff_noLN).sum(dim=-1)/torch.abs(diff_vanilla).sum(dim=-1)
# relative_error_noLN  = 1-abs_diff_noLN_normalized
# mean_noLN = relative_error_noLN.mean().cpu().numpy()
# std_noLN = relative_error_noLN.std().cpu().numpy()
# plt.plot([mean_noLN - std_noLN, mean_noLN + std_noLN], [3.6, 3.6], color='orange', linewidth=1)
# plt.hist(relative_error_noLN.cpu().numpy(), label='noLN', alpha=0.3)
# plt.axvline(mean_noLN, color='orange', linestyle='--', label='noLN mean')
# plt.legend()
# plt.show()
# #%%
# # compare 
# imshow(
#     [baseline_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy(),
#     finetuned_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy(),
#     nln_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1, 
#     title="Activation Patching (residual stream (pre block))"
#     )
# imshow(
#     [baseline_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy(),
#     finetuned_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy(),
#     nln_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1,
#     title="Attribution Patching (residual stream (pre block))"
#     )
# imshow(
#     [baseline_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy() - baseline_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy(),
#     finetuned_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy() - finetuned_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy(),
#     nln_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy() - nln_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1,
#     title="Activation - Attribution (residual stream (pre block))"
#     )

# imshow(
#     [baseline_results['every_block']['activation'].mean(dim=0)[1].cpu().numpy(),
#     finetuned_results['every_block']['activation'].mean(dim=0)[1].cpu().numpy(),
#     nln_results['every_block']['activation'].mean(dim=0)[1].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1, 
#     title="Activation Patching (Attention output)"
#     )
# imshow(
#     [baseline_results['every_block']['attribution'].mean(dim=0)[1].cpu().numpy(),
#     finetuned_results['every_block']['attribution'].mean(dim=0)[1].cpu().numpy(),
#     nln_results['every_block']['attribution'].mean(dim=0)[1].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1,
#     title="Attribution Patching (Attention output)"
#     )
# imshow(
#     [baseline_results['every_block']['activation'].mean(dim=0)[1].cpu().numpy() - baseline_results['every_block']['attribution'].mean(dim=0)[1].cpu().numpy(),
#     finetuned_results['every_block']['activation'].mean(dim=0)[1].cpu().numpy() - finetuned_results['every_block']['attribution'].mean(dim=0)[1].cpu().numpy(),
#     nln_results['every_block']['activation'].mean(dim=0)[1].cpu().numpy() - nln_results['every_block']['attribution'].mean(dim=0)[1].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1,
#     title="Activation - Attribution (Attention output)"
#     )

# imshow(
#     [baseline_results['every_block']['activation'].mean(dim=0)[2].cpu().numpy(),
#     finetuned_results['every_block']['activation'].mean(dim=0)[2].cpu().numpy(),
#     nln_results['every_block']['activation'].mean(dim=0)[2].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1, 
#     title="Activation Patching (Attention output)"
#     )
# imshow(
#     [baseline_results['every_block']['attribution'].mean(dim=0)[2].cpu().numpy(),
#     finetuned_results['every_block']['attribution'].mean(dim=0)[2].cpu().numpy(),
#     nln_results['every_block']['attribution'].mean(dim=0)[2].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1,
#     title="Attribution Patching (Attention output)"
#     )
# imshow(
#     [baseline_results['every_block']['activation'].mean(dim=0)[2].cpu().numpy() - baseline_results['every_block']['attribution'].mean(dim=0)[2].cpu().numpy(),
#     finetuned_results['every_block']['activation'].mean(dim=0)[2].cpu().numpy() - finetuned_results['every_block']['attribution'].mean(dim=0)[2].cpu().numpy(),
#     nln_results['every_block']['activation'].mean(dim=0)[2].cpu().numpy() - nln_results['every_block']['attribution'].mean(dim=0)[2].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1,
#     title="Activation - Attribution (Attention output)"
#     )

# #%%
# imshow(
#     [baseline_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy(),
#     vanilla_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy(),
#     noLN_results['every_block']['activation'].mean(dim=0)[0].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "vanilla", "noLN"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1, 
#     title="Activation Patching (residual stream (pre block))"
#     )
# imshow(
#     [baseline_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy(),
#     vanilla_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy(),
#     noLN_results['every_block']['attribution'].mean(dim=0)[0].cpu().numpy()],
#     facet_col=0,
#     facet_labels=["baseline", "vanilla", "noLN"],
#     xaxis='Pos',
#     yaxis='Layer',
#     zmin=-1,
#     zmax=1,
#     title="Attribution Patching (residual stream (pre block))"
#     )

# #%% attention out
# index = 1
# average_over_prompts = True     
# sum_over_positions = False
# return_variance = True

# # Get both mean and variance
# baseline_l1, baseline_var = get_l1(baseline_results, average_over_prompts=average_over_prompts, 
#                                   sum_over_positions=sum_over_positions, return_index=index, 
#                                   return_variance=return_variance)
# finetuned_l1, finetuned_var = get_l1(finetuned_results, average_over_prompts=average_over_prompts, 
#                                     sum_over_positions=sum_over_positions, return_index=index, 
#                                     return_variance=return_variance)
# nln_l1, nln_var = get_l1(nln_results, average_over_prompts=average_over_prompts, 
#                         sum_over_positions=sum_over_positions, return_index=index, 
#                         return_variance=return_variance)
# noLN_l1, noLN_var = get_l1(noLN_results, average_over_prompts=average_over_prompts, 
#                         sum_over_positions=sum_over_positions, return_index=index, 
#                         return_variance=return_variance)

# # Convert tensors to numpy arrays for plotting
# baseline_mean = baseline_l1.sum(dim=-1).cpu().numpy()
# baseline_std = torch.sqrt(baseline_var.sum(dim=-1)).cpu().numpy()
# baseline_sem = baseline_std / np.sqrt(baseline_results['every_block']['activation'].shape[0])
# finetuned_mean = finetuned_l1.sum(dim=-1).cpu().numpy()
# finetuned_std = torch.sqrt(finetuned_var.sum(dim=-1)).cpu().numpy()
# finetuned_sem = finetuned_std / np.sqrt(finetuned_results['every_block']['activation'].shape[0])
# nln_mean = nln_l1.sum(dim=-1).cpu().numpy()
# nln_std = torch.sqrt(nln_var.sum(dim=-1)).cpu().numpy()
# nln_sem = nln_std / np.sqrt(nln_results['every_block']['activation'].shape[0])
# noLN_mean = noLN_l1.sum(dim=-1).cpu().numpy()
# noLN_std = torch.sqrt(noLN_var.sum(dim=-1)).cpu().numpy()
# noLN_sem = noLN_std / np.sqrt(noLN_results['every_block']['activation'].shape[0])

# # Create x-axis values (layer indices)
# x = np.arange(len(baseline_mean))

# # Create data for each model type with std
# data_std = []
# for i in range(len(x)):
#     # Baseline data
#     data_std.append({'Layer': i, 'L1 Loss': baseline_mean[i], 
#                 'Lower': baseline_mean[i] - baseline_std[i],
#                 'Upper': baseline_mean[i] + baseline_std[i], 
#                 'Model': 'baseline'})
#     # Finetuned data
#     data_std.append({'Layer': i, 'L1 Loss': finetuned_mean[i], 
#                 'Lower': finetuned_mean[i] - finetuned_std[i],
#                 'Upper': finetuned_mean[i] + finetuned_std[i], 
#                 'Model': 'finetuned'})
#     # NLN data
#     data_std.append({'Layer': i, 'L1 Loss': nln_mean[i], 
#                 'Lower': nln_mean[i] - nln_std[i],
#                 'Upper': nln_mean[i] + nln_std[i], 
#                 'Model': 'nln'})
#     # noLN data
#     data_std.append({'Layer': i, 'L1 Loss': noLN_mean[i], 
#                 'Lower': noLN_mean[i] - noLN_std[i],
#                 'Upper': noLN_mean[i] + noLN_std[i], 
#                 'Model': 'noLN'})

# df_std = pd.DataFrame(data_std)

# # Plot using seaborn with area (fill_between) - STD version
# plt.figure(figsize=(10, 6))
# for model, color in zip(['baseline', 'finetuned', 'nln', 'noLN'], ['blue', 'orange', 'green', 'red']):
#     model_data = df_std[df_std['Model'] == model]
#     sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model, color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)

# plt.xlabel('Layer')
# plt.ylabel('L1 Loss (summed over positions)')
# plt.title('L1 Loss of Activation vs Attribution Patching (±1 std dev)')
# plt.legend(title='Model')
# plt.show()

# # Create data for each model type with SEM
# data_sem = []
# for i in range(len(x)):
#     # Baseline data
#     data_sem.append({'Layer': i, 'L1 Loss': baseline_mean[i], 
#                 'Lower': baseline_mean[i] - baseline_sem[i],
#                 'Upper': baseline_mean[i] + baseline_sem[i], 
#                 'Model': 'baseline'})
#     # Finetuned data
#     data_sem.append({'Layer': i, 'L1 Loss': finetuned_mean[i], 
#                 'Lower': finetuned_mean[i] - finetuned_sem[i],
#                 'Upper': finetuned_mean[i] + finetuned_sem[i], 
#                 'Model': 'finetuned'})
#     # NLN data
#     data_sem.append({'Layer': i, 'L1 Loss': nln_mean[i], 
#                 'Lower': nln_mean[i] - nln_sem[i],
#                 'Upper': nln_mean[i] + nln_sem[i], 
#                 'Model': 'nln'})
#     # noLN data
#     data_sem.append({'Layer': i, 'L1 Loss': noLN_mean[i], 
#                 'Lower': noLN_mean[i] - noLN_sem[i],
#                 'Upper': noLN_mean[i] + noLN_sem[i], 
#                 'Model': 'noLN'})

# df_sem = pd.DataFrame(data_sem)

# # Plot using seaborn with area (fill_between) - SEM version
# plt.figure(figsize=(10, 6))
# for model, color in zip(['baseline', 'finetuned', 'nln', 'noLN'], ['blue', 'orange', 'green', 'red']):
#     model_data = df_sem[df_sem['Model'] == model]
#     sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model, color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)

# plt.xlabel('Layer')
# plt.ylim(0, 1)
# plt.ylabel('Absolute difference (summed over positions)')
# plt.title('Absolute difference between activation and attribution patching (±1 SEM)')
# plt.legend(title='Model')
# plt.show()

# # %% mlp out
# index = 2
# average_over_prompts = True     
# sum_over_positions = False
# return_variance = True

# # Get both mean and variance
# baseline_l1, baseline_var = get_l1(baseline_results, average_over_prompts=average_over_prompts, 
#                                   sum_over_positions=sum_over_positions, return_index=index, 
#                                   return_variance=return_variance)
# finetuned_l1, finetuned_var = get_l1(finetuned_results, average_over_prompts=average_over_prompts, 
#                                     sum_over_positions=sum_over_positions, return_index=index, 
#                                     return_variance=return_variance)
# nln_l1, nln_var = get_l1(nln_results, average_over_prompts=average_over_prompts, 
#                         sum_over_positions=sum_over_positions, return_index=index, 
#                         return_variance=return_variance)
# noLN_l1, noLN_var = get_l1(noLN_results, average_over_prompts=average_over_prompts, 
#                         sum_over_positions=sum_over_positions, return_index=index, 
#                         return_variance=return_variance)

# # Convert tensors to numpy arrays for plotting
# baseline_mean = baseline_l1.sum(dim=-1).cpu().numpy()
# baseline_std = torch.sqrt(baseline_var.sum(dim=-1)).cpu().numpy()
# baseline_sem = baseline_std / np.sqrt(baseline_results['every_block']['activation'].shape[0])
# finetuned_mean = finetuned_l1.sum(dim=-1).cpu().numpy()
# finetuned_std = torch.sqrt(finetuned_var.sum(dim=-1)).cpu().numpy()
# finetuned_sem = finetuned_std / np.sqrt(finetuned_results['every_block']['activation'].shape[0])
# nln_mean = nln_l1.sum(dim=-1).cpu().numpy()
# nln_std = torch.sqrt(nln_var.sum(dim=-1)).cpu().numpy()
# nln_sem = nln_std / np.sqrt(nln_results['every_block']['activation'].shape[0])
# noLN_mean = noLN_l1.sum(dim=-1).cpu().numpy()
# noLN_std = torch.sqrt(noLN_var.sum(dim=-1)).cpu().numpy()
# noLN_sem = noLN_std / np.sqrt(noLN_results['every_block']['activation'].shape[0])
# # Create x-axis values (layer indices)
# x = np.arange(len(baseline_mean))

# # Create data for each model type with std
# data_std = []
# for i in range(len(x)):
#     # Baseline data
#     data_std.append({'Layer': i, 'L1 Loss': baseline_mean[i], 
#                 'Lower': baseline_mean[i] - baseline_std[i],
#                 'Upper': baseline_mean[i] + baseline_std[i], 
#                 'Model': 'baseline'})
#     # Finetuned data
#     data_std.append({'Layer': i, 'L1 Loss': finetuned_mean[i], 
#                 'Lower': finetuned_mean[i] - finetuned_std[i],
#                 'Upper': finetuned_mean[i] + finetuned_std[i], 
#                 'Model': 'finetuned'})
#     # NLN data
#     data_std.append({'Layer': i, 'L1 Loss': nln_mean[i], 
#                 'Lower': nln_mean[i] - nln_std[i],
#                 'Upper': nln_mean[i] + nln_std[i], 
#                 'Model': 'nln'})
#     # noLN data
#     data_std.append({'Layer': i, 'L1 Loss': noLN_mean[i], 
#                 'Lower': noLN_mean[i] - noLN_std[i],
#                 'Upper': noLN_mean[i] + noLN_std[i], 
#                 'Model': 'noLN'})

# df_std = pd.DataFrame(data_std)

# # Plot using seaborn with area (fill_between) - STD version
# plt.figure(figsize=(10, 6))
# for model, color in zip(['baseline', 'finetuned', 'nln'], ['blue', 'orange', 'green']):
#     model_data = df_std[df_std['Model'] == model]
#     sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model, color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)

# plt.xlabel('Layer')
# plt.ylabel('L1 Loss (summed over positions)')
# plt.title('L1 Loss of Activation vs Attribution Patching (±1 std dev)')
# plt.legend(title='Model')
# plt.show()

# # Create data for each model type with SEM
# data_sem = []
# for i in range(len(x)):
#     # Baseline data
#     data_sem.append({'Layer': i, 'L1 Loss': baseline_mean[i], 
#                 'Lower': baseline_mean[i] - baseline_sem[i],
#                 'Upper': baseline_mean[i] + baseline_sem[i], 
#                 'Model': 'baseline'})
#     # Finetuned data
#     data_sem.append({'Layer': i, 'L1 Loss': finetuned_mean[i], 
#                 'Lower': finetuned_mean[i] - finetuned_sem[i],
#                 'Upper': finetuned_mean[i] + finetuned_sem[i], 
#                 'Model': 'finetuned'})
#     # NLN data
#     data_sem.append({'Layer': i, 'L1 Loss': nln_mean[i], 
#                 'Lower': nln_mean[i] - nln_sem[i],
#                 'Upper': nln_mean[i] + nln_sem[i], 
#                 'Model': 'nln'})
#     # noLN data
#     data_sem.append({'Layer': i, 'L1 Loss': noLN_mean[i], 
#                 'Lower': noLN_mean[i] - noLN_sem[i],
#                 'Upper': noLN_mean[i] + noLN_sem[i], 
#                 'Model': 'noLN'})

# df_sem = pd.DataFrame(data_sem)

# # Plot using seaborn with area (fill_between) - SEM version
# plt.figure(figsize=(10, 6))
# for model, color in zip(['baseline', 'finetuned', 'nln', 'noLN'], ['blue', 'orange', 'green', 'red']):
#     model_data = df_sem[df_sem['Model'] == model]
#     sns.lineplot(data=model_data, x='Layer', y='L1 Loss', label=model, color=color)
#     plt.fill_between(model_data['Layer'], model_data['Lower'], model_data['Upper'], 
#                     alpha=0.3, color=color)

# plt.xlabel('Layer')
# plt.ylabel('Absolute difference (summed over positions)')
# plt.title('Absolute difference between activation and attribution patching (±1 SEM)')
# plt.legend(title='Model')
# plt.show()

# # %%
# index=2
# imshow(baseline_results['every_block']['activation'].mean(dim=0)[index], title='Activation patching mlp out',  xaxis='Position', yaxis='Layer', zmin=-1, zmax=1)
# imshow(baseline_results['every_block']['attribution'].mean(dim=0)[index], title='Attribution patching mlp out', xaxis='Position', yaxis='Layer', zmin=-1, zmax=1)


# # imshow(nln_results['every_block']['activation'].mean(dim=0)[index])
# # imshow(nln_results['every_block']['attribution'].mean(dim=0)[index])

# #%%
# index=0
# context_length = baseline_results['every_block']['activation'].mean(dim=0)[index].shape[1]
# n_layers = baseline_results['every_block']['activation'].mean(dim=0)[index].shape[0]

# def plot_attribution_vs_activation(baseline_results, finetuned_results, nln_results, index, component='every_block'):
#     if component == 'every_block':
#         if index == 0:
#             suptitle = 'Residual stream (pre block)'
#         if index == 1:
#             suptitle = 'Attention out'
#         if index == 2:
#             suptitle = 'MLP out'
#     if component == 'every_attn':
#         if index == 0:
#             suptitle = 'output'
#         if index == 1:
#             suptitle = 'query'
#         if index == 2:
#             suptitle = 'key'
#         if index == 3:
#             suptitle = 'value'
#         if index == 4:
#             suptitle = 'pattern'
#     context_length = baseline_results[component]['activation'].mean(dim=0)[index].shape[1]
#     n_layers = baseline_results[component]['activation'].mean(dim=0)[index].shape[0]
#     fig = plt.figure(figsize=(12, 4.5))
#     gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.1])

#     axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

#     for ax,results, title in zip(axes,[baseline_results, finetuned_results, nln_results], ['baseline', 'finetuned', 'nln']):
#         act_patch = results[component]['activation'].mean(dim=0)[index].cpu().numpy()
#         attr_patch = results[component]['attribution'].mean(dim=0)[index].cpu().numpy()
#         ax.axvline(0, color='grey', linewidth=0.5, linestyle='-')
#         ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
#         ax.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), '-', color='grey', linewidth=0.5)
#         for i in range(context_length):
#             scatter_plot = ax.scatter(act_patch[:, i],
#                         attr_patch[:, i],
#                         c=torch.arange(n_layers), cmap='inferno')
#             ax.set_xlabel('Activation Patch')
#             if ax == axes[0]:  # Only set y-label for the first subplot
#                 ax.set_ylabel('Attribution Patch')
#             ax.set_title(f'{title} model')
#             ax.set_xlim(-1.15, 1.15)
#             ax.set_ylim(-1.15, 1.15)
#             ax.set_xticks([-1, -0.5, 0, 0.5, 1])
#             ax.set_yticks([-1, -0.5, 0, 0.5, 1])
#             ax.set_aspect('equal', 'box')
            
#             # Remove right and top spines
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
            
#             # Offset left and bottom spines (similar to sns.despine offset)
#             ax.spines['left'].set_position(('outward', 5))
#             ax.spines['bottom'].set_position(('outward', 5))
            
#             # Trim spines to be closer to data (similar to sns.despine trim)
#             x_ticks = ax.get_xticks()
#             y_ticks = ax.get_yticks()
#             x_tick_min, x_tick_max = min(x_ticks), max(x_ticks)
#             y_tick_min, y_tick_max = min(y_ticks), max(y_ticks)
#             ax.spines['bottom'].set_bounds(x_tick_min, x_tick_max)
#             ax.spines['left'].set_bounds(y_tick_min, y_tick_max)
#             if ax == axes[1] or ax == axes[2]:
#                 ax.spines['left'].set_visible(False)    
#                 ax.set_yticks([])

#     # Add a colorbar axis, but with less spacing
#     cbar_ax = fig.add_subplot(gs[0, 3])
#     cbar = plt.colorbar(scatter_plot, cax=cbar_ax)
#     cbar.set_label('Layer')
#     cbar.set_ticks(np.arange(0, n_layers, 1))
#     plt.tight_layout()
#     fig.suptitle(f'Attribution vs Activation Patching: {suptitle}', fontsize=16)
#     plt.show()

# #%%
# for index in range(3):
#     plot_attribution_vs_activation(baseline_results, finetuned_results, nln_results, index=index, component='every_block')
# # %% # now lets look at attention heads
# # indeces ["output", "query", "key", "value", "pattern"]
# index = 0
# act = baseline_results['every_attn']['activation'].mean(dim=0)[index]
# attr = baseline_results['every_attn']['attribution'].mean(dim=0)[index]
# imshow(act, title='Attention head activation_patching', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# imshow(attr, title='Attention head attribution_patching', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# imshow(act-attr, title='Attention head activation - attribution', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)

# #%%
# act = finetuned_results['every_attn']['activation'].mean(dim=0)[index]
# attr = finetuned_results['every_attn']['attribution'].mean(dim=0)[index]
# imshow(act, title='Attention head activation_patching', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# imshow(attr, title='Attention head attribution_patching', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# imshow(act-attr, title='Attention head activation - attribution', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)

# #%%
# act = nln_results['every_attn']['activation'].mean(dim=0)[index]
# attr = nln_results['every_attn']['attribution'].mean(dim=0)[index]
# imshow(act, title='Attention head activation_patching', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# imshow(attr, title='Attention head attribution_patching', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# imshow(act-attr, title='Attention head activation - attribution', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)

# #%%
# act = nln_results['every_attn']['activation'].mean(dim=0)[index]
# # %%
# for index in range(5):
#     plot_attribution_vs_activation(baseline_results, finetuned_results, nln_results, index=index, component='every_attn')
# # %%

# imshow(baseline_results['every_attn']['activation'].mean(dim=0)[1], title='Attention head activation_patching', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# imshow(baseline_results['every_attn']['attribution'].mean(dim=0)[1], title='Attention head attribution_patching', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# imshow(baseline_results['every_attn']['activation'].mean(dim=0)[1] - baseline_results['every_attn']['attribution'].mean(dim=0)[1], title='Attention head activation - attribution', xaxis='Head', yaxis='Layer', zmin=-1, zmax=1)
# # %%
# imshow(
#     [baseline_results['every_attn']['activation'].mean(dim=0)[1] - baseline_results['every_attn']['attribution'].mean(dim=0)[1], 
#     finetuned_results['every_attn']['activation'].mean(dim=0)[1] - finetuned_results['every_attn']['attribution'].mean(dim=0)[1], 
#     nln_results['every_attn']['activation'].mean(dim=0)[1] - nln_results['every_attn']['attribution'].mean(dim=0)[1]],
#     facet_col=0,
#     facet_labels=["baseline", "finetuned", "nln"],
#     xaxis='Head',
#     yaxis='Layer',
#     title="activation - attribution",
#     zmin=-1, zmax=1)

# # %%



# %%
