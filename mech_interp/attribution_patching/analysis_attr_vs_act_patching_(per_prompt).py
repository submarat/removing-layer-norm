#%%
import sys
sys.path.append("/workspace/removing-layer-norm/")
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import AutoTokenizer
from mech_interp.load_models import ModelFactory


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
                                        [pickleload(f"/workspace/removing-layer-norm/mech_interp/attribution_patching/results/prompts_{prompts_set}/new_resid_pre_attr_patch_result_{model_name}.pkl") 
                                        for prompts_set in range(0, 30)
                                        ], 
                                        dim=-1),
            'activation' : torch.cat([pickleload(f"/workspace/removing-layer-norm/mech_interp/attribution_patching/results/prompts_{prompts_set}/new_resid_pre_act_patch_result_{model_name}.pkl") 
                                        for prompts_set in range(0, 30)
                                        ], 
                                        dim=-1),
        },
    }   
    return results

def plot_patching_heatmaps_combined(
        norm_patch_baseline,
        norm_patch_vanilla, 
        norm_patch_noLN,
        norm_attr_patch_baseline, 
        norm_attr_patch_vanilla, 
        norm_attr_patch_noLN, 
        sentence_tokens, 
        zmin=-1, zmax=1, 
        save_path=None
        ):
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


baseline_results = get_results_dicts('baseline')
vanilla_results = get_results_dicts('vanilla')
noLN_results = get_results_dicts('noLN')

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

minmax_value = torch.max(torch.abs(torch.stack([
    norm_act_patch_baseline.mean(dim=-1), 
    norm_attr_patch_baseline.mean(dim=-1),
    norm_act_patch_vanilla.mean(dim=-1), 
    norm_attr_patch_vanilla.mean(dim=-1),
    norm_act_patch_noLN.mean(dim=-1), 
    norm_attr_patch_noLN.mean(dim=-1),
])))


prompts = pickleload('/workspace/removing-layer-norm/mech_interp/attribution_patching/results/prompts_1/prompts.pkl')
prompt = prompts['prompts'][0]


factory = ModelFactory( 
    ['baseline'],
    model_dir="models",
    model_size="small"
)
model = factory.models['baseline']
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

# Plot combined activation and attribution patching results
plot_patching_heatmaps_combined(
    norm_act_patch_baseline, 
    norm_act_patch_vanilla, 
    norm_act_patch_noLN, 
    norm_attr_patch_baseline, 
    norm_attr_patch_vanilla, 
    norm_attr_patch_noLN, 
    sentence_tokens, 
    zmin=-minmax_value, 
    zmax=minmax_value,
    save_path='combined_patching_heatmap.png'
)

