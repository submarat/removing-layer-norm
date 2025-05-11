#%%
import sys
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import load_nln_model, load_finetuned_model
import torch
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict
import numpy as np
from transformers import AutoTokenizer
import einops
import matplotlib.pyplot as plt
from neel_plotly import imshow, scatter
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set numpy to use float64
np.set_printoptions(precision=15)
np.random.seed(42)
np.seterr(all='raise')

# Set torch to use float64 as default
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load both models
model_finetuned = load_finetuned_model()
model_nln = load_nln_model()

# Convert model parameters to float64
for model in [model_finetuned, model_nln]:
    for param in model.parameters():
        param.data = param.data.to(torch.float64)
    model.set_use_attn_result(True)
    model.set_use_attn_in(True)
    model.set_use_hook_mlp_in(True)
#%%

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
#%%
# Define a function to get logit diff
def get_logit_diff(logits_and_answer_token_indices):
    logits, answer_token_indices = logits_and_answer_token_indices
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :].to(torch.float64)
    else:
        logits = logits.to(torch.float64)
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

# Function to ablate a specific neuron in the final MLP layer
def ablate_neuron(model, tokens, neuron_idx, position):
    # Create a hook function that zeros out the specified neuron
    def ablation_hook(act, hook):
        # Make a copy and ensure it's the same dtype as the input
        modified_act = act.clone()
        # Zero out 50 neurons starting from neuron_idx across all batch items and positions
        modified_act[:, :, neuron_idx:neuron_idx+50] = 0
        return modified_act
    
    # Run the model with the ablation hook on the final MLP layer
    model.reset_hooks()
    model.add_hook(position, ablation_hook)
    ablated_logits = model(tokens)
    model.reset_hooks()
    return ablated_logits

# Get activations and gradients for the final MLP layer
def get_neuron_attributions(model, tokens, metric_fn, position):
    model.reset_hooks()
    
    # Store activations
    activations = {}
    def forward_hook(act, hook):
        activations[hook.name] = act.detach().to(torch.float64)
    
    # Store gradients
    gradients = {}
    def backward_hook(grad, hook):
        gradients[hook.name] = grad.detach().to(torch.float64)
    
    # Add hooks
    model.add_hook(position, forward_hook, "fwd")
    model.add_hook(position, backward_hook, "bwd")
    
    # Forward and backward pass
    logits = model(tokens)
    logit_diff = metric_fn(logits)
    logit_diff.backward()
    
    # Clean up
    model.reset_hooks()
    
    # Calculate attributions (activation * gradient)
    mlp_activations = activations[position]
    mlp_gradients = gradients[position]
    
    # Sum over batch and position dimensions to get per-neuron attribution
    attributions = (mlp_activations * mlp_gradients).sum(dim=(0, 1))
    
    return attributions

#%%
# Process both models in parallel
results = {}
positions = []
for i in range(12):
    positions.extend([
        f'blocks.{i}.hook_mlp_out',
        f'blocks.{i}.hook_resid_post', 
        f'blocks.{i}.hook_attn_out'
    ])

fig, axes = plt.subplots(12, 3, figsize=(12,48))
for position, ax in zip(positions, axes.flatten()):
    results[position] = {}
    for model_name, model in [("finetuned", model_finetuned), ("nln", model_nln)]:
        print(f"\nProcessing {model_name} model...")
        
        # Tokenize prompts
        clean_tokens = model.to_tokens(prompts)
        answer_token_indices = torch.tensor(
            [
                [model.to_single_token(answers[i][j]) for j in range(2)]
                for i in range(len(answers))
            ],
            device=model.cfg.device,
            dtype=torch.int64  # Keep indices as int64
        )
        
        # Option 2: Convert model parameters to float32 (preferred)
        model = model.to(torch.float32)
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        
        clean_logit_diff = get_logit_diff([clean_logits, answer_token_indices]).item()
        print(f"Clean logit diff: {clean_logit_diff:.15f}")
        
        # Ablation experiment
        neuron_effects = []
        num_neurons_to_test = 768 - 50  
        test_every_n_neurons = 20
        
        print("Running ablation experiment...")
        for neuron_idx in tqdm(range(0, num_neurons_to_test, test_every_n_neurons)):
            ablated_logits = ablate_neuron(model, clean_tokens, neuron_idx, position).to(torch.float64)
            ablated_logit_diff = get_logit_diff([ablated_logits, answer_token_indices]).item()
            effect = clean_logit_diff - ablated_logit_diff
            neuron_effects.append(effect)
        
        # Attribution patching
        print("Running attribution patching...")
        
        # Store activations
        activations = {}
        def forward_hook(act, hook):
            activations[hook.name] = act.detach().to(torch.float64)
        # Store gradients
        gradients = {}
        def backward_hook(grad, hook):
            gradients[hook.name] = grad.detach().to(torch.float64)

        model.reset_hooks()
        # Add hooks
        model.add_hook(position, forward_hook, "fwd")
        model.add_hook(position, backward_hook, "bwd")
        # Forward and backward pass
        logits = model(clean_tokens)
        logit_diff = get_logit_diff([logits, answer_token_indices])
        logit_diff.backward()
        # Clean up
        model.reset_hooks()
        # Calculate attributions (activation * gradient)
        mlp_activations = activations[position]
        mlp_gradients = gradients[position]
        # Sum over batch and position dimensions to get per-neuron attribution
        neuron_attributions = (mlp_activations * mlp_gradients).sum(dim=(0, 1))
        
        # Group attribution scores in chunks of 50
        grouped_attributions = []
        for i in range(0, len(neuron_attributions)-50, test_every_n_neurons):
            group_score = neuron_attributions[i:i+50].sum()
            grouped_attributions.append(group_score)
        
        # Store results for comparison
        results[position][model_name] = {
            "neuron_effects": neuron_effects,
            "neuron_attributions": neuron_attributions,
            "grouped_attributions": grouped_attributions,
            'attribution_ablation_correlation': np.corrcoef(torch.tensor(neuron_effects).cpu().numpy(),torch.tensor(grouped_attributions).cpu().numpy())[0,1]
        }   

        import matplotlib.pyplot as plt
        ax.scatter(torch.tensor(results[position][model_name]['neuron_effects']).cpu().numpy(), torch.tensor(results[position][model_name]['neuron_effects']).cpu().numpy()- torch.tensor(results[position] [model_name]['grouped_attributions']).cpu().numpy(), label=model_name)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("ablation - attribution")
    ax.set_ylabel("attribution")
    ax.set_title(f"{position}")
    ax.legend()
    ax.set_axis_off()


#%%
fig, axes = plt.subplots(12, 3, figsize=(12,48))
for position, ax in zip(positions, axes.flatten()):
    for model_name in ['finetuned', 'nln']:
        ax.scatter(torch.tensor(results[position][model_name]['neuron_effects']).cpu().numpy(), torch.tensor(results[position][model_name]['neuron_effects']).cpu().numpy()- torch.tensor(results[position] [model_name]['grouped_attributions']).cpu().numpy(), label=model_name)
    ax.axhline(0, color='black', linewidth=0.5)
    maximum = np.max([
        np.abs(results[position]['finetuned']['neuron_effects']), 
        np.abs(results[position]['nln']['neuron_effects'])
        ])*1.05
    minimum = - maximum
    ax.set_xlim(minimum, maximum)
    ax.set_xlabel("ablation - attribution")
    ax.set_ylabel("attribution")
    ax.set_title(f"{position}")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
plt.show()
# %%
fig, axes = plt.subplots(12, 3, figsize=(12,48))
for position, ax in zip(positions, axes.flatten()):
    for model_name in ['finetuned', 'nln']:
        ax.scatter(torch.tensor(results[position][model_name]['neuron_effects']).cpu().numpy(),  torch.tensor(results[position] [model_name]['grouped_attributions']).cpu().numpy(), label=model_name)
    ax.axhline(0, color='black', linewidth=0.5)
    maximum = np.max([
        np.abs(results[position]['finetuned']['neuron_effects']), 
        np.abs(results[position]['nln']['neuron_effects'])
        ])*1.05
    minimum = - maximum
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.plot()
    ax.set_xlabel("ablation - attribution")
    ax.set_ylabel("attribution")
    ax.set_title(f"{position}")
    ax.legend()
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.show()
# %%
fig, axes = plt.subplots(1,3, figsize=(12,4))
fig.suptitle("attribution ablation correlation")
for i, ax in enumerate(axes):
    positions_subset = positions[i::3]
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions_subset)))
    for position, color in zip(positions_subset, colors):
        ax.plot([0,1], [results[position]['finetuned']['attribution_ablation_correlation'], 
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
plt.show()

# %%
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
plt.show()
# %%
