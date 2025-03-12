"""
Generate standard deviations used for layernorm removal.

Usage:
    print_std.py [--model-name=<model>] [--output=<file>] [--checkpoint-path=<ckpt_path>]

Options:
    --model-name=<model>              Name of model to analyze [default: gpt2]
    --output=<file>                   Output JSON file defaults to {model-name}_std_dicts.json
    --checkpoint-path=<ckpt_path>     Checkpoint path
"""

import json
import torch
import datasets
import random
from docopt import docopt
from transformer_lens import HookedTransformer
from prepare_dataset import prepare_dataset
from train import load_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def generate_std_values_from_checkpoint(model_name, checkpoint_path, output_file):

    model = load_model(model_name, remove_ln=True, checkpoint_path=checkpoint_path)

    std_dict = {}
    std_bos_dict = {}
    n_layers = len(model.transformer.h)
    for i, block in enumerate(model.transformer.h):
        std_dict[f"blocks.{i}.hook_resid_pre"] = block.ln_1.average_std.mean().item()
        std_dict[f"blocks.{i}.hook_resid_mid"] = block.ln_2.average_std.mean().item()
        std_bos_dict[f"blocks.{i}.hook_resid_pre"] = block.ln_1.bos_std.mean().item()
        std_bos_dict[f"blocks.{i}.hook_resid_mid"] = block.ln_2.bos_std.mean().item()
    std_dict[f"blocks.{n_layers-1}.hook_resid_post"] = model.transformer.ln_f.average_std.mean().item()
    std_bos_dict[f"blocks.{n_layers-1}.hook_resid_post"] = model.transformer.ln_f.bos_std.mean().item()

    combined_dict = {
        "std_dict": std_dict,
        "std_bos_dict": std_bos_dict
    }

    open(output_file, "w").write(json.dumps(combined_dict, indent=4))

def generate_std_values(model_name, output_file):
    # Load the model using HookedTransformer
    model = HookedTransformer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Initialize a dictionary to store the std values
    std_dict = {}
    std_bos_dict = {}

    # Load tokenized dataset and get a random sample
    print("Loading tokenized dataset...")
    tokenized, _ = prepare_dataset()
    train_size = len(tokenized["train"])
    random_idx = random.randint(0, train_size-1)
    sample = tokenized["train"][random_idx]
    input_ids = torch.tensor(sample["input_ids"], device=device).unsqueeze(0) # Add batch dimension
    
    # Run model with hooks to get intermediate activations
    def save_hook(tensor, hook):
        # Calculate std for BOS token (position 0) and rest separately
        bos_std = tensor[:, 0, :].std().item()
        std = tensor[:, 1:, :].std().item()
        
        # Store both values in respective dictionaries
        if hook.name not in std_dict:
            std_dict[hook.name] = std
            std_bos_dict[hook.name] = bos_std
        return tensor

    # Create list of hooks for all residual stream points
    hooks = []
    n_layers = model.cfg.n_layers
    for block_id in range(n_layers):  # GPT-2 has 12 layers
        hooks.extend([
            (f'blocks.{block_id}.hook_resid_pre', save_hook),
            (f'blocks.{block_id}.hook_resid_mid', save_hook),
            (f'blocks.{block_id}.hook_resid_post', save_hook)
        ])

    # Run forward pass with hooks
    _ = model.run_with_hooks(
        input_ids,
        return_type="logits", 
        fwd_hooks=hooks
    )

    combined_dict = {
        "std_dict": std_dict,
        "std_bos_dict": std_bos_dict
    }
    
    from std_dicts import std_dicts
    if model_name in std_dicts:
        # Compare with reference values from std_dicts
        ref_std_dict = std_dicts[model_name]['std_dict']
        ref_std_bos_dict = std_dicts[model_name]['std_bos_dict']

        print("\nComparing with reference values:")
        print("Delta and relative difference between computed and reference std values:")
        for key in std_dict:
            if key in ref_std_dict:
                delta = abs(std_dict[key] - ref_std_dict[key])
                rel_diff = delta / ref_std_dict[key] * 100
                print(f"{key}: abs diff = {delta:.6f}, rel diff = {rel_diff:.2f}%")
            else:
                print(f"{key} not found in reference dictionary")

        print("\nDelta and relative difference between computed and reference BOS std values:")
        for key in std_bos_dict:
            if key in ref_std_bos_dict:
                delta = abs(std_bos_dict[key] - ref_std_bos_dict[key])
                rel_diff = delta / ref_std_bos_dict[key] * 100
                print(f"{key}: abs diff = {delta:.6f}, rel diff = {rel_diff:.2f}%")
            else:
                print(f"{key} not found in reference dictionary")

    open(output_file, "w").write(json.dumps(combined_dict, indent=4))

if __name__ == "__main__":
    args = docopt(__doc__)
    model_name = args["--model-name"]
    output_file = args["--output"]
    checkpoint_path = args["--checkpoint-path"]
    if output_file is None:
        output_file = f"{model_name}_std_dicts.json"

    print(f"Model name: {model_name}")
    print(f"Output file: {output_file}")
    if checkpoint_path:
        generate_std_values_from_checkpoint(model_name, checkpoint_path, output_file)
    else:
        generate_std_values(model_name, output_file)
