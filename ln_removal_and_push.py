#!/usr/bin/env python
"""
Load a model checkpoint, remove layer normalization by scaling, and push to HuggingFace Hub.

This script requires the docopt package. Install with:
    pip install docopt

Usage:
    ln_removal_and_push.py --checkpoint=<checkpoint_path> --model-name=<model_name> [--output-name=<output_name>] [--push]
    ln_removal_and_push.py -h | --help

Arguments:
    --checkpoint=<checkpoint_path>   Path to the checkpoint
    --model-name=<model_name>       Base model name (e.g., "gpt2", "gpt2-medium")
    --output-name=<output_name>     Name for the output model on HuggingFace Hub [default: None]
    --push                          Push to HuggingFace Hub [default: False]
    -h --help                       Show this help message and exit
"""

import os
import torch
from docopt import docopt
from transformers import GPT2LMHeadModel
from utils import extract_std_from_checkpoint, remove_layernorm_by_scaling, get_device


def main():
    # Parse command line arguments using docopt
    args = docopt(__doc__)
    
    # Extract arguments
    checkpoint_path = args['--checkpoint']
    model_name = args['--model-name']
    output_name = args['--output-name']
    should_push = args['--push']
    
    # Set default output name if not provided
    if output_name == 'None':
        output_name = f"{model_name}-no-ln"

    # Get the device
    device = get_device()
    print(f"Using device: {device}")

    try:
        # Step 1: Extract std values from the checkpoint
        print(f"Extracting std values from checkpoint: {checkpoint_path}")
        std_dict = extract_std_from_checkpoint(model_name, checkpoint_path)
        print(f"Successfully extracted std values for {len(std_dict)} layers")

        # Step 2: Load the model from the checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        model.to(device)
        print("Model loaded successfully")

        # Step 3: Remove layer normalization by scaling
        print("Removing layer normalization by scaling weights")
        model = remove_layernorm_by_scaling(model, std_dict)
        print("Layer normalization effectively removed")

        # Step 4: Save the model locally
        output_dir = f"{output_name}-local"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving modified model to {output_dir}")
        model.save_pretrained(output_dir)
        print("Model saved successfully")

        # Step 5: Push to HuggingFace Hub if specified
        if should_push:
            print(f"Pushing model to HuggingFace Hub as: {output_name}")
            model.push_to_hub(output_name)
            print(f"Model successfully pushed to: https://huggingface.co/{output_name}")
        else:
            print("Skipping push to HuggingFace Hub (--push not specified)")
            print(f"To push later, run: transformers-cli upload {output_dir}")

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main()) 