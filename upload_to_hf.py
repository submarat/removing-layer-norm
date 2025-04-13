"""
Examples:
    python upload_to_hf.py -c results/checkpoint-300 -t submarat/gpt2-medium-without-ln

Usage:
    upload_to_hf.py -c CKPT -t TARGET -m MODEL

Options:
    -h --help                      Show this help message
    -c CKPT --ckpt CKPT            Model checkpoint path [REQUIRED]
    -t TARGET --target TARGET      Model huggingface repo [REQUIRED]
    -m MODEL --model MODEL         Model name [REQUIRED]
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
from huggingface_hub.utils import RepositoryNotFoundError
from std_dicts import std_dicts
import os
import json

def verify_ln_eps(model):
    """Verify that all LayerNorm epsilon values are set correctly."""
    epsilon_value = 1e12
    all_correct = True
    
    for id, block in enumerate(model.transformer.h):
        if abs(block.ln_1.eps - epsilon_value) > 1e-5:
            print(f"Block {id} ln_1 eps incorrect: {block.ln_1.eps}")
            all_correct = False
        if abs(block.ln_2.eps - epsilon_value) > 1e-5:
            print(f"Block {id} ln_2 eps incorrect: {block.ln_2.eps}")
            all_correct = False
    
    if abs(model.transformer.ln_f.eps - epsilon_value) > 1e-5:
        print(f"Final LayerNorm eps incorrect: {model.transformer.ln_f.eps}")
        all_correct = False
    
    return all_correct

def main():
    from docopt import docopt
    args = docopt(__doc__)
    
    # Parse arguments
    ckpt = args['--ckpt']
    model_name = args['--model']
    target = args['--target']
    
    # 1. Login to Hugging Face
    login()

    # 2. Set the model names
    source_model_name = ckpt
    target_model_name = target
    temp_dir = "./model-without-ln"

    # 3. Load and modify the model
    print(f"Loading model from {source_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(source_model_name)
    
    print("Applying remove_ln transformation...")
    modified_model = remove_ln(model, model_name)
    
    # Verify epsilon values before saving
    print("\nVerifying epsilon values before saving...")
    if verify_ln_eps(modified_model):
        print("All epsilon values are correct before saving.")
    else:
        print("Warning: Some epsilon values are incorrect before saving!")

    # 4. Save the model
    print(f"\nSaving modified model to {temp_dir}...")
    modified_model.save_pretrained(temp_dir)
    
    # 5. Load the saved model and verify
    print("\nLoading saved model to verify persistence...")
    loaded_model = AutoModelForCausalLM.from_pretrained(temp_dir)
    
    print("\nVerifying epsilon values after loading...")
    if verify_ln_eps(loaded_model):
        print("All epsilon values are correct after loading.")
    else:
        print("Warning: Some epsilon values were reset after loading!")
        
        # Print config information for debugging
        print("\nConfig information:")
        print(f"Original layer_norm_epsilon: {loaded_model.config.original_layer_norm_epsilon}")
        print(f"Current layer_norm_epsilon: {loaded_model.config.layer_norm_epsilon}")
        if hasattr(loaded_model.config, 'layer_norm_eps_values'):
            print("Stored epsilon values:", json.dumps(loaded_model.config.layer_norm_eps_values, indent=2))

    # 6. Upload if verification passes
    if verify_ln_eps(loaded_model):
        # Handle repository creation/update
        api = HfApi()
        try:
            repo_info = api.repo_info(target_model_name, repo_type="model")
            print(f"\nRepository {target_model_name} exists. Preparing to update...")
        except RepositoryNotFoundError:
            print(f"\nRepository {target_model_name} not found. Creating new repository...")
            api.create_repo(target_model_name, repo_type="model", private=False)

        print("Uploading model to Hugging Face Hub...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=target_model_name,
            repo_type="model"
        )
        print("Model upload completed successfully!")
    else:
        print("\nAborting upload due to incorrect epsilon values!")

    # 7. Cleanup
    print("\nCleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()