import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import HfApi, login
from huggingface_hub.utils import RepositoryNotFoundError
from std_dicts import std_dict
import os

def remove_ln(model_hf):
    """Remove layer normalization by scaling weights and setting high epsilon values."""
    for id, block in enumerate(model_hf.transformer.h):
        with torch.no_grad():
            # Get the standard deviations from the std_dict
            ln1_std = std_dict[f'blocks.{id}.hook_resid_pre']
            ln2_std = std_dict[f'blocks.{id}.hook_resid_mid']
            block.ln_1.weight.data *= 1e6 / ln1_std
            block.ln_2.weight.data *= 1e6 / ln2_std
            block.ln_1.eps = 1e12
            block.ln_2.eps = 1e12
    
    with torch.no_grad():
        lnf_std = std_dict[f'blocks.11.hook_resid_post']
        model_hf.transformer.ln_f.weight.data *= 1e6 / lnf_std
        model_hf.transformer.ln_f.eps = 1e12
    return model_hf

def main():
    # 1. Login to Hugging Face
    login()  # You'll need your API token from huggingface.co/settings/tokens

    # 2. Set the model names
    source_model_name = "gpt2"  # or your source model name
    target_model_name = "submarat/model-without-ln"

    # 3. Load the model
    print(f"Loading model from {source_model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(source_model_name)
    
    # 4. Apply the remove_ln transformation
    print("Applying remove_ln transformation...")
    modified_model = remove_ln(model)

    # 5. Create temporary directory and save model
    temp_dir = "./model-without-ln"
    print(f"Saving modified model to {temp_dir}...")
    modified_model.save_pretrained(temp_dir)

    # 6. Handle repository creation/update
    api = HfApi()
    try:
        repo_info = api.repo_info(target_model_name, repo_type="model")
        print(f"Repository {target_model_name} exists. Preparing to update...")
    except RepositoryNotFoundError:
        print(f"Repository {target_model_name} not found. Creating new repository...")
        api.create_repo(target_model_name, repo_type="model", private=False)

    # 7. Upload the modified model
    print("Uploading model to Hugging Face Hub...")
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=target_model_name,
        repo_type="model"
    )

    # 8. Cleanup
    print("Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir)

    print("Model upload completed successfully!")

if __name__ == "__main__":
    main()