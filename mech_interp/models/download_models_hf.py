from huggingface_hub import snapshot_download

# Download first model variant with custom name
model_1_path = snapshot_download(
    repo_id="apollo-research/gpt2_noLN",
    revision="main",
    local_dir="apollo_gpt2_noLN"
)

# Download second variant
model_2_path = snapshot_download(
    repo_id="apollo-research/gpt2_noLN",
    revision="vanilla_1200", 
    local_dir="apollo_gpt2_finetuned"
)
