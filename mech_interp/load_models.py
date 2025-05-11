import os
import torch as t
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens import HookedTransformer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from huggingface_hub import snapshot_download

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

def download_model_if_not_exists(repo_id, revision, local_dir):
    if not os.path.exists(local_dir):
        print(f"Model not found locally. Downloading {repo_id} ({revision}) to {local_dir}...")
        model_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir
        )
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Model already exists at {local_dir}. Skipping download.")

# baseline models
def load_baseline_small():
    hooked_model = HookedTransformer.from_pretrained("gpt2-small", fold_ln=True, center_unembed=False)
    return hooked_model

def load_baseline_medium():
    hooked_model = HookedTransformer.from_pretrained("gpt2-medium", fold_ln=True, center_unembed=False)
    return hooked_model

#%% vanilla models
def load_vanilla_small():
    model_path = "/workspace/removing-layer-norm/mech_interp/models/vanilla_gpt2_small"
    download_model_if_not_exists(
    repo_id="schaeff/gpt2-small_vanilla300",
    revision="main",
    local_dir=model_path
    )
    model = GPT2LMHeadModel.from_pretrained(model_path)
    hooked_model = HookedTransformer.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=False)
    return hooked_model

def load_vanilla_medium():
    model_path = "/workspace/removing-layer-norm/mech_interp/models/vanilla_gpt2_medium"
    download_model_if_not_exists(
    repo_id="schaeff/gpt-2medium_vanilla500",
    revision="main",
    local_dir=model_path
    )
    model = GPT2LMHeadModel.from_pretrained(model_path)
    hooked_model = HookedTransformer.from_pretrained("gpt2-medium", hf_model=model, fold_ln=True, center_unembed=False)
    return hooked_model

#%%
def load_noLN_small():
    model_path = "/workspace/removing-layer-norm/mech_interp/models/marat_gpt2_noLN_small" 
    download_model_if_not_exists(
    repo_id="submarat/gpt2-noln-ma-aux",
    revision="main",
    local_dir=model_path
    )

    model = GPT2LMHeadModel.from_pretrained(model_path)
    for block in model.transformer.h:
        block.ln_1.weight.data = block.ln_1.weight.data / 1e6
        block.ln_1.eps = 1e-5
        block.ln_2.weight.data = block.ln_2.weight.data / 1e6
        block.ln_2.eps = 1e-5
    model.transformer.ln_f.weight.data = model.transformer.ln_f.weight.data / 1e6
    model.transformer.ln_f.eps = 1e-5
    
    # Properly replace LayerNorms by Identities
    class HookedTransformerNoLN(HookedTransformer):
        def removeLN(self):
            for i in range(len(self.blocks)):
                self.blocks[i].ln1 = t.nn.Identity()
                self.blocks[i].ln2 = t.nn.Identity()
            self.ln_final = t.nn.Identity()
    
    hooked_model = HookedTransformerNoLN.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=False)

    hooked_model.removeLN()
    hooked_model.cfg.normalization_type = None
    return hooked_model


def load_noLN_medium():
    model_path = "/workspace/removing-layer-norm/mech_interp/models/gpt2_noLN_medium" 
    download_model_if_not_exists(
    repo_id="submarat/gpt2-medium-noln-aux",
    revision="main",
    local_dir=model_path
    )

    model = GPT2LMHeadModel.from_pretrained(model_path)
    for block in model.transformer.h:
        block.ln_1.weight.data = block.ln_1.weight.data / 1e6
        block.ln_1.eps = 1e-5
        block.ln_2.weight.data = block.ln_2.weight.data / 1e6
        block.ln_2.eps = 1e-5
    model.transformer.ln_f.weight.data = model.transformer.ln_f.weight.data / 1e6
    model.transformer.ln_f.eps = 1e-5
    
    # Properly replace LayerNorms by Identities
    class HookedTransformerNoLN(HookedTransformer):
        def removeLN(self):
            for i in range(len(self.blocks)):
                self.blocks[i].ln1 = t.nn.Identity()
                self.blocks[i].ln2 = t.nn.Identity()
            self.ln_final = t.nn.Identity()
    
    hooked_model = HookedTransformerNoLN.from_pretrained("gpt2-medium", hf_model=model, fold_ln=True, center_unembed=False)

    hooked_model.removeLN()
    hooked_model.cfg.normalization_type = None
    return hooked_model


#%% older 

def load_finetuned_model():
    model_path = "/workspace/removing-layer-norm/mech_interp/models/apollo_gpt2_finetuned"
    download_model_if_not_exists(
    repo_id="apollo-research/gpt2_noLN",
    revision="main",
    local_dir=model_path
    )
    model = GPT2LMHeadModel.from_pretrained(model_path)
    hooked_model = HookedTransformer.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=False)
    return hooked_model


def load_nln_model():
    model_path = "/workspace/removing-layer-norm/mech_interp/models/apollo_gpt2_noLN" 
    download_model_if_not_exists(
    repo_id="apollo-research/gpt2_noLN",
    revision="main",
    local_dir=model_path
    )

    model = GPT2LMHeadModel.from_pretrained(model_path)
    for block in model.transformer.h:
        block.ln_1.weight.data = block.ln_1.weight.data / 1e6
        block.ln_1.eps = 1e-5
        block.ln_2.weight.data = block.ln_2.weight.data / 1e6
        block.ln_2.eps = 1e-5
    model.transformer.ln_f.weight.data = model.transformer.ln_f.weight.data / 1e6
    model.transformer.ln_f.eps = 1e-5
    
    # Properly replace LayerNorms by Identities
    class HookedTransformerNoLN(HookedTransformer):
        def removeLN(self):
            for i in range(len(self.blocks)):
                self.blocks[i].ln1 = t.nn.Identity()
                self.blocks[i].ln2 = t.nn.Identity()
            self.ln_final = t.nn.Identity()
    
    hooked_model = HookedTransformerNoLN.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=False)

    hooked_model.removeLN()
    hooked_model.cfg.normalization_type = None
    return hooked_model


if __name__ == '__main__':
    # Load correct backend
    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

    # Load hooked transformer for each model
    hooked_baseline_small = load_baseline_small()
    # hooked_baseline_medium = load_baseline_medium()
    hooked_vanilla_small = load_vanilla_small()
    # hooked_vanilla_medium = load_vanilla_medium()
    hooked_noLN_small = load_noLN_small()
    # hooked_noLN_medium = load_noLN_medium()

    reference_text = "Hello, my name is"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = tokenizer.encode(reference_text, return_tensors="pt")

    with t.no_grad():
        baseline_small_logit, baseline_small_token = t.max(hooked_baseline_small(tokens)[:, -1, :], dim=-1)
        # baseline_medium_logit, baseline_medium_token = t.max(hooked_baseline_medium(tokens)[:, -1, :], dim=-1)
        vanilla_small_logit, vanilla_small_token = t.max(hooked_vanilla_small(tokens)[:, -1, :], dim=-1)
        # vanilla_medium_logit, vanilla_medium_token = t.max(hooked_vanilla_medium(tokens)[:, -1, :], dim=-1)
        noLN_small_logit, noLN_small_token = t.max(hooked_noLN_small(tokens)[:, -1, :], dim=-1)
        # noLN_medium_logit, noLN_medium_token = t.max(hooked_noLN_medium(tokens)[:, -1, :], dim=-1)

    baseline_small_word = tokenizer.decode([baseline_small_token.item()])
    # baseline_medium_word = tokenizer.decode([baseline_medium_token.item()])
    vanilla_small_word = tokenizer.decode([vanilla_small_token.item()])
    # vanilla_medium_word = tokenizer.decode([vanilla_medium_token.item()])
    noLN_small_word = tokenizer.decode([noLN_small_token.item()])
    # noLN_medium_word = tokenizer.decode([noLN_medium_token.item()])
   
    print(f"Input sequence : '{reference_text}'")
    print(f"Baseline Model: Next token = '{baseline_small_word}', logit = {baseline_small_logit.item() :.2f}")
    print(f"Finetuned Model: Next token = '{vanilla_small_word}', logit = {vanilla_small_logit.item() :.2f}")
    print(f"NoLN Model: Next token = '{noLN_small_word}', logit = {noLN_small_logit.item() :.2f}")
    # print(f"Baseline Model: Next token = '{baseline_medium_word}', logit = {baseline_medium_logit.item() :.2f}")
    # print(f"Finetuned Model: Next token = '{vanilla_medium_word}', logit = {vanilla_medium_logit.item() :.2f}")
    # print(f"NoLN Model: Next token = '{noLN_medium_word}', logit = {noLN_medium_logit.item() :.2f}")
