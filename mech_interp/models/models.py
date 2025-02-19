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

def load_baseline():
    hooked_model = HookedTransformer.from_pretrained("gpt2-small", fold_ln=True, center_unembed=False)
    return hooked_model

def load_hf_model():
    download_model_if_not_exists(
    repo_id="apollo-research/gpt2_noLN",
    revision="vanilla_1200",
    local_dir="apollo_gpt2_finetuned"
    )
    model = GPT2LMHeadModel.from_pretrained("apollo_gpt2_finetuned")
    hooked_model = HookedTransformer.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=False)
    return hooked_model
 
def load_nln_hf_model():
    download_model_if_not_exists(
    repo_id="apollo-research/gpt2_noLN",
    revision="main",
    local_dir="apollo_gpt2_noLN"
    )
    model = GPT2LMHeadModel.from_pretrained("apollo_gpt2_noLN")
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
    hooked_baseline = load_baseline()
    hooked_finetuned = load_hf_model()
    hooked_noLN = load_nln_hf_model()

    reference_text = "Hello, my name is"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = tokenizer.encode(reference_text, return_tensors="pt")

    with t.no_grad():
        baseline_logit, baseline_token = t.max(hooked_baseline(tokens)[:, -1, :], dim=-1)
        finetuned_logit, finetuned_token = t.max(hooked_finetuned(tokens)[:, -1, :], dim=-1)
        noLN_logit, noLN_token = t.max(hooked_noLN(tokens)[:, -1, :], dim=-1)

    baseline_word = tokenizer.decode([baseline_token.item()])
    finetuned_word = tokenizer.decode([finetuned_token.item()])
    noLN_word = tokenizer.decode([noLN_token.item()])
   
    print(f"Input sequence : '{reference_text}'")
    print(f"Baseline Model: Next token = '{baseline_word}', logit = {baseline_logit.item() :.2f}")
    print(f"Finetuned Model: Next token = '{finetuned_word}', logit = {finetuned_logit.item() :.2f}")
    print(f"NoLN Model: Next token = '{noLN_word}', logit = {noLN_logit.item() :.2f}")
