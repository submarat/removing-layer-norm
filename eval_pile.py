"""
Evaluate language model performance on The Pile dataset variants.

Example:
    python eval_pile.py -m ckpt1200.pt -f nanogpt -d pile-10k -n 20000 -b 16
    python eval_pile.py -m gpt2 -f transformers -d pile-apollo -b 8
    python eval_pile.py -m results/checkpoint-1200 -f transformers --slay-ln -b 4

Usage:
    eval_pile.py -m MODEL [-f FORMAT] [-d DATASET] [-n NUM_SAMPLES] [-b BATCH_SIZE] [--slay-ln] [--model-name MODEL_NAME]
    eval_pile.py -h | --help

Options:
    -h --help                       Show this help message
    -m MODEL --model MODEL          Model checkpoint path or model id [REQUIRED]
    -f FORMAT --format FORMAT       Model format: nanogpt/transformers [default: transformers]
    -d DATASET --dataset DATASET    Dataset variant: pile-10k/pile-apollo/pile-uncopyrighted [default: pile-apollo]
    -n NUM --num-samples NUM        Number of samples to evaluate [default: 20000]
    -b BATCH_SIZE --batch-size BATCH_SIZE  Batch size for evaluation [default: 8]
    --model-name MODEL_NAME         Base model name [default: gpt2]
    --slay-ln                       Remove LayerNorm from model [default: False]
"""

import os
import torch
from transformer_lens import HookedTransformer
from std_dicts import std_dicts
from utils import remove_layernorm
from pile_eval import preprocess_pile_dataset, evaluate_model_on_pile
from transformers import GPT2LMHeadModel, AutoModelForCausalLM

# Load model with appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_saved_model(model_name: str, model_path=None):
    if model_path is not None: 
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        # Load OpenAI's GPT model - use specific model name like "gpt2-large" or "gpt2-xl"
        model = AutoModelForCausalLM.from_pretrained(model_name)  # or other OpenAI model version
    return model


def load_hf_model(model_id_or_ckpt_path, model_name, slay_ln=False):
    """ Loads huggingface transformers model and removes layernorm """
    model_hf = GPT2LMHeadModel.from_pretrained(model_id_or_ckpt_path)

    if slay_ln:
        remove_layernorm(model_name, model_hf)

    return model_hf


def load_pt_file(filepath, model_name, slay_ln=False):
    """ Loads nanoGPT checkpoint and removes layernorm """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    sd = checkpoint["model"]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    # Load the HF GPT2 model
    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained(model_name)
    sd_hf = model_hf.state_dict()
    # Now, use the state dict from the checkpoint to overwrite the weights in the model
    sd_keys_hf = list(sd_hf.keys())
    sd_keys = list(sd.keys())
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd_hf[k].copy_(sd[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd_hf[k].copy_(sd[k])

    model_hf.load_state_dict(sd_hf)

    # Now kill the layer norm by setting layer_norm_epsilon to 1e12, and multiplied the ln scaling parameters by 1e6
    if slay_ln:
        remove_layernorm(model_hf)
    return model_hf


def load_nln_hf_model(model_name, name=None, model=None):
    if model is None and name is None or model is not None and name is not None:
        raise ValueError("Either name or model must be provided, but not both")
    if model is not None:
        model = model.to("cpu")
    else:
        model = GPT2LMHeadModel.from_pretrained(name).to("cpu")
    
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
                self.blocks[i].ln1 = torch.nn.Identity()
                self.blocks[i].ln2 = torch.nn.Identity()
            self.ln_final = torch.nn.Identity()
    
    hooked_model = HookedTransformerNoLN.from_pretrained(model_name, hf_model=model, fold_ln=True, center_unembed=False).to("cpu")

    hooked_model.removeLN()
    hooked_model.cfg.normalization_type = None
    print(f"loaded {name} and removed LN hack, returning transformerlens hooked model")
    return hooked_model


def custom_tokenizer(examples, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Add EOS token to the end of each example
    examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]

    # Tokenize the examples
    tokenized_examples = tokenizer(
        examples["text"], truncation=False, padding=False, return_tensors=None
    )

    # Concatenate the tokenized examples
    concatenated = []
    for seq in tokenized_examples["input_ids"]:
        concatenated.extend(seq)

    # Chunk into 1024 token chunks, dropping any remainder
    n_chunks = (
        len(concatenated) // 1024
    )  # Integer division to get complete chunks only
    chunks = [concatenated[i * 1024 : (i + 1) * 1024] for i in range(n_chunks)]

    return {"input_ids": chunks}


def main():
    from docopt import docopt
    args = docopt(__doc__)
    
    # Parse arguments
    model_path = args['--model']
    format_type = args['--format']
    dataset_name = args['--dataset']
    num_samples = int(args['--num-samples'])
    batch_size = int(args['--batch-size'])
    model_name = args['--model-name'] or "gpt2"
    slay_ln = args['--slay-ln']
    
    # Create cache directory if it doesn't exist
    os.makedirs("processed_datasets", exist_ok=True)
    
    # Load model based on format
    if format_type == 'nanogpt':
        model = load_pt_file(model_path, model_name, slay_ln=slay_ln)
    elif format_type == 'transformers':
        model = load_hf_model(model_path, model_name, slay_ln=slay_ln)
    else:
        raise ValueError(f"Unknown format type: {format_type}")

    model = model.to(device)
    
    #if slay_ln:
    #    model = load_nln_hf_model(model=model, model_name=model_name)

    # Using shared preprocessing function
    processed_examples, tokenizer = preprocess_pile_dataset(dataset_name, model_name, num_samples)
    
    # Using shared evaluation function
    ce_loss = evaluate_model_on_pile(model, processed_examples, tokenizer, batch_size)
    print(f"Final Cross-Entropy Loss on {dataset_name}: {ce_loss:.4f}")

if __name__ == "__main__":
    main()
