"""
Evaluate language model performance on The Pile dataset variants.

Example:
    python eval_pile.py -m ckpt1200.pt -f nanogpt -d pile-10k -n 20000
    python eval_pile.py -m gpt2 -f transformers -d pile-apollo
    python eval_pile.py -m results/checkpoint-1200 -f transformers --slay-ln

Usage:
    eval_pile.py -m MODEL [-f FORMAT] [-d DATASET] [-n NUM_SAMPLES] [--slay-ln]
    eval_pile.py -h | --help

Options:
    -h --help                       Show this help message
    -m MODEL --model MODEL          Model checkpoint path or model id [REQUIRED]
    -f FORMAT --format FORMAT       Model format: nanogpt/transformers [default: transformers]
    -d DATASET --dataset DATASET    Dataset variant: pile-10k/pile-apollo/pile-uncopyrighted [default: pile-10k]
    -n NUM --num-samples NUM        Number of samples to evaluate [default: 20000]
    --slay-ln                       Remove LayerNorm from model [default: False]
"""

import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, GPT2Config
from transformer_lens import HookedTransformer
from std_dicts import std_dict

def load_saved_model(model_path=None):
    if model_path is not None: 
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        # Load OpenAI's GPT model - use specific model name like "gpt2-large" or "gpt2-xl"
        model = AutoModelForCausalLM.from_pretrained("gpt2")  # or other OpenAI model version    
    return model


def remove_layernorm(model_hf):
    # Now kill the layer norm by setting layer_norm_epsilon to 1e12, and multiplied the ln scaling parameters by 1e6
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

def load_hf_model(model_id_or_ckpt_path, slay_ln=False):
    """ Loads huggingface transformers model and removes layernorm """
    model_hf = GPT2LMHeadModel.from_pretrained(model_id_or_ckpt_path)

    if slay_ln:
        remove_layernorm(model_hf)

    return model_hf


def load_pt_file(filepath, slay_ln=False):
    """ Loads nanoGPT checkpoint and removes layernorm """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    
    # Load model with appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    sd = checkpoint["model"]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    # Load the HF GPT2 model
    # init a huggingface/transformers model
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
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

def load_nln_hf_model(name=None, model=None):
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
    
    hooked_model = HookedTransformerNoLN.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=False).to("cpu")

    hooked_model.removeLN()
    hooked_model.cfg.normalization_type = None
    print(f"loaded {name} and removed LN hack, returning transformerlens hooked model")
    return hooked_model

def custom_tokenizer(examples):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
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


def evaluate_on_pile_ce(model, dataset_name, num_samples=5000, device=None):
    print(f"Evaluating on {dataset_name}, using {num_samples} samles")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_name == "pile-apollo":
        dataset = load_dataset("apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2", streaming=True, split="train")
        dataset = dataset.shuffle(seed=42)
        dataset = list(dataset.take(num_samples))
    elif dataset_name == "pile-uncopyrighted":
        dataset = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")
        dataset = dataset.shuffle(seed=42)
        dataset = list(dataset.take(num_samples))
    elif dataset_name == "pile-10k":
        dataset = load_dataset("NeelNanda/pile-10k", streaming=False, split="train")
    
    if dataset_name == "pile-apollo":
        processed_examples = []
        for i, example in enumerate(tqdm(dataset)):
            processed_examples.append(torch.tensor(example["input_ids"]))
    else:
        # Process dataset in fixed-size chunks
        # processed_examples = []
        # for i in range(0, len(dataset), 1024):
        #     batch = dataset[i:i + 1024]
        #     processed = custom_tokenizer({"text": [ex["text"] for ex in batch]})
        #     processed_examples.extend(processed["input_ids"])
        # Process in chunks without limiting total size
        batch_size = 1024
        processed_examples = []
        
        # Use iterator to process dataset in chunks
        dataset_iterator = iter(dataset)
        while True:
            try:
                batch = []
                # for _ in range(batch_size):
                #     batch.append(next(dataset_iterator)["text"])
                while len(batch) < batch_size:
                    example = next(dataset_iterator)
                    # Filter out OpenWebText2
                    if 1: # example.get('meta', {}).get('pile_set_name') != "OpenWebText2":
                        batch.append(example["text"])
                processed = custom_tokenizer({"text": batch})
                print(len(processed["input_ids"][0]))
                processed_examples.extend(processed["input_ids"])
            except StopIteration:
                break
    
    print(f"Processed data shape: {(len(processed_examples), len(processed_examples[0]))}")
    
    # For testing purposes, print the first 100 examples
    # for i, example in enumerate(dataset):
    #     if i >= 100:
    #         break
    #     print(tokenizer.decode(example["input_ids"]))
    
    model.to(device)
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, chunk in enumerate(tqdm(processed_examples, desc="Evaluating")):
            input_ids = torch.tensor(chunk).unsqueeze(0).to(device)
            
            if isinstance(model, HookedTransformer):
                logits = model(input_ids)
                loss = model.loss_fn(logits, input_ids, per_token=True)
                
                total_loss += loss.sum().item()
                total_tokens += input_ids.numel()
            else:
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item() * len(chunk)
                total_tokens += len(chunk)
            if i % 100 == 0:
                print(
                    f"Processed {i} examples. Current ce loss: {total_loss/total_tokens:.2f}"
                )
            
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def main():
    from docopt import docopt
    args = docopt(__doc__)
    
    # Parse arguments
    model_path = args['--model']
    format_type = args['--format']
    dataset_name = args['--dataset']
    num_samples = int(args['--num-samples'])
    slay_ln = args['--slay-ln']
    
    # Load model based on format
    if format_type == 'nanogpt':
        model = load_pt_file(model_path, slay_ln=slay_ln)
    elif format_type == 'transformers':
        model = load_hf_model(model_path, slay_ln=slay_ln)
    else:
        raise ValueError(f"Unknown format type: {format_type}")

    model = load_nln_hf_model(model=hf_model)

    # Evaluate model
    ce_loss = evaluate_on_pile_ce(model, dataset_name, num_samples)
    print(f"Final Cross-Entropy Loss on {dataset_name}: {ce_loss:.4f}")

if __name__ == "__main__":
    main()