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
    -d DATASET --dataset DATASET    Dataset variant: pile-10k/pile-apollo/pile-uncopyrighted [default: pile-10k]
    -n NUM --num-samples NUM        Number of samples to evaluate [default: 20000]
    -b BATCH_SIZE --batch-size BATCH_SIZE  Batch size for evaluation [default: 8]
    --model-name MODEL_NAME         Base model name [default: gpt2]
    --slay-ln                       Remove LayerNorm from model [default: False]
"""

import os
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, GPT2Config
from transformer_lens import HookedTransformer
from std_dicts import std_dicts
from utils import remove_layernorm


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
    
    # Load model with appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def preprocess_dataset(dataset_name, model_name, num_samples=5000, batch_size=8, cache_dir="processed_datasets"):
    """Preprocess the dataset for evaluation using parallel processing"""
    print(f"Preprocessing {dataset_name} dataset...")
    
    # Check if preprocessed dataset exists
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{model_name}_{num_samples}")
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from {cache_path}")
        processed_dataset = torch.load(cache_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        return processed_dataset, tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if dataset_name == "pile-apollo":
        dataset = load_dataset("apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2", streaming=False, split="train")
        if num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
    elif dataset_name == "pile-uncopyrighted":
        dataset = load_dataset("monology/pile-uncopyrighted", streaming=False, split="train")
        if num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
    elif dataset_name == "pile-10k":
        dataset = load_dataset("NeelNanda/pile-10k", streaming=False, split="train")
        if num_samples < len(dataset):
            # Sample randomly if num_samples is less than dataset size
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    # Process without batching to avoid PyArrow errors
    print("Tokenizing examples...")
    all_tokens = []
    
    # Process dataset without batching to avoid issues with variable-length chunks
    if dataset_name == "pile-apollo":
        # For pre-tokenized datasets
        for example in tqdm(dataset, desc="Processing"):
            all_tokens.extend(example["input_ids"])
    else:
        # For text datasets
        for example in tqdm(dataset, desc="Processing"):
            text = example["text"] + tokenizer.eos_token  # Add EOT token
            tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]
            all_tokens.extend(tokens)
    
    # Chunk into fixed-size blocks
    block_size = 1024
    n_chunks = len(all_tokens) // block_size
    print(f"Creating {n_chunks} chunks from {len(all_tokens)} tokens")
    
    processed_examples = []
    for i in range(n_chunks):
        chunk = all_tokens[i * block_size : (i + 1) * block_size]
        if len(chunk) == block_size:  # Only use complete chunks
            processed_examples.append(torch.tensor(chunk))
    
    print(f"Processed data into {len(processed_examples)} evaluation chunks")
    
    # Cache processed dataset
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(processed_examples, cache_path)
    print(f"Saved processed dataset to {cache_path}")
    
    # Create batches for evaluation - now we're just returning the processed examples
    return processed_examples, tokenizer


def evaluate_on_pile_ce(model, processed_examples, tokenizer, batch_size=8, device=None, pin_memory=True):
    """Evaluate model on preprocessed examples with optimized batch handling"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Preparing evaluation batches (batch_size={batch_size})...")
    
    # Pre-form batches and optionally pin them in memory
    batches = []
    for i in range(0, len(processed_examples), batch_size):
        batch_chunks = processed_examples[i:min(i+batch_size, len(processed_examples))]
        if len(batch_chunks) < batch_size:
            continue
        # Stack chunks into a tensor
        batch_tensor = torch.stack(batch_chunks)
        if pin_memory and device.type == 'cuda':
            batch_tensor = batch_tensor.pin_memory()
        batches.append(batch_tensor)
    
    print(f"Created {len(batches)} evaluation batches")
    
    model.to(device)
    model.eval()

    total_loss = 0
    total_tokens = 0

    # Evaluate with pre-formed batches
    with torch.no_grad():
        for i, batch_input_ids in enumerate(tqdm(batches, desc="Evaluating")):
            # Transfer batch to device
            batch_input_ids = batch_input_ids.to(device, non_blocking=True)
            
            # Compute loss
            if isinstance(model, HookedTransformer):
                logits = model(batch_input_ids)
                loss = model.loss_fn(logits, batch_input_ids, per_token=True)
                
                batch_loss = loss.sum().item()
                batch_tokens = loss.numel()
            else:
                outputs = model(input_ids=batch_input_ids, labels=batch_input_ids)
                
                # For HF models, get more accurate per-token loss
                batch_loss = outputs.loss.item() * batch_input_ids.numel()
                
                # Don't count tokens at EOT positions
                eot_mask = (batch_input_ids == tokenizer.eos_token_id)
                batch_tokens = (~eot_mask).sum().item()
            
            total_loss += batch_loss
            total_tokens += batch_tokens
            
            if i % 10 == 0 or i == 0:
                avg_loss_so_far = total_loss / total_tokens if total_tokens > 0 else float("inf")
                print(f"Batch {i}/{len(batches)}: Current loss = {avg_loss_so_far:.4f}")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    print(f"Evaluation complete. Final loss: {avg_loss:.4f}")
    return avg_loss


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

    if slay_ln:
        model = load_nln_hf_model(model=model, model_name=model_name)

    # Preprocess dataset with caching
    processed_examples, tokenizer = preprocess_dataset(dataset_name, model_name, num_samples, batch_size)
    
    # Evaluate model with pinned memory batches
    ce_loss = evaluate_on_pile_ce(model, processed_examples, tokenizer, batch_size)
    print(f"Final Cross-Entropy Loss on {dataset_name}: {ce_loss:.4f}")

if __name__ == "__main__":
    main()
