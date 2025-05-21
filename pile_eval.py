"""
Shared evaluation functions for Pile datasets.
Used by both train.py and eval_pile.py.
"""

import os
import numpy as np
import random
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

def preprocess_pile_dataset(dataset_name, model_name, num_samples=5000, cache_dir="processed_datasets", filter_subsets=True):
    """Preprocess dataset for evaluation using efficient batching"""
    print(f"Preprocessing {dataset_name} dataset...")
    
    # Check if preprocessed dataset exists
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{model_name}_{num_samples}")
    if os.path.exists(cache_path):
        print(f"Loading preprocessed dataset from {cache_path}")
        processed_examples = torch.load(cache_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return processed_examples, tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    if dataset_name == "pile-apollo":
        dataset = load_dataset("apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2", streaming=True, split="train")
        dataset = dataset.take(num_samples)
    elif dataset_name == "pile-apollo-luca":
        dataset = load_dataset("lucabaroni/apollo-pile-filtered-10k", streaming=False, split="train")
        if num_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
    elif dataset_name == "pile-uncopyrighted":
        dataset = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")
        dataset = dataset.take(num_samples)
    elif dataset_name == "pile-10k":
        dataset = load_dataset("NeelNanda/pile-10k", streaming=False, split="train")
        if num_samples < len(dataset):
            # Sample randomly if num_samples is less than dataset size
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
    dataset = dataset.shuffle(seed=42)
    
    # Process without batching to avoid PyArrow errors
    print("Tokenizing examples...")
    all_tokens = []
    
    # Process dataset without batching to avoid issues with variable-length chunks
    if dataset_name == "pile-apollo" or dataset_name == "pile-apollo-luca":
        # For pre-tokenized datasets
        for example in tqdm(dataset, desc="Processing"):
            all_tokens.extend(example["input_ids"])
    else:
        # For text datasets
        for example in tqdm(dataset, desc="Processing"):
            # Filter out OpenWebText2 and books
            add_example = True
            if filter_subsets:
                try:
                    if example.get('meta', {}).get('pile_set_name') in ["OpenWebText2", "Books3", "BookCorpus2"]:
                        add_example = False
                except Exception as e:
                    pass
            if add_example:
                text = example["text"] + tokenizer.eos_token  # Add EOT token
                tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]
                all_tokens.extend(tokens)
    
    # Chunk into fixed-size blocks
    block_size = 1024
    n_chunks = min(num_samples, len(all_tokens) // block_size)

    print(f"Creating {n_chunks} chunks from {len(all_tokens)} tokens")
    
    processed_examples = []
    for i in range(n_chunks):
        chunk = all_tokens[i * block_size : (i + 1) * block_size]
        if len(chunk) == block_size:  # Only use complete chunks
            processed_examples.append(torch.tensor(chunk))
    
    # Shufflink after chunking, important if there are long samples that yield many chunks.
    print("Shufflink after chunking, important if there are long samples that yield many chunks.")
    random.shuffle(processed_examples)
    print(f"Processed data into {len(processed_examples)} evaluation chunks")
    
    # Cache processed dataset
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(processed_examples, cache_path)
    print(f"Saved processed dataset to {cache_path}")
    
    return processed_examples, tokenizer


def evaluate_model_on_pile(model, processed_examples, batch_size=8, device=None, pin_memory=True, tokenizer=None):
    """Evaluate model with efficient batched inference"""
    # Use the model's device if device not explicitly specified
    if device is None:
        device = next(model.parameters()).device
        print(f"Using model's device: {device}")
    else:
        # If device specified, ensure model is on that device
        model = model.to(device)
        print(f"Moving model to specified device: {device}")
    
    print(f"Preparing evaluation batches (batch_size={batch_size})...")
    
    # Pre-form batches and optionally pin them in memory
    batches = []
    for i in range(0, len(processed_examples), batch_size):
        batch_chunks = processed_examples[i:min(i+batch_size, len(processed_examples))]
        if len(batch_chunks) < batch_size:
            continue
        # Stack chunks into a tensor
        batch_tensor = torch.stack(batch_chunks)
        # Only pin memory if transferring to CUDA device
        if pin_memory and str(device).startswith('cuda'):
            batch_tensor = batch_tensor.pin_memory()
        batches.append(batch_tensor)
    
    print(f"Created {len(batches)} evaluation batches")
    
    model.eval()

    total_loss = 0
    total_tokens = 0
    loss_list = []
    # Evaluate with pre-formed batches
    with torch.no_grad():
        for i, batch_input_ids in enumerate(tqdm(batches, desc="Evaluating")):
            # Transfer batch to device, non_blocking only beneficial for CUDA
            non_blocking = str(device).startswith('cuda')
            batch_input_ids = batch_input_ids.to(device, non_blocking=non_blocking)
            
            if isinstance(model, HookedTransformer):
                logits = model(batch_input_ids)
                per_token_loss = model.loss_fn(logits, batch_input_ids, per_token=True)
                loss_list.append(per_token_loss.flatten().cpu().numpy())
                batch_loss = per_token_loss.sum().item()
                batch_tokens = per_token_loss.numel()
            else:
                outputs = model(input_ids=batch_input_ids, labels=batch_input_ids)
                # outputs.loss is the average loss
            
                # Get logits
                logits = outputs.logits
                # Shift logits and labels for causal language modeling
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch_input_ids[:, 1:].contiguous()
                # Calculate per-token loss using CrossEntropyLoss with reduction='none'
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))            
                # Append to your loss list (assuming loss_list is defined elsewhere)
                loss_list.append(per_token_loss.flatten().cpu().numpy())

                
                # For HF models, get more accurate per-token loss
                batch_loss = outputs.loss.item() * batch_input_ids.numel()

                # TODO: Fix this as a feature to mask out EOT tokens. 
                # Don't count tokens at EOT positions, needs fixing, 
                # Commented out code currently only adapts count and not yet loss.
                # eot_mask = (batch_input_ids == tokenizer.eos_token_id)
                # batch_tokens = (~eot_mask).sum().item()
                batch_tokens = batch_input_ids.numel()
                
            total_loss += batch_loss
            total_tokens += batch_tokens

            # If per_token_loss is too high for any token of a sample, save the sample
            for j in range(batch_input_ids.size(0)):
                if per_token_loss[j].max() > 50.0:
                    print(f"Sample {i} has high loss: {per_token_loss[j].max().item()}")
                    # Save the sample for further analysis
                    with open("high_loss_samples.txt", "a") as f:
                        # format the batch input IDs and loss
                        batch_ids = " ".join(str(batch_input_ids[j].cpu().numpy()))
                        f.write(f"Sample {i}: {batch_ids}\n")
                        loss = " ".join(str(per_token_loss[j].cpu().numpy()))
                        f.write(f"Loss: {loss}\n")
                        # Save the sample decoded
                        if tokenizer is not None:
                            # Decode the input IDs to text
                            decoded_sample = tokenizer.decode(batch_input_ids[j].cpu().numpy(), skip_special_tokens=True)
                            f.write(f"Decoded: {decoded_sample}\n")
                        f.write("\n")
                        

            
            if i % 10 == 0 or i == 0:
                avg_loss_so_far = total_loss / total_tokens if total_tokens > 0 else float("inf")
                print(f"Batch {i}/{len(batches)}: Current loss = {avg_loss_so_far:.4f}")

    loss_list = np.array(loss_list).flatten()
    print(f"{loss_list.shape=}")
    print(f"Median: {np.median(loss_list)}")
    print(f"Mean: {np.mean(loss_list)}")
    # 3. IQR and 95% Range
    # IQR (Interquartile Range)
    q75, q25 = np.percentile(loss_list, [75, 25])
    iqr = q75 - q25
    print(f"50% Range: [{q25:.4f}, {q75:.4f}]")
    print(f"\nInterquartile Range (IQR): {iqr:.4f}")
    # 95% Range (using percentiles)
    lower_bound_95 = np.percentile(loss_list, 2.5)
    upper_bound_95 = np.percentile(loss_list, 97.5)
    print(f"95% Range: [{lower_bound_95:.4f}, {upper_bound_95:.4f}]")
    # 99% Range (using percentiles)
    lower_bound_99 = np.percentile(loss_list, 1)
    upper_bound_99 = np.percentile(loss_list, 99)
    print(f"95% Range: [{lower_bound_99:.4f}, {upper_bound_99:.4f}]")
    # 99% Range (using percentiles)
    lower_bound_999 = np.percentile(loss_list, 0.1)
    upper_bound_999 = np.percentile(loss_list, 99.9)
    print(f"95% Range: [{lower_bound_999:.4f}, {upper_bound_999:.4f}]")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    print(f"Evaluation complete. Final loss: {avg_loss:.4f}")

    np.savetxt('losses.out', loss_list, delimiter=',')

    return avg_loss 

def convert_for_trainer(processed_examples, tokenizer, cache_dir="processed_datasets", dataset_name="pile-10k", model_name="gpt2", num_samples=1000):
    """
    Convert processed examples to a format compatible with HuggingFace Trainer
    """
    cache_path = os.path.join(cache_dir, f"{dataset_name}_trainer_{model_name}_{num_samples}")
    if os.path.exists(cache_path):
        print(f"Loading preprocessed Trainer dataset from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    
    print("Converting processed examples to Trainer-compatible dataset...")
    
    # Create a list of input_ids in the format expected by Trainer
    input_ids_list = [example.tolist() for example in processed_examples]
    
    # Create a HuggingFace Dataset with appropriate format
    dataset_dict = {
        "input_ids": input_ids_list,
        "labels": input_ids_list.copy(),
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Cache processed dataset
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(dataset, cache_path)
    
    print(f"Converted to Trainer-compatible dataset with {len(dataset)} examples")
    return dataset 
