import os
import datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

def prepare_dataset(model_name="gpt2", ctx_len=None):
    # Determine block size
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if ctx_len is not None:
        block_size = ctx_len
    else:
        block_size = 1024
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1e9:
            block_size = tokenizer.model_max_length
        else:
            # Try to load config
            from transformers import AutoConfig
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                if hasattr(config, 'n_ctx'):
                    block_size = config.n_ctx
                elif hasattr(config, 'max_position_embeddings'):
                    block_size = config.max_position_embeddings
            except:
                pass
            
    cache_dir = f"tokenized_dataset_{model_name.replace('/', '_')}"
    if ctx_len is not None:
        cache_dir += f"_ctx{ctx_len}"
        
    # First check if the tokenized dataset exists on disk
    if os.path.exists(os.path.join(cache_dir, "train")) and os.path.exists(os.path.join(cache_dir, "test")):
        print(f"Loading tokenized dataset from {cache_dir}...")
        tokenized = datasets.load_from_disk(cache_dir)
    else:
        print("Tokenized dataset not found. Processing from scratch...")

        print("Downloading openwebtext dataset...")
        dataset = datasets.load_dataset("openwebtext", num_proc=8, trust_remote_code=True)

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )

        # split_dataset["train"] = split_dataset["train"].select(range(100))

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):

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

            # Chunk into block_size token chunks, dropping any remainder
            n_chunks = (
                len(concatenated) // block_size
            )  # Integer division to get complete chunks only
            chunks = [concatenated[i * block_size : (i + 1) * block_size] for i in range(n_chunks)]

            return {"input_ids": chunks}

        print("Tokenizing dataset...")
        tokenized = split_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=512,
            num_proc=32,
            remove_columns=split_dataset["train"].column_names,  # Remove original columns
        )
        
        print(f"Saving tokenized dataset to {cache_dir}...")
        tokenized.save_to_disk(cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use DataCollatorForLanguageModeling with mlm=False for causal language modeling (GPT-2)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Set to False for autoregressive/causal language modeling
        return_tensors="pt",  # Explicitly set return tensors to PyTorch
    )

    # Create a training batch using the data collator
    training_batch = data_collator([tokenized["train"][i] for i in range(4)])

    # Print batch information
    print("Training batch shape:", training_batch["input_ids"].shape)
    print("Labels shape:", training_batch["labels"].shape)

    return tokenized, data_collator

if __name__ == "__main__":
    model_name = "gpt2" 
    print("Caching tokenized dataset to disk...")
    tokenized, data_collator = prepare_dataset(model_name)
    print("Done caching tokenized dataset to disk.")
    
    assert tokenized is not None
    assert data_collator is not None
    print("Tokenized dataset and data collator successfully loaded and cached.")
