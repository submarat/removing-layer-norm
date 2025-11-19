import os
import datasets
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, DatasetDict, Features, Sequence, Value

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
            
    env_dataset_path = os.environ.get("TOKENIZED_DATASET_PATH")
    dataset_name = os.environ.get("DATASET_NAME")
    dataset_split = os.environ.get("DATASET_SPLIT", "train")
    dataset_text_field = os.environ.get("DATASET_TEXT_FIELD", "text")
    dataset_sample_limit = int(os.environ.get("DATASET_SAMPLE_LIMIT", "0")) or None
    dataset_max_chunks = int(os.environ.get("DATASET_MAX_CHUNKS", "0")) or None
    dataset_test_fraction = float(os.environ.get("DATASET_TEST_FRACTION", "0.01"))
    use_synthetic_dataset = os.environ.get("USE_SYNTHETIC_DATASET", "0") == "1"
    tokenized = None
        
    if env_dataset_path:
        if not os.path.isdir(env_dataset_path):
            raise FileNotFoundError(f"TOKENIZED_DATASET_PATH '{env_dataset_path}' does not exist or is not a directory")
        print(f"Loading tokenized dataset from override path: {env_dataset_path}")
        tokenized = datasets.load_from_disk(env_dataset_path)
    elif dataset_name:
        cache_dir = os.environ.get("OUTPUT_DATASET_PATH")
        if cache_dir is None:
            safe_dataset = dataset_name.replace("/", "_")
            cache_dir = f"tokenized_dataset_{safe_dataset}"
            if ctx_len is not None:
                cache_dir += f"_ctx{ctx_len}"
        if os.path.exists(os.path.join(cache_dir, "train")) and os.path.exists(os.path.join(cache_dir, "test")):
            print(f"Loading tokenized dataset from {cache_dir}...")
            tokenized = datasets.load_from_disk(cache_dir)
        else:
            print(f"Tokenized dataset for {dataset_name} not found. Processing from scratch...")
            print(f"Loading raw dataset: {dataset_name} (split={dataset_split})")
            raw_dataset = load_dataset(dataset_name, split=dataset_split)
            if dataset_sample_limit:
                limit = min(dataset_sample_limit, len(raw_dataset))
                print(f"Selecting first {limit} samples out of {len(raw_dataset)}")
                raw_dataset = raw_dataset.select(range(limit))

            print("Tokenizing and chunking dataset...")
            features = Features({"input_ids": Sequence(Value("int32"))})

            def chunk_generator():
                buffer = []
                chunk_count = 0
                for idx, record in enumerate(raw_dataset):
                    text = record.get(dataset_text_field)
                    if text is None:
                        continue
                    text = str(text) + tokenizer.eos_token
                    token_ids = tokenizer(
                        text,
                        truncation=False,
                        padding=False,
                        return_tensors=None,
                    )["input_ids"]
                    buffer.extend(token_ids)
                    while len(buffer) >= block_size:
                        chunk = buffer[:block_size]
                        buffer = buffer[block_size:]
                        yield {"input_ids": chunk}
                        chunk_count += 1
                        if dataset_max_chunks and chunk_count >= dataset_max_chunks:
                            print(f"Reached DATASET_MAX_CHUNKS={dataset_max_chunks}, stopping chunk generation.")
                            return
                if buffer:
                    padded = buffer + [tokenizer.pad_token_id] * (block_size - len(buffer))
                    yield {"input_ids": padded[:block_size]}

            chunk_dataset = Dataset.from_generator(chunk_generator, features=features)
            if len(chunk_dataset) == 0:
                raise RuntimeError("No chunks were created from the dataset. Check dataset_text_field or tokenizer settings.")
            test_fraction = min(max(dataset_test_fraction, 0.001), 0.2)
            print(f"Total chunks produced: {len(chunk_dataset)}. Splitting with test fraction {test_fraction:.3f}")
            tokenized = chunk_dataset.train_test_split(test_size=test_fraction, seed=2357)
            print(f"Saving tokenized dataset to {cache_dir}...")
            tokenized.save_to_disk(cache_dir)
    elif use_synthetic_dataset:
        print("Using synthetic dataset (USE_SYNTHETIC_DATASET=1)")
        total_samples = max(2, int(os.environ.get("SYNTHETIC_TOTAL_SAMPLES", "32")))
        test_samples = int(os.environ.get("SYNTHETIC_TEST_SAMPLES", max(1, total_samples // 4)))
        test_samples = min(test_samples, total_samples - 1)
        sequences = torch.randint(
            low=0,
            high=tokenizer.vocab_size - 1,
            size=(total_samples, block_size),
            dtype=torch.long,
        )
        dataset = datasets.Dataset.from_dict({"input_ids": [seq.tolist() for seq in sequences]})
        tokenized = dataset.train_test_split(test_size=test_samples, shuffle=True, seed=2357)
    else:
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
    sample_count = min(4, len(tokenized["train"]))
    training_batch = data_collator([tokenized["train"][i] for i in range(sample_count)])

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
