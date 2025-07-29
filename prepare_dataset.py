import os
import datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

def prepare_dataset(model_name="gpt2"):
    dataset_name = "openwebtext" if model_name == "gpt2" else "NeelNanda/pile-10k"
    
    # Set chunk size based on model type
    # GPT-2: 1024 tokens per chunk
    # Pythia: 2049 tokens per chunk (2048 sequence + 1 for shifted labels)
    if "pythia" in model_name.lower():
        chunk_size = 2049
    else:
        chunk_size = 1024
    
    # First check if the tokenized dataset exists on disk
    if os.path.exists("tokenized_dataset/train") and os.path.exists("tokenized_dataset/test"):
        print("Loading tokenized dataset from disk...")
        tokenized = datasets.load_from_disk("tokenized_dataset")
    else:
        print("Tokenized dataset not found. Processing from scratch...")

        print("Downloading openwebtext dataset...")
        dataset = datasets.load_dataset(dataset_name, num_proc=8, trust_remote_code=True)

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        
        # Shuffle the training split again before processing (Pythia-style)
        split_dataset["train"] = split_dataset["train"].shuffle(seed=2357)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        breakpoint()

        def tokenize_function(examples):

            examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]

            # Tokenize the examples
            tokenized_examples = tokenizer(
                examples["text"], truncation=False, padding=False, return_tensors=None
            )

            # Concatenate the tokenized examples
            concatenated = []
            for seq in tokenized_examples["input_ids"]:
                concatenated.extend(seq)

            # Chunk into model-appropriate sizes, dropping any remainder
            # Following Pythia approach: "sample boundaries do not respect document boundaries"  
            # GPT-2: 1024 tokens, Pythia: 2049 tokens (2048 sequence + 1 for shifted labels)
            n_chunks = (
                len(concatenated) // chunk_size
            )  # Integer division to get complete chunks only
            chunks = [concatenated[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]

            return {"input_ids": chunks}

        print("Tokenizing dataset...")
        tokenized = split_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=512,
            num_proc=32,
            remove_columns=split_dataset["train"].column_names,  # Remove original columns
        )
        
        print("Saving tokenized dataset to disk...")
        tokenized.save_to_disk("tokenized_dataset")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
