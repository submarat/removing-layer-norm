# %%
import torch
import os
from transformers import Trainer, TrainingArguments
import transformers
import datasets
from transformers import AutoTokenizer
import tqdm

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# Check whether the dataset is saved to disk
try: 
    dataset = datasets.load_from_disk("openwebtext")
    print("Dataset loaded from disk")
except FileNotFoundError:
    print("Dataset not found on disk")
    dataset = datasets.load_dataset("openwebtext", num_proc=8)
    dataset.save_to_disk("openwebtext")

# %%
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
# reduce the size of the training set
# split_dataset["train"] = split_dataset["train"].select(range(50000))

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    output = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    # Create labels (same as input_ids for language modeling)
    # input_ids = output["input_ids"]
    # labels = input_ids.copy()
    # # Add EOS token
    # input_ids.append(tokenizer.eos_token_id)
    # labels.append(tokenizer.eos_token_id)
    return {
        "input_ids": output["input_ids"],
        "labels": output["input_ids"],
        "attention_mask": output["attention_mask"],
    }
    # ids = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=418)["input_ids"]
    # ids.append(tokenizer.eos_token_id)
    # out = {"input_ids": ids, "len": len(ids)}
    # return out
try:
    tokenized = datasets.load_from_disk("tokenized_openwebtext")
    print("Tokenized dataset loaded from disk")
except FileNotFoundError:
    print("Tokenized dataset not found on disk")
    tokenized = split_dataset.map(tokenize_function, remove_columns=["text"], num_proc=8)
    tokenized.save_to_disk("tokenized_openwebtext")

# %%
tokenized.save_to_disk("tokenized_openwebtext")

# %%
model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="gpt2_cache")

# %% movwe the model and the data to the device
model.to(device)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# %%
training_args = TrainingArguments(
    output_dir="./results",
    max_steps=1000,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    prediction_loss_only=True,
    learning_rate=2e-5,
    report_to="wandb",
    run_name="gpt2-openwebtext-512",
    logging_steps=1,
    logging_first_step=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# %%
import wandb
wandb.init(project="hf-transformer-trials")
trainer.train()
wandb.finish()
# %%