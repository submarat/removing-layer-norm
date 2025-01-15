# %%
import torch

# %%
import transformers
import datasets
from transformers import AutoTokenizer
import tqdm
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
split_dataset["train"] = split_dataset["train"].select(range(50000))

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    ids = tokenizer(examples["text"], truncation=True)["input_ids"]
    ids.append(tokenizer.eos_token_id)
    out = {"ids": ids, "len": len(ids)}
    return out

# %%

tokenized = split_dataset.map(tokenize_function, remove_columns=["text"], num_proc=8)

# %%
from transformers import Trainer, TrainingArguments

# %%
model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
# %% save the model to disk 
import os
if not os.path.exists("gpt2"):
    model.save_pretrained("gpt2")
# %%
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=None,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)
# %%