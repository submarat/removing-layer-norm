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
split_dataset["train"] = split_dataset["train"].select(range(50000))

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    output = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
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
tokenized = split_dataset.map(tokenize_function, remove_columns=["text"], num_proc=8)

# %%
model = transformers.GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="gpt2_cache")

# %% movwe the model and the data to the device
model.to(device)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# %%
training_args = TrainingArguments(
    output_dir="./results",
    max_steps=10,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    prediction_loss_only=True,
    learning_rate=2e-5,
    report_to="wandb",
    run_name="gpt2-openwebtext-1024"
)

from transformers import TrainerCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# %%
trainer.train()
# %%

training_args = TrainingArguments(
    output_dir="./results",
    max_steps=10,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    prediction_loss_only=True,
    learning_rate=2e-5,
)

from transformers import TrainerCallback

class FakeLayerNorm(torch.nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
    
    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x):
        return x * self.std

fake_lns = {}
def replace_layernorm_with_fake_layernorm(model, std):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            # Split the module name to navigate to the parent module
            components = name.split('.')
            parent = model
            for comp in components[:-1]:
                parent = getattr(parent, comp)
            # Replace the LayerNorm with FakeLayerNorm
            fake_ln = FakeLayerNorm(std)
            setattr(parent, components[-1], FakeLayerNorm(std))

            # Construct the module path - a fully qualified path to the module
            module_path = '.'.join(components)
            # Add to fake_lns dict 
            fake_lns[module_path] = fake_ln
            
# Replace all LayerNorm instances with FakeLayerNorm
replace_layernorm_with_fake_layernorm(model, std=1.0)

# %%
class LNRemover():
    """
    Schedules the "removal" of LayerNorms by replacing the standard deviation with
    a constant value. The constructor takes the std, the desired training step,
    and the a callback that will be called when the training step is reached.
    """
    def __init__(self, std, step, callback):
        self.std = std
        self.step = step
        self.callback = callback

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            self.callback()
        return control


def disable_ln1(model):
    model.ln_f.weight.data = torch.ones_like(model.ln_f.weight.data) * 0.0
    model.ln_f.bias.data = torch.zeros_like(model.ln_f.bias.data)

    
class LNRemoverCallback(TrainerCallback):
    def __init__(self, ln_removers):
        self.ln_removers = ln_removers

    def on_step_begin(self, args, state, control, **kwargs):
        print("Removing LN")
        # 
        return control

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    # callbacks=[BeginStepCallback()]
)

# %%
trainer.train()
# %%
