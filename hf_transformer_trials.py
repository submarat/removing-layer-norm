import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback

import wandb
import transformers
import datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import tqdm
from std_dicts import std_dict, std_bos_dict

import numpy as np

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset():
    # Check whether the dataset is saved to disk
    if not os.path.exists("tokenized_openwebtext"):
        try:
            dataset = datasets.load_from_disk("openwebtext")
            print("Dataset loaded from disk")
        except FileNotFoundError:
            print("Dataset not found on disk")
            dataset = datasets.load_dataset("openwebtext", num_proc=8)
            dataset.save_to_disk("openwebtext")

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
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

        # Chunk into 1024 token chunks, dropping any remainder
        n_chunks = (
            len(concatenated) // 1024
        )  # Integer division to get complete chunks only
        chunks = [concatenated[i * 1024 : (i + 1) * 1024] for i in range(n_chunks)]

        return {"input_ids": chunks}

    # Group the tokenized dataset into chunks
    tokenized = split_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=512,
        num_proc=32,
        remove_columns=split_dataset["train"].column_names,  # Remove original columns
    )

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
    print(
        "\nSample input_ids (first sequence):\n", training_batch["input_ids"][0][:-50]
    )
    print("\nSample labels (first sequence):\n", training_batch["labels"][0][:-50])


def load_model():
    return transformers.GPT2LMHeadModel.from_pretrained(
        "gpt2",
        cache_dir="gpt2_cache",
        config=transformers.GPT2Config.from_pretrained(
            "gpt2", dropout=0.0, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0
        ),
    )


def finetune_with_ln(model, tokenized):
    training_args = TrainingArguments(
        output_dir="./results",
        max_steps=1200,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        prediction_loss_only=True,
        learning_rate=6e-4,
        lr_scheduler_type="cosine",
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
        data_collator=data_collator,
    )

    wandb.init(project="hf-remove-ln")
    trainer.train()
    wandb.finish()


def finetune_without_ln(model, tokenized):

    training_args = TrainingArguments(
        output_dir="./results",
        save_safetensors=False,
        max_steps=1200,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        prediction_loss_only=True,
        learning_rate=6e-4,
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name="gpt2-openwebtext-512",
        logging_steps=1,
        logging_first_step=True,
    )

    class FakeLayerNorm(nn.Module):
        """LayerNorm using a fixed std instead of the actual standard deviation."""

        def __init__(self, ndim, layer, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
            # Flag whether the LayerNorm is enabled ("real") or disabled ("fake")
            self.mode = "real"
            self.attn_v_mode = "real"
            self.average_std = torch.ones(ndim, device="cuda") * std_dict[layer]
            self.average_std[0] = std_bos_dict[layer]
            self.bos_std = torch.ones(ndim, device="cuda") * std_bos_dict[layer]

        def forward(self, input, std_type="avg", attn_v=False):
            # We want all the enable / disable information to be in this class, but the class is re-used
            # for both the QK and V paths. Thus we add the attn_v flag to the call that is True only for
            # the V path. Thus we get to have flags `mode` and `attn_v_mode` to enable / disable the
            # LN for the QK and V paths separately.
            mode = self.attn_v_mode if attn_v else self.mode
            if mode == "fake":
                # Which std values to use: We use (1) average std (which is actually a vector of length
                # n_ctx for most of the time*) [a, b, b, ...] where a is the average std for position 1,
                # and b is the average std for all other positions. We also have the option to use (2)
                # the bos std [a, a, a, ...] for all positions, which we do if the input token is EOT.
                # Note that we could differentiate between EOT and BOS, but I didn't need it here.
                # *at the end (with disable_eot_std) we make the latter be like the former, and with
                # disable_bos_std we make both vectors to be [b, b, b, ...], equivalent to scalars.
                assert std_type in ["avg", "bos"]
                std = self.average_std if std_type == "avg" else self.bos_std
                return (
                    (input - input.mean(-1, keepdim=True)) / std * self.weight
                    + self.bias
                    if self.bias is not None
                    else input * self.weight
                )
            elif mode == "real":
                return F.layer_norm(
                    input, self.weight.shape, self.weight, self.bias, 1e-5
                )
            else:
                raise ValueError(f"Unknown mode {mode}")

    def replace_layernorm_with_fake_layernorm(model, std):
        n_layers = 12
        n_embd = model.transformer.h[0].ln_1.weight.shape[0]
        # Replace ln_1 and ln_2 with FakeLayerNorm
        for i in range(n_layers):
            block = model.transformer.h[i]
            ln_1_weight = block.ln_1.weight.detach()
            ln_1_bias = block.ln_1.bias.detach()

            block.ln_1 = FakeLayerNorm(
                ndim=n_embd,
                layer=f"blocks.{i}.hook_resid_pre",
                bias=block.ln_1.bias is not None,
            )
            block.ln_1.weight = nn.Parameter(ln_1_weight)
            block.ln_1.bias = nn.Parameter(ln_1_bias)

            ln_2_weight = block.ln_2.weight.detach()
            ln_2_bias = block.ln_2.bias.detach()

            block.ln_2 = FakeLayerNorm(
                ndim=n_embd,
                layer=f"blocks.{i}.hook_resid_mid",
                bias=block.ln_2.bias is not None,
            )
            block.ln_2.weight = nn.Parameter(ln_2_weight)
            block.ln_2.bias = nn.Parameter(ln_2_bias)

        # Replace ln_f with FakeLayerNorm
        ln_f = model.transformer.ln_f
        ln_f_weight = ln_f.weight.detach()
        ln_f_bias = ln_f.bias.detach()
        model.transformer.ln_f = FakeLayerNorm(
            ndim=n_embd,
            layer=f"blocks.11.hook_resid_post",
            bias=model.transformer.ln_f.bias is not None,
        )
        model.transformer.ln_f.weight = nn.Parameter(ln_f_weight)
        model.transformer.ln_f.bias = nn.Parameter(ln_f_bias)

    # Replace all LayerNorm instances with FakeLayerNorm
    replace_layernorm_with_fake_layernorm(model, std=1.0)

    def disable_ln_2(block_index):
        model.transformer.h[block_index].ln_2.mode = "fake"
        print(f"disabled ln_2 for block {block_index}")

    def disable_ln_1qk(block_index):
        model.transformer.h[block_index].ln_1.mode = "fake"
        print(f"disabled ln_1 for block {block_index}")

    def disable_ln_1v(block_index):
        model.transformer.h[block_index].ln_1.attn_v_mode = "fake"
        print(f"disabled ln_1v for block {block_index}")

    def disable_ln_f():
        model.transformer.ln_f.mode = "fake"
        print("disabled ln_f")

    def disable_eot_std(block_index):
        model.transformer.h[block_index].ln_1.bos_std = model.transformer.h[
            block_index
        ].ln_1.average_std
        print(f"disabled eot std for block {block_index}")

    def disable_bos_std(block_index):
        model.transformer.h[block_index].ln_1.average_std[0] = model.transformer.h[
            block_index
        ].ln_1.average_std[1]
        model.transformer.h[block_index].ln_1.bos_std[0] = model.transformer.h[
            block_index
        ].ln_1.bos_std[1]
        print(f"disabled bos std for block {block_index}")

    gap_ln2 = 20
    gap_ln1qk = 20
    gap_ln1v = 30
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0

    n_ln2 = 200
    n_ln1qk = n_ln2 + 12 * gap_ln2
    n_ln1v = n_ln1qk + 12 * gap_ln1qk
    n_lnf = n_ln1v + 12 * gap_ln1v
    n_eot = n_lnf + 20
    n_bos = n_eot + 100

    class LNRemover:
        """
        Schedules the "removal" of LayerNorms by calling the disable function.
        """

        def __init__(self, start_step, layer_gap_steps, function):
            self.n_layers = 12
            self.start_step = start_step
            self.layer_gap_steps = layer_gap_steps
            self.function = function

        def __call__(self, step):
            if self.layer_gap_steps is None:
                # LN functions without layer (i.e. ln_f)
                if step == self.start_step:
                    print(f"step {step}")
                    self.function()
            elif self.layer_gap_steps == 0:
                # LNs where we disable all layers at once
                if step == self.start_step:
                    print(f"step {step}")
                    [self.function(i) for i in range(self.n_layers)]
            elif (step - self.start_step) % self.layer_gap_steps == 0:
                # LNs where we disable one layer at a time
                layer_index = (step - self.start_step) // self.layer_gap_steps
                if 0 <= layer_index < self.n_layers:
                    print(f"step {step}")
                    self.function(layer_index)
            else:
                # Not at a step where we need to disable a LN
                pass

        def log(self, wandb):
            name = self.function.__name__
            wandb.log(
                {
                    f"{name}.start_step": self.start_step,
                    f"{name}.layer_gap_steps": self.layer_gap_steps,
                }
            )

    class LNRemoverCallback(TrainerCallback):
        def __init__(self, ln_removers):
            self.ln_removers = ln_removers

        def on_step_begin(self, args, state, control, **kwargs):
            # Iterate over the ln_removers
            for ln_remover in self.ln_removers:
                ln_remover(state.global_step)
            return control

    ln_removers = [
        LNRemover(n_ln2, gap_ln2, disable_ln_2),
        LNRemover(n_ln1qk, gap_ln1qk, disable_ln_1qk),
        LNRemover(n_ln1v, gap_ln1v, disable_ln_1v),
        LNRemover(n_lnf, gap_lnf, disable_ln_f),
        LNRemover(n_eot, gap_eot, disable_eot_std),
        LNRemover(n_bos, gap_bos, disable_bos_std),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        callbacks=[LNRemoverCallback(ln_removers)],
    )

    wandb.init(project="hf-remove-ln")
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    tokenized = prepare_dataset()
    model = load_model()

    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune model with or without layer normalization"
    )
    parser.add_argument(
        "--mode",
        choices=["with_ln", "without_ln", "both"],
        required=True,
        help="Finetuning mode",
    )
    args = parser.parse_args()

    if args.mode == "with_ln":
        finetune_with_ln(model, tokenized)
    elif args.mode == "without_ln":
        finetune_without_ln(model, tokenized)
    elif args.mode == "both":
        finetune_with_ln(model, tokenized)
        finetune_without_ln(model, tokenized)
    finetune_with_ln(model, tokenized)
    finetune_without_ln(model, tokenized)
