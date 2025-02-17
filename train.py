import os

import argparse
import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers
import wandb
from prepare_dataset import prepare_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from std_dicts import std_dicts
from pydantic import BaseModel, Field
from typing import Dict, Optional, Callable, Any

# TODO: support multi-GPU. If multiple GPUs are available, this will select the first one.
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)
# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

_USE_WANDB = True

class ArchitectureConfig(BaseModel):
    model_name: str
    n_layers: int

class TrainingConfig(BaseModel):
    base_batch_size: int
    max_steps: int
    block_size: int = 1024
    target_batch_tokens: int = Field(default=2**19, description="Desired total tokens per batch")
    warmup_steps: int = 100
    weight_decay: float = 0.01
    learning_rate: float = 6e-4

class LayerNormSchedule(BaseModel):
    gaps: Dict[str, Optional[int]]
    start_steps: Dict[str, Any]  # Mix of ints and lambdas

class FinetuneConfig(BaseModel):
    architecture: ArchitectureConfig
    training: TrainingConfig
    layernorm_schedule: LayerNormSchedule

# Define configurations
_FINETUNE_CONFIGS: Dict[str, FinetuneConfig] = {
    "gpt2_standard": FinetuneConfig(
        architecture=ArchitectureConfig(
            model_name="gpt2",
            n_layers=12,
        ),
        training=TrainingConfig(
            base_batch_size=32,
            max_steps=1200,
        ),
        layernorm_schedule=LayerNormSchedule(
            gaps={
                "ln2": 2,
                "ln1qk": 2,
                "ln1v": 3,
                "lnf": None,
                "eot": 0,
                "bos": 0,
            },
            start_steps={
                "ln2": 20,
                "ln1qk": lambda n_ln2, gap_ln2: n_ln2 + 24 * gap_ln2,
                "ln1v": lambda n_ln1qk, gap_ln1qk: n_ln1qk + 24 * gap_ln1qk,
                "lnf": lambda n_ln1v, gap_ln1v: n_ln1v + 24 * gap_ln1v,
                "eot": lambda n_lnf: n_lnf + 2,
                "bos": lambda n_eot: n_eot + 10,
            }
        )
    ),
    "gpt2_medium_standard": FinetuneConfig(
        architecture=ArchitectureConfig(
            model_name="gpt2-medium",
            n_layers=24,
        ),
        training=TrainingConfig(
            base_batch_size=22,
            max_steps=1200,
            warmup_steps=10,
        ),
        layernorm_schedule=LayerNormSchedule(
            gaps={
                "ln2": 2,
                "ln1qk": 2,
                "ln1v": 3,
                "lnf": None,
                "eot": 0,
                "bos": 0,
            },
            start_steps={
                "ln2": 20,
                "ln1qk": lambda n_ln2, gap_ln2: n_ln2 + 24 * gap_ln2,
                "ln1v": lambda n_ln1qk, gap_ln1qk: n_ln1qk + 24 * gap_ln1qk,
                "lnf": lambda n_ln1v, gap_ln1v: n_ln1v + 24 * gap_ln1v,
                "eot": lambda n_lnf: n_lnf + 2,
                "bos": lambda n_eot: n_eot + 10,
            }
        )
    ),
    "gpt2_medium_extended": FinetuneConfig(
        architecture=ArchitectureConfig(
            model_name="gpt2-medium",
            n_layers=24,
        ),
        training=TrainingConfig(
            base_batch_size=16,
            max_steps=2400,
        ),
        layernorm_schedule=LayerNormSchedule(
            gaps={
                "ln2": 4,
                "ln1qk": 4,
                "ln1v": 6,
                "lnf": None,
                "eot": 0,
                "bos": 0,
            },
            start_steps={
                "ln2": 20,
                "ln1qk": lambda n_ln2, gap_ln2: n_ln2 + 24 * gap_ln2,
                "ln1v": lambda n_ln1qk, gap_ln1qk: n_ln1qk + 24 * gap_ln1qk,
                "lnf": lambda n_ln1v, gap_ln1v: n_ln1v + 24 * gap_ln1v,
                "eot": lambda n_lnf: n_lnf + 2,
                "bos": lambda n_eot: n_eot + 10,
            }
        )
    ),
    "gpt2-large": FinetuneConfig(
        architecture=ArchitectureConfig(
            model_name="gpt2-large",
            n_layers=36,
        ),
        training=TrainingConfig(
            base_batch_size=16,
            max_steps=1200,
        ),
        layernorm_schedule=LayerNormSchedule(
            gaps={
                "ln2": 2,
                "ln1qk": 2,
                "ln1v": 3,
                "lnf": None,
                "eot": 0,
                "bos": 0,
            },
            start_steps={
                "ln2": 20,
                "ln1qk": lambda n_ln2, gap_ln2: n_ln2 + 24 * gap_ln2,
                "ln1v": lambda n_ln1qk, gap_ln1qk: n_ln1qk + 24 * gap_ln1qk,
                "lnf": lambda n_ln1v, gap_ln1v: n_ln1v + 24 * gap_ln1v,
                "eot": lambda n_lnf: n_lnf + 2,
                "bos": lambda n_eot: n_eot + 10,
            }
        )
    ),
    "gpt2-xl": FinetuneConfig(
        architecture=ArchitectureConfig(
            model_name="gpt2-xl",
            n_layers=48,
        ),
        training=TrainingConfig(
            base_batch_size=8,
            max_steps=1200,
        ),
        layernorm_schedule=LayerNormSchedule(
            gaps={
                "ln2": 2,
                "ln1qk": 2,
                "ln1v": 3,
                "lnf": None,
                "eot": 0,
                "bos": 0,
            },
            start_steps={
                "ln2": 20,
                "ln1qk": lambda n_ln2, gap_ln2: n_ln2 + 24 * gap_ln2,
                "ln1v": lambda n_ln1qk, gap_ln1qk: n_ln1qk + 24 * gap_ln1qk,
                "lnf": lambda n_ln1v, gap_ln1v: n_ln1v + 24 * gap_ln1v,
                "eot": lambda n_lnf: n_lnf + 2,
                "bos": lambda n_eot: n_eot + 10,
            }
        )
    ),
    "gpt2_test": FinetuneConfig(
        architecture=ArchitectureConfig(
            model_name="gpt2",  # Smallest GPT-2 model
            n_layers=12,
        ),
        training=TrainingConfig(
            base_batch_size=1,     # Minimal batch size
            max_steps=10,          # Very few steps for testing
            block_size=512,        # Reduced context length
            target_batch_tokens=2**12,  # Much smaller effective batch
            warmup_steps=2,        # Minimal warmup
            weight_decay=0.01,
            learning_rate=6e-4,
        ),
        layernorm_schedule=LayerNormSchedule(
            gaps={
                "ln2": 1,          # Faster schedule for testing
                "ln1qk": 1,
                "ln1v": 1,
                "lnf": None,
                "eot": 0,
                "bos": 0,
            },
            start_steps={
                "ln2": 2,          # Start changes earlier
                "ln1qk": lambda n_ln2, gap_ln2: n_ln2 + 2 * gap_ln2,
                "ln1v": lambda n_ln1qk, gap_ln1qk: n_ln1qk + 2 * gap_ln1qk,
                "lnf": lambda n_ln1v, gap_ln1v: n_ln1v + 2 * gap_ln1v,
                "eot": lambda n_lnf: n_lnf + 1,
                "bos": lambda n_eot: n_eot + 1,
            }
        )
    ),
}

def construct_training_config(config: FinetuneConfig) -> dict:
    """Calculate derived training parameters from base config."""
    training = config.training
    
    # Calculate derived parameters
    batch_size = training.base_batch_size
    block_size = training.block_size
    desired_batch_size = training.target_batch_tokens / block_size
    grad_accum_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule steps
    ln_schedule = config.layernorm_schedule
    gaps = ln_schedule.gaps
    start_steps = ln_schedule.start_steps
    
    # Calculate dependent start steps in order
    n_ln2 = start_steps["ln2"]
    n_ln1qk = start_steps["ln1qk"](n_ln2, gaps["ln2"])
    n_ln1v = start_steps["ln1v"](n_ln1qk, gaps["ln1qk"])
    n_lnf = start_steps["lnf"](n_ln1v, gaps["ln1v"])
    n_eot = start_steps["eot"](n_lnf)
    n_bos = start_steps["bos"](n_eot)
    
    return {
        "max_steps": training.max_steps,
        "batch_size": batch_size,
        "block_size": block_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "warmup_steps": training.warmup_steps,
        "weight_decay": training.weight_decay,
        "learning_rate": training.learning_rate,
        "ln_schedule": {
            "gaps": gaps,
            "start_steps": {
                "ln2": n_ln2,
                "ln1qk": n_ln1qk,
                "ln1v": n_ln1v,
                "lnf": n_lnf,
                "eot": n_eot,
                "bos": n_bos,
            }
        }
    }

def load_model(model_name="gpt2"):
    model = transformers.GPT2LMHeadModel.from_pretrained(
        model_name,
        cache_dir=f"{model_name}_cache",
        config=transformers.GPT2Config.from_pretrained(
            model_name, dropout=0.0, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0
        ),
    )
    return model


def finetune_with_ln(model, training_args, tokenized, data_collator, config):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
    )

    trainer.train()

def finetune_without_ln(model, training_args, tokenized, data_collator, config):

    class FakeLayerNorm(nn.Module):
        """LayerNorm using a fixed std instead of the actual standard deviation."""

        def __init__(self, ndim, layer, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(ndim))
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
            # Flag whether the LayerNorm is enabled ("real") or disabled ("fake")
            self.mode = "real"
            self.attn_v_mode = "real"
            # Get the correct std dictionaries for the current model
            model_name = model.config._name_or_path
            model_dicts = std_dicts[model_name]
            self.average_std = torch.ones(ndim, device=device) * model_dicts["std_dict"][layer]
            self.average_std[0] = model_dicts["std_bos_dict"][layer]
            self.bos_std = torch.ones(ndim, device=device) * model_dicts["std_bos_dict"][layer]

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
        # Get number of layers dynamically
        n_layers = len(model.transformer.h)
        n_embd = model.transformer.h[0].ln_1.weight.shape[0]
        
        # Replace ln_1 and ln_2 with FakeLayerNorm
        for i in range(n_layers):
            block = model.transformer.h[i]
            ln_1_weight = block.ln_1.weight.clone().detach()
            ln_1_bias = block.ln_1.bias.clone().detach()

            block.ln_1 = FakeLayerNorm(
                ndim=n_embd,
                layer=f"blocks.{i}.hook_resid_pre",
                bias=block.ln_1.bias is not None,
            )
            block.ln_1.weight = nn.Parameter(ln_1_weight)
            block.ln_1.bias = nn.Parameter(ln_1_bias)

            ln_2_weight = block.ln_2.weight.clone().detach()
            ln_2_bias = block.ln_2.bias.clone().detach()

            block.ln_2 = FakeLayerNorm(
                ndim=n_embd,
                layer=f"blocks.{i}.hook_resid_mid",
                bias=block.ln_2.bias is not None,
            )
            block.ln_2.weight = nn.Parameter(ln_2_weight)
            block.ln_2.bias = nn.Parameter(ln_2_bias)

        # Replace ln_f with FakeLayerNorm
        ln_f = model.transformer.ln_f
        ln_f_weight = ln_f.weight.clone().detach()
        ln_f_bias = ln_f.bias.clone().detach()
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

    class LNRemover:
        """
        Schedules the "removal" of LayerNorms by calling the disable function.
        """

        def __init__(self, start_step, layer_gap_steps, function):
            self.n_layers = len(model.transformer.h)  # Get layers dynamically
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
            if _USE_WANDB:
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
                ln_remover.log(wandb)
            return control

    # Get schedule from config
    model_name = config.architecture.model_name
    training_config = construct_training_config(config)
    ln_schedule = training_config["ln_schedule"]
    
    ln_removers = [
        LNRemover(ln_schedule["start_steps"]["ln2"], ln_schedule["gaps"]["ln2"], disable_ln_2),
        LNRemover(ln_schedule["start_steps"]["ln1qk"], ln_schedule["gaps"]["ln1qk"], disable_ln_1qk),
        LNRemover(ln_schedule["start_steps"]["ln1v"], ln_schedule["gaps"]["ln1v"], disable_ln_1v),
        LNRemover(ln_schedule["start_steps"]["lnf"], ln_schedule["gaps"]["lnf"], disable_ln_f),
        LNRemover(ln_schedule["start_steps"]["eot"], ln_schedule["gaps"]["eot"], disable_eot_std),
        LNRemover(ln_schedule["start_steps"]["bos"], ln_schedule["gaps"]["bos"], disable_bos_std),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        callbacks=[LNRemoverCallback(ln_removers)],
    )

    trainer.train()

def main():
    parser = argparse.ArgumentParser(
        description="Finetune model with or without layer normalization"
    )
    parser.add_argument(
        "--mode",
        choices=["with_ln", "without_ln"],
        required=True,
        help="Finetuning mode",
    )
    parser.add_argument(
        "--config",
        choices=list(_FINETUNE_CONFIGS.keys()),
        required=True,
        help="Training configuration to use",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the model to disk",
    )
    args = parser.parse_args()

    # Get model name from config
    config = _FINETUNE_CONFIGS[args.config]
    model_name = config.architecture.model_name

    tokenized, data_collator = prepare_dataset(model_name)
    model = load_model(model_name)

    if args.save:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

    # Construct training config
    training_config = construct_training_config(config)
    
    training_args = TrainingArguments(
        output_dir="./results",
        bf16=True, # Use for mixed precision training
        save_safetensors=False,
        max_steps=training_config["max_steps"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        warmup_steps=training_config["warmup_steps"],
        weight_decay=training_config["weight_decay"],
        learning_rate=training_config["learning_rate"],
        max_grad_norm=1.0,
        logging_dir="./logs",
        prediction_loss_only=True,
        lr_scheduler_type="cosine",
        report_to="wandb" if _USE_WANDB else "none",
        run_name=f"{args.config}-{args.mode}",
        logging_steps=1,
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=True
    )

    if args.mode == "with_ln":
        finetune_with_ln(model, training_args, tokenized, data_collator, config)
        if args.save:
            model.save_pretrained("model-with-ln")
            tokenizer.save_pretrained("model-with-ln")
    elif args.mode == "without_ln":
        finetune_without_ln(model, training_args, tokenized, data_collator, config)
        if args.save:
            model.save_pretrained("model-without-ln")
            tokenizer.save_pretrained("model-without-ln")
    else:
        raise ValueError(f"Unknown mode {args.mode}")
    

if __name__ == "__main__":
    main()
