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
from config import FINETUNE_CONFIGS
from prepare_dataset import prepare_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import types
from datasets import load_dataset
from std_dicts import std_dicts
from pydantic import BaseModel, Field
from typing import Dict, Optional, Callable, Any
from pile_eval import preprocess_pile_dataset, evaluate_model_on_pile, convert_for_trainer

# TODO: support multi-GPU. If multiple GPUs are available, this will select the first one.
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)
# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

_USE_WANDB = True

# Default to gpt2 std_dicts
std_dict = std_dicts["gpt2"]["std_dict"]
std_bos_dict = std_dicts["gpt2"]["std_bos_dict"]


class FakeLayerNorm(nn.Module):
    """LayerNorm using a fixed std instead of the actual standard deviation."""

    def __init__(self, ndim, layer, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        # Flag whether the LayerNorm is enabled ("real") or disabled ("fake")
        self.mode = "real"
        self.attn_v_mode = "real"
        self.average_std = torch.ones(ndim, device=device) * std_dict[layer]
        self.average_std[0] = std_bos_dict[layer]
        self.bos_std = torch.ones(ndim, device=device) * std_bos_dict[layer]

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


def load_model(model_name="gpt2", remove_ln=False):
    model = transformers.GPT2LMHeadModel.from_pretrained(
        model_name,
        cache_dir=f"{model_name}_cache",
        config=transformers.GPT2Config.from_pretrained(
            model_name, dropout=0.0, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0
        ),
    )

    def replace_layernorm_with_fake_layernorm(model):
        n_layers = model.config.n_layer
        n_embd = model.transformer.h[0].ln_1.weight.shape[0]
        
        # Replace ln_1 and ln_2 with FakeLayerNorm
        for i in range(n_layers):
            block = model.transformer.h[i]
            
            # Store original weights
            ln_1_weight = block.ln_1.weight.clone().detach()
            ln_1_bias = block.ln_1.bias.clone().detach() if block.ln_1.bias is not None else None
            ln_2_weight = block.ln_2.weight.clone().detach()
            ln_2_bias = block.ln_2.bias.clone().detach() if block.ln_2.bias is not None else None
            
            # Replace with FakeLayerNorm
            block.ln_1 = FakeLayerNorm(ndim=n_embd, layer=f"blocks.{i}.hook_resid_pre", bias=block.ln_1.bias is not None)
            block.ln_2 = FakeLayerNorm(ndim=n_embd, layer=f"blocks.{i}.hook_resid_mid", bias=block.ln_2.bias is not None)
            
            # Restore weights
            block.ln_1.weight = nn.Parameter(ln_1_weight)
            if ln_1_bias is not None:
                block.ln_1.bias = nn.Parameter(ln_1_bias)
            block.ln_2.weight = nn.Parameter(ln_2_weight)
            if ln_2_bias is not None:
                block.ln_2.bias = nn.Parameter(ln_2_bias)
            
            # Monkey patch the attention forward to handle separate ln1_qk and ln1_v
            def make_attn_forward(old_forward):
                def new_forward(self, x_qk, x_v):
                    B, T, C = x_qk.size()

                    # Calculate q,k from x_qk and v from x_v
                    # Correct matrix multiplication order and reshape
                    qkv_qk = x_qk.reshape(B*T, C) @ self.c_attn.weight[:, :2*C]  # For Q and K
                    v = x_v.reshape(B*T, C) @ self.c_attn.weight[:, 2*C:]  # For V
                    
                    # Split qkv into q and k
                    q, k = qkv_qk.split(C, dim=1)
                    
                    if self.c_attn.bias is not None:
                        q = q + self.c_attn.bias[:C]
                        k = k + self.c_attn.bias[C:2*C]
                        v = v + self.c_attn.bias[2*C:]

                    # Reshape
                    q = q.view(B, T, self.num_heads, C//self.num_heads).transpose(1, 2)
                    k = k.view(B, T, self.num_heads, C//self.num_heads).transpose(1, 2)
                    v = v.view(B, T, self.num_heads, C//self.num_heads).transpose(1, 2)

                    # Causal self-attention
                    y = F.scaled_dot_product_attention(
                        q, k, v,
                        dropout_p=self.attn_dropout.p,
                        is_causal=True
                    )
                    y = y.transpose(1, 2).contiguous().view(B, T, C)

                    # Output projection
                    y = self.c_proj(y)
                    y = self.resid_dropout(y)
                    return y

                return types.MethodType(new_forward, block.attn)

            block.attn.forward = make_attn_forward(block.attn.forward)
            
            # Monkey patch the forward method of the block
            def make_forward(old_forward):
                def new_forward(self, x, *args, **kwargs):
                    # Get EOT mask from the input
                    eot_mask = kwargs.pop('eot_mask', None)
                    
                    # Calculate LN'd x for Q and K
                    x_qk = self.ln_1(x)
                    # Calculate LN'd x for V
                    x_v = self.ln_1(x, attn_v=True)
                    
                    if eot_mask is not None:
                        x_v_eot = self.ln_1(x, std_type='bos', attn_v=True)
                        x_v[eot_mask] = x_v_eot[eot_mask]
                        del x_v_eot
                    
                    # Modify attention call to use both x_qk and x_v
                    attn_output = self.attn(x_qk, x_v)
                    x = x + attn_output
                    x = x + self.mlp(self.ln_2(x))
                    return x
                return types.MethodType(new_forward, block)
            
            block.forward = make_forward(block.forward)

        # Replace ln_f with FakeLayerNorm
        ln_f = model.transformer.ln_f
        ln_f_weight = ln_f.weight.clone().detach()
        ln_f_bias = ln_f.bias.clone().detach() if ln_f.bias is not None else None
        
        model.transformer.ln_f = FakeLayerNorm(
            ndim=n_embd,
            layer=f"blocks.{n_layers-1}.hook_resid_post",
            bias=ln_f.bias is not None
        )
        model.transformer.ln_f.weight = nn.Parameter(ln_f_weight)
        if ln_f_bias is not None:
            model.transformer.ln_f.bias = nn.Parameter(ln_f_bias)

        # Monkey patch the transformer's forward to include eot_mask
        def make_transformer_forward(old_forward):
            def new_forward(self, *args, **kwargs):
                # Extract input_ids from either kwargs or first positional arg
                input_ids = kwargs.get('input_ids', args[0] if args else None)
                
                # Create eot_mask if we have input_ids
                eot_mask = None
                if input_ids is not None:
                    eot_mask = input_ids == 50256
                
                # If args contains positional arguments that match kwargs, we should use kwargs only
                if args and isinstance(args[0], torch.Tensor):  # First arg is likely input_ids
                    kwargs['input_ids'] = args[0]
                    args = args[1:]  # Remove the first argument
                
                # Get embeddings
                hidden_states = self.wte(kwargs['input_ids'])
                position_ids = torch.arange(0, hidden_states.size(1), dtype=torch.long, device=hidden_states.device)
                hidden_states = hidden_states + self.wpe(position_ids)
                hidden_states = self.drop(hidden_states)

                # Forward through blocks with eot_mask
                for block in self.h:
                    hidden_states = block(hidden_states, eot_mask=eot_mask)
                
                hidden_states = self.ln_f(hidden_states)

                # Create BaseModelOutputWithPastAndCrossAttentions object
                return transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    cross_attentions=None,
                )

            return types.MethodType(new_forward, model.transformer)
        
        model.transformer.forward = make_transformer_forward(model.transformer.forward)

    if remove_ln:
        # Replace all LayerNorm instances with FakeLayerNorm
        replace_layernorm_with_fake_layernorm(model)
    
    return model

def finetune_with_ln(model, training_args, tokenized, data_collator, config, pile_eval_dataset=None):
    """Finetune model with layer normalization"""
    # Create multi-dataset dictionary if pile_eval_dataset is provided
    if pile_eval_dataset is not None:
        eval_datasets = {
            # "openwebtext": tokenized["test"],
            "pile10k": pile_eval_dataset
        }
    else:
        eval_datasets = tokenized["test"]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_datasets,
        data_collator=data_collator,
    )

    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )


def finetune_without_ln(model, training_args, tokenized, data_collator, config, pile_eval_dataset=None):
    """Finetune model without layer normalization"""
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
    model_name = config.model_name
    training_config = config
    
    ln_removers = [
        LNRemover(training_config.start_ln2, training_config.gap_ln2, disable_ln_2),
        LNRemover(training_config.start_ln1qk, training_config.gap_ln1qk, disable_ln_1qk),
        LNRemover(training_config.start_ln1v, training_config.gap_ln1v, disable_ln_1v),
        LNRemover(training_config.start_lnf, training_config.gap_lnf, disable_ln_f),
        LNRemover(training_config.start_eot, training_config.gap_eot, disable_eot_std),
        LNRemover(training_config.start_bos, training_config.gap_bos, disable_bos_std),
    ]

    # If resuming, apply all removals that should have happened up to resume_step
    if training_args.resume_from_checkpoint:
        try:
            checkpoint_path = training_args.resume_from_checkpoint
            resume_step = int(checkpoint_path.split("-")[-1])
            print(f"\nRetroactively applying LN removals up to step {resume_step}")
            for i in range(resume_step + 1):
                for ln_remover in ln_removers:
                    ln_remover(i)
            print("Finished applying retroactive LN removals\n")
        except Exception as e:
            print(f"Warning: Failed to extract step from checkpoint: {e}")

    callbacks = [
        LNRemoverCallback(ln_removers),
    ]

    # Create multi-dataset dictionary if pile_eval_dataset is provided
    if pile_eval_dataset is not None:
        eval_datasets = {
            # "openwebtext": tokenized["test"],
            "pile10k": pile_eval_dataset
        }
    else:
        eval_datasets = tokenized["test"]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )

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
        choices=list(FINETUNE_CONFIGS.keys()),
        required=True,
        help="Training configuration to use",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the model to disk",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        required=False,
        help="Checkpoint to resume from",
    )
    args = parser.parse_args()

    # Get model name from config
    config = FINETUNE_CONFIGS[args.config]
    model_name = config.model_name

    # Update std_dict and std_bos_dict if model is not gpt2
    global std_dict, std_bos_dict
    std_dict = std_dicts[model_name]["std_dict"]
    std_bos_dict = std_dicts[model_name]["std_bos_dict"]
    
    # Prepare datasets
    tokenized, data_collator = prepare_dataset(model_name)
    
    # Initialize model
    model = load_model(model_name, remove_ln=args.mode == "without_ln")
    
    # Initialize Pile-10k dataset once at the beginning
    print("Preparing Pile-10k evaluation dataset...")

    processed_examples, pile_tokenizer = preprocess_pile_dataset(
        "pile-10k", model_name, num_samples=config.num_eval_samples
    )
    
    pile_eval_dataset = convert_for_trainer(
        processed_examples, 
        pile_tokenizer,
        model_name=model_name,
        num_samples=config.num_eval_samples
    )

    # Training arguments with evaluation settings
    training_args = TrainingArguments(
        output_dir="./results",
        bf16=True,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_safetensors=False,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
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
        dataloader_persistent_workers=True,
        save_steps=config.save_steps,
        save_total_limit=12,
        eval_accumulation_steps=1,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        load_best_model_at_end=False,
    )

    # Pass the pile_eval_dataset to the appropriate training function
    if args.mode == "with_ln":
        finetune_with_ln(model, training_args, tokenized, data_collator, config, pile_eval_dataset)
        if args.save:
            model.save_pretrained("model-with-ln")
            tokenizer.save_pretrained("model-with-ln")
    elif args.mode == "without_ln":
        finetune_without_ln(model, training_args, tokenized, data_collator, config, pile_eval_dataset)
        if args.save:
            model.save_pretrained("model-without-ln")
            tokenizer.save_pretrained("model-without-ln")
    else:
        raise ValueError(f"Unknown mode {args.mode}")
    

if __name__ == "__main__":
    main()
