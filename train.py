"""
Feature flags:

EXP_CORRECT_BOS: Use correct BOS special treatment
EXP_RECOMPUTE_STD_ON_FAKE: Recompute std when FakeLayerNorm is fake - will recompute std on every forward pass
EXP_RECOMPUTE_STD_ON_REAL: Recompute std when FakeLayerNorm is real - will freeze std when FakeLayerNorm goes fake
"""
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
from datetime import datetime
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
from devtools import pprint

# TODO: support multi-GPU. If multiple GPUs are available, this will select the first one.
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)
# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

_USE_WANDB = True

class FakeLayerNorm(nn.Module):
    """LayerNorm using a fixed std instead of the actual standard deviation."""

    def __init__(self, n_embd, n_ctx, layer, bias, init_average_std, init_bos_std):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
        
        # Flag whether the LayerNorm is enabled ("real") or disabled ("fake")
        self.is_fake = False
        self.attn_v_is_fake = False
        self.layer = layer
        self.n_embd = n_embd
        self.n_ctx = n_ctx
        self.bos_special_treatment = True

        self.real_average_std = init_average_std
        self.real_bos_std = init_bos_std

        if os.environ.get("EXP_CORRECT_BOS", "0") == "1":
            std_dim = n_ctx
        else:
            std_dim = n_embd
            
        # Register non-parameter tensors as buffers so they're saved in state_dict
        self.register_buffer("average_std", torch.ones(std_dim, device=device) * init_average_std)
        self.register_buffer("bos_std", torch.ones(std_dim, device=device) * init_bos_std)
        
        # Special handling for position 0
        self.average_std[0] = init_bos_std

        self.iteration = 0
        self.update_freq = 1
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # Call the parent method to save parameters and buffers
        super()._save_to_state_dict(destination, prefix, keep_vars)
        
        # Add custom attributes to state dict
        destination[prefix + 'is_fake'] = torch.tensor(self.is_fake)
        destination[prefix + 'attn_v_is_fake'] = torch.tensor(self.attn_v_is_fake)
        destination[prefix + 'real_average_std'] = torch.tensor(self.real_average_std)
        destination[prefix + 'real_bos_std'] = torch.tensor(self.real_bos_std)
        destination[prefix + 'bos_special_treatment'] = torch.tensor(self.bos_special_treatment)
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Load custom attributes from state dict
        for key, attr in [
            ('is_fake', 'is_fake'),
            ('attn_v_is_fake', 'attn_v_is_fake'),
            ('real_average_std', 'real_average_std'),
            ('real_bos_std', 'real_bos_std'),
            ('bos_special_treatment', 'bos_special_treatment'),
        ]:
            full_key = prefix + key
            if full_key in state_dict:
                value = state_dict[full_key]
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value
                setattr(self, attr, value)
                # Remove from state_dict to avoid unexpected key warnings
                del state_dict[full_key]
            elif strict:
                missing_keys.append(full_key)
        
        # Call the parent method to load parameters and buffers
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, input, std_type="avg", attn_v=False):
        # We want all the enable / disable information to be in this class, but the class is re-used
        # for both the QK and V paths. Thus we add the attn_v flag to the call that is True only for
        # the V path. Thus we get to have flags `is_fake` and `attn_v_is_fake` to enable / disable the
        # LN for the QK and V paths separately.
        is_fake = self.attn_v_is_fake if attn_v else self.is_fake

        self.iteration += 1
        # Calculate the std of the input
        if self.iteration % self.update_freq == 0:
            self.iteration = 0
            self.real_average_std, self.real_bos_std = self.recompute_average_std(input)

        if is_fake:
            # Which std values to use: We use (1) average std (which is actually a vector of length
            # n_ctx for most of the time*) [a, b, b, ...] where a is the average std for position 0,
            # and b is the average std for all other positions. We also have the option to use (2)
            # the bos std [a, a, a, ...] for all positions, which we do if the input token is EOT.
            # Note that we could differentiate between EOT and BOS, but I didn't need it here.
            # *at the end (with disable_eot_std) we make the latter be like the former, and with
            # disable_bos_std we make both vectors to be [b, b, b, ...], equivalent to scalars.
            if os.environ.get("EXP_RECOMPUTE_STD_ON_FAKE", "0") == "1":
                with torch.no_grad():
                    self.sync_std()

            assert std_type in ["avg", "bos"]
            std = self.average_std if std_type == "avg" else self.bos_std
            if os.environ.get("EXP_CORRECT_BOS", "0") == "1":
                return (
                    (input - input.mean(-1, keepdim=True)) / std.view(1, -1, 1) * self.weight
                    + self.bias
                    if self.bias is not None
                    else input * self.weight
                )
            else:
                return (
                    (input - input.mean(-1, keepdim=True)) / std * self.weight
                    + self.bias
                    if self.bias is not None
                    else input * self.weight
                )
        else:
            if os.environ.get("EXP_RECOMPUTE_STD_ON_REAL", "0") == "1":
                with torch.no_grad():
                    self.sync_std()

            return F.layer_norm(
                input, self.weight.shape, self.weight, self.bias, 1e-5
            )
    
    def recompute_average_std(self, x):
        with torch.no_grad():
            std = x.var(dim=(0, -1))**0.5
            average_std = std.mean().detach().item()
            bos_std = std[0].detach().item()
        return average_std, bos_std
    
    def sync_std(self):
        """Sync the average and bos std values (that are used in the forward pass) with the real std values."""
        with torch.no_grad():
            # Create new tensors instead of modifying in-place because we don't want to track gradients for these
            # This is conditional on the recomputation mode
            if self.bos_special_treatment:
                new_average_std = self.average_std.clone()
                new_average_std[1:] = torch.ones_like(new_average_std[1:]) * self.real_average_std
                new_average_std[0] = self.real_bos_std
                self.average_std = new_average_std.detach().requires_grad_(False)
            else:
                self.average_std = torch.ones_like(self.average_std) * self.real_average_std
            
            # This will always enable BOS special treatment
            if self.bos_special_treatment:
                new_bos_std = torch.ones_like(self.bos_std) * self.real_bos_std
            else:
                new_bos_std = torch.ones_like(self.bos_std) * self.real_average_std
            self.bos_std = new_bos_std.detach().requires_grad_(False)
    
    def disable_eos_special_treatment(self):
        # Disable EOT special treatment, note that bos_std[0] will be set to average_std[1]
        # which is not always the same as bos_std[1]
        self.bos_std = self.average_std
    
    def disable_bos_special_treatment(self):
        # Disable BOS special treatment
        self.bos_special_treatment = False
        with torch.no_grad():
            # Special treatment of position 0 is now disabled
            self.average_std = torch.ones_like(self.average_std) * self.real_average_std
            self.bos_std = torch.ones_like(self.bos_std) * self.real_average_std


def load_model(model_name="gpt2", remove_ln=False):
    model = transformers.GPT2LMHeadModel.from_pretrained(
        model_name,
        cache_dir=f"{model_name}_cache",
        config=transformers.GPT2Config.from_pretrained(
            model_name, dropout=0.0, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0
        ),
    )

    def replace_layernorm_with_fake_layernorm(model, std_dict, std_bos_dict):
        n_layers = model.config.n_layer
        n_embd = model.transformer.h[0].ln_1.weight.shape[0]
        n_ctx = model.config.n_ctx

        # Check if model already has FakeLayerNorm
        has_fake_ln = hasattr(model.transformer.h[0].ln_1, 'is_fake')
        
        # Replace ln_1 and ln_2 with FakeLayerNorm
        for i in range(n_layers):
            block = model.transformer.h[i]
            
            # If model doesn't already have FakeLayerNorm, replace standard LayerNorm with FakeLayerNorm
            if not has_fake_ln:
                # Store original weights
                ln_1_weight = block.ln_1.weight.clone().detach()
                ln_1_bias = block.ln_1.bias.clone().detach() if block.ln_1.bias is not None else None
                ln_2_weight = block.ln_2.weight.clone().detach()
                ln_2_bias = block.ln_2.bias.clone().detach() if block.ln_2.bias is not None else None
                
                # Replace with FakeLayerNorm
                layer = f"blocks.{i}.hook_resid_pre"
                block.ln_1 = FakeLayerNorm(
                    n_embd=n_embd,
                    n_ctx=n_ctx,
                    layer=layer,
                    bias=block.ln_1.bias is not None,
                    init_average_std=std_dict[layer],
                    init_bos_std=std_bos_dict[layer])

                layer = f"blocks.{i}.hook_resid_mid"
                block.ln_2 = FakeLayerNorm(
                    n_embd=n_embd,
                    n_ctx=n_ctx,
                    layer=layer,
                    bias=block.ln_2.bias is not None,
                    init_average_std=std_dict[layer],
                    init_bos_std=std_bos_dict[layer])
                
                # Restore weights
                block.ln_1.weight = nn.Parameter(ln_1_weight)
                if ln_1_bias is not None:
                    block.ln_1.bias = nn.Parameter(ln_1_bias)
                block.ln_2.weight = nn.Parameter(ln_2_weight)
                if ln_2_bias is not None:
                    block.ln_2.bias = nn.Parameter(ln_2_bias)
            
            # Monkey patch the attention forward to handle separate ln1_qk and ln1_v
            # Only do this if we haven't already patched this model
            if not has_fake_ln or not hasattr(block.attn, '_patched_for_fake_ln'):
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
                block.attn._patched_for_fake_ln = True  # Mark as patched
            
            # Monkey patch the forward method of the block
            # Only do this if we haven't already patched this model
            if not has_fake_ln or not hasattr(block, '_patched_for_fake_ln'):
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
                block._patched_for_fake_ln = True  # Mark as patched

        # Replace ln_f with FakeLayerNorm if not already FakeLayerNorm
        if not has_fake_ln:
            ln_f = model.transformer.ln_f
            ln_f_weight = ln_f.weight.clone().detach()
            ln_f_bias = ln_f.bias.clone().detach() if ln_f.bias is not None else None
            
            layer = f"blocks.{n_layers-1}.hook_resid_post"
            model.transformer.ln_f = FakeLayerNorm(
                n_embd=n_embd,
                n_ctx=n_ctx,
                layer=layer,
                bias=ln_f.bias is not None,
                init_average_std=std_dict[layer],
                init_bos_std=std_bos_dict[layer]
            )
            model.transformer.ln_f.weight = nn.Parameter(ln_f_weight)
            if ln_f_bias is not None:
                model.transformer.ln_f.bias = nn.Parameter(ln_f_bias)

        # Monkey patch the transformer's forward to include eot_mask
        if not has_fake_ln or not hasattr(model.transformer, '_patched_for_fake_ln'):
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
            model.transformer._patched_for_fake_ln = True  # Mark as patched

    if remove_ln:
        # Replace all LayerNorm instances with FakeLayerNorm
        std_dict = std_dicts[model_name]["std_dict"]
        std_bos_dict = std_dicts[model_name]["std_bos_dict"]

        replace_layernorm_with_fake_layernorm(model, std_dict, std_bos_dict)
    
    return model


def finetune(model, training_args, tokenized, data_collator, config, pile_eval_dataset=None, remove_ln=False):
    """Finetune model with or without layer normalization"""
    def disable_ln_2(block_index):
        model.transformer.h[block_index].ln_2.is_fake = True
        print(f"disabled ln_2 for block {block_index}")

    def disable_ln_1qk(block_index):
        model.transformer.h[block_index].ln_1.is_fake = True
        print(f"disabled ln_1 for block {block_index}")

    def disable_ln_1v(block_index):
        model.transformer.h[block_index].ln_1.attn_v_is_fake = True
        print(f"disabled ln_1v for block {block_index}")

    def disable_ln_f():
        model.transformer.ln_f.is_fake = True
        print("disabled ln_f")

    def disable_eot_std(block_index):
        model.transformer.h[block_index].ln_1.disable_eos_special_treatment()
        print(f"disabled eot std for block {block_index}")

    def disable_bos_std(block_index):
        model.transformer.h[block_index].ln_1.disable_bos_special_treatment()
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
            self.applied_steps = set()  # Track which steps we've already applied

        def __call__(self, step):
            # Skip if we've already applied this step
            if step in self.applied_steps:
                return
                
            if self.layer_gap_steps is None:
                if step == self.start_step:
                    self.function()
                    self.log_event(step)
                    self.applied_steps.add(step)
            elif self.layer_gap_steps == 0:
                if step == self.start_step:
                    [self.function(i) for i in range(self.n_layers)]
                    self.log_event(step)
                    self.applied_steps.add(step)
            elif step >= self.start_step and (step - self.start_step) % self.layer_gap_steps == 0:
                layer_index = (step - self.start_step) // self.layer_gap_steps
                if 0 <= layer_index < self.n_layers:
                    self.function(layer_index)
                    self.log_event(step, layer_index)
                    self.applied_steps.add(step)
            else:
                pass

        def log_event(self, step, layer_index=None):
            if _USE_WANDB:
                event_name = f"{self.function.__name__}"
                if layer_index is not None:
                    event_name += f"_layer_{layer_index}"
                # Log a nan value - it will show up at the top of the graph in wandb
                wandb.log({event_name: float('nan')})

        def log(self, wandb):
            if _USE_WANDB:
                wandb.log(
                    {
                        f"{self.function.__name__}.start_step": self.start_step,
                        f"{self.function.__name__}.layer_gap_steps": self.layer_gap_steps,
                    }
                )

    class LNRemoverCallback(TrainerCallback):
        def __init__(self, ln_removers):
            self.ln_removers = ln_removers

        def on_step_begin(self, args, state, control, **kwargs):
            print(f"on_step_begin: {state.global_step}")
            # Iterate over the ln_removers
            for ln_remover in self.ln_removers:
                ln_remover(state.global_step)
                ln_remover.log(wandb)
            return control
    
    class LogFakeLayerNormState(TrainerCallback):
        def __ini__(self):
            pass

        def on_step_begin(self, args, state, control, **kwargs):
            """ Log to wandb: block number, mode, att_v_mode, average_std, bos_std, average_std[0], average_std[1], bos_std[0], bos_std[1] """
            if not _USE_WANDB:
                return control

            for i, block in enumerate(model.transformer.h):
                wandb.log({
                    f"block_{i}_ln_1_real_average_std": block.ln_1.real_average_std,
                    f"block_{i}_ln_1_real_bos_std": block.ln_1.real_bos_std,
                    f"block_{i}_ln_1_average_std_0": block.ln_1.average_std[0],
                    f"block_{i}_ln_1_average_std_1": block.ln_1.average_std[1],
                    f"block_{i}_ln_1_bos_std_0": block.ln_1.bos_std[0],
                    f"block_{i}_ln_1_bos_std_1": block.ln_1.bos_std[1],
                })

                wandb.log({
                    f"block_{i}_ln_2_real_average_std": block.ln_2.real_average_std,
                    f"block_{i}_ln_2_real_bos_std": block.ln_2.real_bos_std,
                    f"block_{i}_ln_2_average_std_0": block.ln_2.average_std[0],
                    f"block_{i}_ln_2_average_std_1": block.ln_2.average_std[1],
                    f"block_{i}_ln_2_bos_std_0": block.ln_2.bos_std[0],
                    f"block_{i}_ln_2_bos_std_1": block.ln_2.bos_std[1],
                })

            wandb.log({
                f"ln_f_real_average_std": model.transformer.ln_f.real_average_std,
                f"ln_f_real_bos_std": model.transformer.ln_f.real_bos_std,
                f"ln_f_average_std_0": model.transformer.ln_f.average_std[0],
                f"ln_f_average_std_1": model.transformer.ln_f.average_std[1],
                f"ln_f_bos_std_0": model.transformer.ln_f.bos_std[0],
                f"ln_f_bos_std_1": model.transformer.ln_f.bos_std[1],
            })

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
            
            # Extract step from checkpoint path
            if "-" in checkpoint_path:
                resume_step = int(checkpoint_path.split("-")[-1])
            else:
                # If checkpoint_path is a directory, look for checkpoint files
                import glob
                checkpoint_files = glob.glob(f"{checkpoint_path}/checkpoint-*")
                if checkpoint_files:
                    # Get the latest checkpoint
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("-")[-1]))
                    resume_step = int(latest_checkpoint.split("-")[-1])
                else:
                    # Default to 0 if no step found
                    resume_step = 0
            
            print(f"\nRetroactively applying LN removals up to step {resume_step}")
            
            # Apply all removals in chronological order
            for i in range(resume_step + 1):
                for ln_remover in ln_removers:
                    ln_remover(i)
                    
            print("Finished applying retroactive LN removals\n")
        except Exception as e:
            print(f"Warning: Failed to handle checkpoint: {e}")
            import traceback
            traceback.print_exc()

    callbacks = [
        LogFakeLayerNormState(),
    ]
    if remove_ln:
        callbacks.append(LNRemoverCallback(ln_removers))

    # Create multi-dataset dictionary if pile_eval_dataset is provided
    if pile_eval_dataset is not None:
        eval_datasets = {
            # "openwebtext": tokenized["test"],
            "pile": pile_eval_dataset
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
        default="without_ln",
        help="Finetuning mode",
    )
    parser.add_argument(
        "--config",
        choices=list(FINETUNE_CONFIGS.keys()),
        default="gpt2_test",
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

    # Forcing to use only 1 GPU. Otherwise, tensors end up on different devices.
    # Fixing this is an open TODO but not a priority.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected ({torch.cuda.device_count()}). Forcing single GPU usage.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Get model name from config
    config = FINETUNE_CONFIGS[args.config]
    model_name = config.model_name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare datasets
    tokenized, data_collator = prepare_dataset(model_name)
    
    # Initialize Pile-apollo dataset once at the beginning
    print("Preparing Pile-apollo evaluation dataset...")
    pile_eval_dataset = None
    if os.environ.get("EVAL", "0") == "1":
        processed_examples, pile_tokenizer = preprocess_pile_dataset(
            "pile-apollo", model_name, num_samples=config.num_eval_samples
        )
        
        pile_eval_dataset = convert_for_trainer(
            processed_examples, 
            pile_tokenizer,
            model_name=model_name,
            num_samples=config.num_eval_samples
        )

    # Training arguments with evaluation settings
    output_dir = f"results/{model_name}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
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
        eval_steps=config.eval_steps,
        load_best_model_at_end=False,
    )

    # Initialize model
    model = load_model(model_name, remove_ln=args.mode == "without_ln")

    print("Begin training")
    print(model)
    print(training_args)
    pprint(config)

    # Pass the pile_eval_dataset to the appropriate training function
    finetune(
        model,
        training_args,
        tokenized,
        data_collator,
        config,
        pile_eval_dataset,
        remove_ln=args.mode == "without_ln"
    )
    if args.save:
        # Create a save directory with datetime
        save_dir = f"saved-models/{model_name}-{args.mode}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model and tokenizer saved to {save_dir}")
    

if __name__ == "__main__":
    main()
