"""
Feature flags:

EXP_CORRECT_BOS: Use correct BOS special treatment
EXP_RECOMPUTE_STD_ON_FAKE: Recompute std when FakeLayerNorm is fake - will recompute std on every forward pass
EXP_RECOMPUTE_STD_ON_REAL: Recompute std when FakeLayerNorm is real - will freeze std when FakeLayerNorm goes fake
EXP_NON_BOS_AVERAGE_STD: Compute average std based on input[1:] (i.e. non-BOS) positions
"""
import os

import argparse
import datasets
import gc
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

class TensorDeque:
    """
    A fixed-size FIFO buffer for tensors, emulating a deque using torch tensors.
    """
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = torch.zeros(maxlen)
        self.index = 0
        self.full = False

    def append(self, x):
        self.buffer[self.index] = x
        self.index = (self.index + 1) % self.maxlen
        if self.index == 0:
            self.full = True

    def get_mean(self):
        if self.full:
            return self.buffer.mean()
        else:
            return self.buffer[:self.index].mean()
class CustomTrainer(Trainer):
    """Trainer with auxiliary loss to encourage uniform residual norms."""
    
    def __init__(self, *args, aux_loss_weight=0.1, **kwargs):
        """
        CustomTrainer with auxiliary loss to encourage uniform residual norms.
        
        Args:
            aux_loss_weight: Weight of the auxiliary loss for uniform residual norms.
                             Set to 0 to disable the auxiliary loss.
        """
        super().__init__(*args, **kwargs)
        self.aux_loss_weight = aux_loss_weight
        self.pre_ln_f_activations = None
        
        # Register a hook to capture activations if we're using auxiliary loss
        if self.aux_loss_weight > 0 and hasattr(self.model, 'transformer'):
            # Register forward hook to capture activations before ln_f
            def pre_ln_f_hook(module, input):
                # Store the input to ln_f (which is what we want to normalize)
                self.pre_ln_f_activations = input[0]
                return input
                
            # Register the hook on the final layer norm
            self.model.transformer.ln_f.register_forward_pre_hook(pre_ln_f_hook)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CE loss with an auxiliary loss to encourage uniform residual norms."""
        # Regular forward pass - hook will capture activations before ln_f
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Skip auxiliary loss calculation if weight is 0 or no activations captured
        if self.aux_loss_weight == 0 or self.pre_ln_f_activations is None:
            return (loss, outputs) if return_outputs else loss
        
        # Get input_ids for identifying special tokens
        input_ids = inputs["input_ids"]
        
        # Identify BOS and EOS tokens
        # BOS is the first token (position 0)
        bos_positions = torch.zeros_like(input_ids, dtype=torch.bool)
        bos_positions[:, 0] = True
        
        # For GPT-2, the EOS token ID is 50256
        eos_token_id = 50256  # GPT-2 specific EOS token ID
        eos_positions = (input_ids == eos_token_id)
        
        # Create a mask for non-BOS/EOS positions
        non_special_positions = ~(bos_positions | eos_positions)
        
        # Calculate norm statistics from the captured activations
        # We need to reshape the activations to match the input_ids shape for masking
        hidden_states = self.pre_ln_f_activations
        
        # Compute std along feature dimension for each position
        with torch.no_grad():  # Only for computing statistics, not affecting gradients
            mean = hidden_states.mean(dim=-1, keepdim=True)
            # Compute variance 
            var = ((hidden_states - mean) ** 2).mean(dim=-1)
            # Compute standard deviation
            std = torch.sqrt(var + 1e-5)  # Adding epsilon for numerical stability
            
            # Get target std from non-special positions
            # Create flattened masks aligned with the std tensor
            batch_size, seq_len = input_ids.size()
            flat_std = std.view(-1)
            flat_non_special = non_special_positions.view(-1)
            
            # Extract valid standard deviations using the mask
            valid_stds = flat_std[flat_non_special]
            
            if valid_stds.numel() > 0:
                target_std = valid_stds.mean()
            else:
                # Fallback if no valid positions
                target_std = torch.tensor(1.0, device=std.device)
        
        # Now compute the loss with gradients
        # Recompute mean and std with gradient tracking enabled
        mean_with_grad = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean_with_grad
        var_with_grad = (centered ** 2).mean(dim=-1)
        std_with_grad = torch.sqrt(var_with_grad + 1e-5)
        
        # Calculate auxiliary loss - MSE to encourage uniform norms
        # Computing this using the gradable tensors
        aux_loss = self.aux_loss_weight * ((std_with_grad - target_std) ** 2).mean()
        
        # Free up memory - important to avoid OOM errors
        self.pre_ln_f_activations = None
        
        # Log statistics if using wandb
        if _USE_WANDB:
            try:
                is_main_process = torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True
            except:
                is_main_process = True
                
            if is_main_process:
                # Compute statistics for logging using the already computed values
                with torch.no_grad():
                    flat_bos = bos_positions.view(-1)
                    flat_eos = eos_positions.view(-1)
                    
                    bos_std_val = flat_std[flat_bos].mean().item() if flat_bos.any() else 0
                    eos_std_val = flat_std[flat_eos].mean().item() if flat_eos.any() else 0
                    
                    wandb.log({
                        "target_std": target_std.item(),
                        "avg_std": std.mean().item(),
                        "bos_std": bos_std_val,
                        "eos_std": eos_std_val,
                        "aux_loss": aux_loss.item(),
                        "main_loss": loss.item()
                    })
        
        # Free up memory for other tensors we no longer need
        del std, var
        if _USE_WANDB:
            with torch.no_grad():
                del flat_bos, flat_eos
        torch.cuda.empty_cache()  # Explicitly clear cache if using CUDA
        gc.collect() # Collect garbage to free up memory sometimes this does magic
        
        # Add auxiliary loss to main loss
        loss = loss + aux_loss
        
        return (loss, outputs) if return_outputs else loss

class FakeLayerNorm(nn.Module):
    """LayerNorm using a fixed std instead of the actual standard deviation."""

    def __init__(self, n_embd, n_ctx, layer, bias, init_average_std, init_bos_std, grad_acc_steps=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
        
        # Store layer information
        self.layer = layer
        self.n_embd = n_embd
        self.n_ctx = n_ctx
        
        # Register all flags as buffers so they're automatically included in state_dict
        # Using naming without underscore to be compatible with old checkpoints
        self.register_buffer("is_fake", torch.tensor(False))
        self.register_buffer("attn_v_is_fake", torch.tensor(False))
        self.register_buffer("bos_special_treatment", torch.tensor(True))
        self.register_buffer("real_average_std", torch.tensor(float(init_average_std)))
        self.register_buffer("real_bos_std", torch.tensor(float(init_bos_std)))
        self.register_buffer("grad_acc_steps", torch.tensor(grad_acc_steps))
        self.register_buffer("global_step", torch.tensor(0))
        self.moving_std = TensorDeque(grad_acc_steps)
        self.moving_std_bos = TensorDeque(grad_acc_steps)

        if os.environ.get("EXP_CORRECT_BOS", "0") == "1":
            std_dim = n_ctx
        else:
            std_dim = n_embd
            
        # Register non-parameter tensors as buffers so they're saved in state_dict
        self.register_buffer("average_std_buffer", torch.ones(std_dim, device=device) * init_average_std)
        self.register_buffer("bos_std_buffer", torch.ones(std_dim, device=device) * init_bos_std)
        
        # Special handling for position 0
        self.average_std_buffer[0] = init_bos_std

        self.iteration = 0
        self.update_freq = 10
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Call parent method to load most of the state
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
        # Remove successfully loaded buffer keys from missing_keys
        for key in [prefix + "is_fake", prefix + "attn_v_is_fake", prefix + "bos_special_treatment"]:
            if key in missing_keys:
                missing_keys.remove(key)
    
    # Properties with _prop suffix to avoid name conflicts with buffers
    @property
    def is_fake_prop(self):
        return bool(self.is_fake.item())
    
    @is_fake_prop.setter
    def is_fake_prop(self, value):
        self.is_fake.fill_(bool(value))
    
    @property
    def attn_v_is_fake_prop(self):
        return bool(self.attn_v_is_fake.item())
    
    @attn_v_is_fake_prop.setter
    def attn_v_is_fake_prop(self, value):
        self.attn_v_is_fake.fill_(bool(value))
    
    @property
    def bos_special_treatment_prop(self):
        return bool(self.bos_special_treatment.item())
    
    @bos_special_treatment_prop.setter
    def bos_special_treatment_prop(self, value):
        self.bos_special_treatment.fill_(bool(value))
    
    @property
    def real_average_std_prop(self):
        return float(self.real_average_std.item())
    
    @real_average_std_prop.setter
    def real_average_std_prop(self, value):
        self.real_average_std.fill_(float(value))
    
    @property
    def real_bos_std_prop(self):
        return float(self.real_bos_std.item())
    
    @real_bos_std_prop.setter
    def real_bos_std_prop(self, value):
        self.real_bos_std.fill_(float(value))
    
    def __repr__(self):
        """Return a string representation of the FakeLayerNorm's current state."""
        # Force re-read of the tensor values to ensure we have the latest state
        is_fake_value = bool(self.is_fake.item())
        attn_v_is_fake_value = bool(self.attn_v_is_fake.item())
        mode = "fake" if is_fake_value else "real"
        attn_v_mode = "fake" if attn_v_is_fake_value else "real"
        bos_treatment = "enabled" if self.bos_special_treatment.item() else "disabled"
        
        # Get short name from layer for cleaner output
        layer_name = self.layer.split('.')[-1] if hasattr(self, 'layer') and self.layer else "unknown"
        
        # Sample a few values from the std tensors for debugging
        avg_std_sample = f"[{self.average_std_buffer[0].item():.4f}, {self.average_std_buffer[1].item():.4f}, ...]"
        bos_std_sample = f"[{self.bos_std_buffer[0].item():.4f}, {self.bos_std_buffer[1].item():.4f}, ...]"
        
        return (f"FakeLayerNorm(layer={layer_name}, mode={mode}, attn_v_mode={attn_v_mode}, "
                f"real_avg_std={self.real_average_std.item():.4f}, real_bos_std={self.real_bos_std.item():.4f}, "
                f"bos_treatment={bos_treatment}, "
                f"avg_std={avg_std_sample}, bos_std={bos_std_sample})")
    
    def forward(self, input, std_type="avg", attn_v=False):
        # We want all the enable / disable information to be in this class, but the class is re-used
        # for both the QK and V paths. Thus we add the attn_v flag to the call that is True only for
        # the V path. Thus we get to have flags `is_fake` and `attn_v_is_fake` to enable / disable the
        # LN for the QK and V paths separately.
        is_fake_value = self.attn_v_is_fake.item() if attn_v else self.is_fake.item()


        # Start of std calculation
        self.iteration += 1
        # if self.layer == "blocks.0.hook_resid_pre":
        #     print(f"Current_iteration: {self.iteration}")
        if self.iteration % (3*self.grad_acc_steps) == 0:
            self.global_step += 1
            avg_std = self.moving_std.get_mean()
            bos_std = self.moving_std_bos.get_mean()
            self.real_average_std.fill_(float(avg_std))
            self.real_bos_std.fill_(float(bos_std))
        # if self.layer == "blocks.0.hook_resid_pre":
        #     print(f"Current_global_step: {self.global_step.item()}")

        avg_std, bos_std = self.recompute_average_std(input)
        self.moving_std.append(avg_std)
        self.moving_std_bos.append(bos_std)

        if is_fake_value:
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
            std = self.average_std_buffer if std_type == "avg" else self.bos_std_buffer
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
            if os.environ.get("EXP_NON_BOS_AVERAGE_STD", "0") == "1":
                average_std = std[1:].mean().detach().item()
            else:
                average_std = std.mean().detach().item()
            bos_std = std[0].detach().item()
        return average_std, bos_std
    
    def sync_std(self):
        """Sync the average and bos std values (that are used in the forward pass) with the real std values."""
        with torch.no_grad():
            # Create new tensors instead of modifying in-place because we don't want to track gradients for these
            # This is conditional on the recomputation mode
            if self.bos_special_treatment.item():
                new_average_std = self.average_std_buffer.clone()
                new_average_std[1:] = torch.ones_like(new_average_std[1:]) * self.real_average_std.item()
                new_average_std[0] = self.real_bos_std.item()
                self.average_std_buffer = new_average_std.detach().requires_grad_(False)
            else:
                self.average_std_buffer = torch.ones_like(self.average_std_buffer) * self.real_average_std.item()
            
            # This will always enable BOS special treatment
            if self.bos_special_treatment.item():
                new_bos_std = torch.ones_like(self.bos_std_buffer) * self.real_bos_std.item()
            else:
                new_bos_std = torch.ones_like(self.bos_std_buffer) * self.real_average_std.item()
            self.bos_std_buffer = new_bos_std.detach().requires_grad_(False)
    
    def disable_eos_special_treatment(self):
        # Disable EOT special treatment, note that bos_std_buffer[0] will be set to average_std_buffer[1]
        # which is not always the same as bos_std_buffer[1]
        with torch.no_grad():
            # Create a new tensor with the same values as average_std_buffer to avoid sharing
            self.bos_std_buffer = self.average_std_buffer.clone().detach().requires_grad_(False)
    
    def disable_bos_special_treatment(self):
        # Disable BOS special treatment
        self.bos_special_treatment.fill_(False)
        with torch.no_grad():
            # Special treatment of position 0 is now disabled
            self.average_std_buffer[0] = self.average_std_buffer[1]
            # If EOS mask happens to apply to position 0, should also set bos_std_buffer[0] to average
            self.bos_std_buffer[0] = self.bos_std_buffer[1]


def replace_layernorm_with_fake_layernorm(model, std_dict, std_bos_dict, grad_acc_steps=1):
    n_layers = model.config.n_layer
    n_embd = model.transformer.h[0].ln_1.weight.shape[0]
    n_ctx = model.config.n_ctx

    # Replace ln_1 and ln_2 with FakeLayerNorm for each block
    for i in range(n_layers):
        block = model.transformer.h[i]
        
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
            init_bos_std=std_bos_dict[layer],
            grad_acc_steps=grad_acc_steps)

        layer = f"blocks.{i}.hook_resid_mid"
        block.ln_2 = FakeLayerNorm(
            n_embd=n_embd,
            n_ctx=n_ctx,
            layer=layer,
            bias=block.ln_2.bias is not None,
            init_average_std=std_dict[layer],
            init_bos_std=std_bos_dict[layer],
            grad_acc_steps=grad_acc_steps)
        
        # Restore weights
        block.ln_1.weight = nn.Parameter(ln_1_weight)
        if ln_1_bias is not None:
            block.ln_1.bias = nn.Parameter(ln_1_bias)
        block.ln_2.weight = nn.Parameter(ln_2_weight)
        if ln_2_bias is not None:
            block.ln_2.bias = nn.Parameter(ln_2_bias)
            
        # Monkey patch the attention forward
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
    
    layer = f"blocks.{n_layers-1}.hook_resid_post"
    model.transformer.ln_f = FakeLayerNorm(
        n_embd=n_embd,
        n_ctx=n_ctx,
        layer=layer,
        bias=ln_f.bias is not None,
        init_average_std=std_dict[layer],
        init_bos_std=std_bos_dict[layer],
        grad_acc_steps=grad_acc_steps
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

def load_model(model_name="gpt2", remove_ln=False, grad_acc_steps=1):
    model = transformers.GPT2LMHeadModel.from_pretrained(
        model_name,
        cache_dir=f"{model_name}_cache",
        config=transformers.GPT2Config.from_pretrained(model_name),
    )
    # attn_pdrop=0.1, embd_pdrop=0.1, resid_pdrop=0.1: Default values for GPT2, can be changed with kwargs
    
    if remove_ln:
        # Replace all LayerNorm instances with FakeLayerNorm
        std_dict = std_dicts[model_name]["std_dict"]
        std_bos_dict = std_dicts[model_name]["std_bos_dict"]

        replace_layernorm_with_fake_layernorm(model, std_dict, std_bos_dict, grad_acc_steps=grad_acc_steps)
    
    return model


def finetune(model, training_args, tokenized, data_collator, config, pile_eval_dataset=None, remove_ln=False, checkpoint_step=None):
    """Finetune model with or without layer normalization"""
    def disable_ln_2(block_index):
        model.transformer.h[block_index].ln_2.is_fake_prop = True
        print(f"disabled ln_2 for block {block_index}")

    def disable_ln_1qk(block_index):
        model.transformer.h[block_index].ln_1.is_fake_prop = True
        print(f"disabled ln_1 for block {block_index}")

    def disable_ln_1v(block_index):
        model.transformer.h[block_index].ln_1.attn_v_is_fake_prop = True
        print(f"disabled ln_1v for block {block_index}")

    def disable_ln_f():
        model.transformer.ln_f.is_fake_prop = True
        print("disabled ln_f")

    def disable_eot_std(block_index):
        model.transformer.h[block_index].ln_1.disable_eos_special_treatment()
        print(f"disabled eot std for block {block_index}")

    def disable_bos_std(block_index):
        model.transformer.h[block_index].ln_1.disable_bos_special_treatment()
        print(f"disabled bos std for block {block_index}")
    
    # Log the initial state of all FakeLayerNorm instances
    print("\n===== FakeLayerNorm Initial States =====")
    print(f"Environment settings:")
    print(f"  EXP_CORRECT_BOS: {os.environ.get('EXP_CORRECT_BOS', '0')}")
    print(f"  EXP_RECOMPUTE_STD_ON_FAKE: {os.environ.get('EXP_RECOMPUTE_STD_ON_FAKE', '0')}")
    print(f"  EXP_RECOMPUTE_STD_ON_REAL: {os.environ.get('EXP_RECOMPUTE_STD_ON_REAL', '0')}")
    
    for i, block in enumerate(model.transformer.h):
        print(f"\nBlock {i}:")
        print(f"  ln_1: {block.ln_1}")
        print(f"  ln_2: {block.ln_2}")
    
    print(f"\nFinal ln_f:")
    print(f"  {model.transformer.ln_f}")
    print("========================================\n")

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
                if step == self.start_step:
                    self.function()
                    self.log_event(step)
            elif self.layer_gap_steps == 0:
                if step == self.start_step:
                    [self.function(i) for i in range(self.n_layers)]
                    self.log_event(step)
            elif (step - self.start_step) % self.layer_gap_steps == 0:
                layer_index = (step - self.start_step) // self.layer_gap_steps
                if 0 <= layer_index < self.n_layers:
                    self.function(layer_index)
                    self.log_event(step, layer_index)
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
            """ Log to wandb: block number, mode, att_v_mode, average_std_buffer, bos_std_buffer, average_std_buffer[0], average_std_buffer[1], bos_std_buffer[0], bos_std_buffer[1] """
            if not _USE_WANDB:
                return control

            for i, block in enumerate(model.transformer.h):
                wandb.log({
                    f"block_{i}_ln_1_real_average_std": block.ln_1.real_average_std_prop,
                    f"block_{i}_ln_1_real_bos_std": block.ln_1.real_bos_std_prop,
                    f"block_{i}_ln_1_average_std_0": block.ln_1.average_std_buffer[0],
                    f"block_{i}_ln_1_average_std_1": block.ln_1.average_std_buffer[1],
                    f"block_{i}_ln_1_bos_std_0": block.ln_1.bos_std_buffer[0],
                    f"block_{i}_ln_1_bos_std_1": block.ln_1.bos_std_buffer[1],
                })

                wandb.log({
                    f"block_{i}_ln_2_real_average_std": block.ln_2.real_average_std_prop,
                    f"block_{i}_ln_2_real_bos_std": block.ln_2.real_bos_std_prop,
                    f"block_{i}_ln_2_average_std_0": block.ln_2.average_std_buffer[0],
                    f"block_{i}_ln_2_average_std_1": block.ln_2.average_std_buffer[1],
                    f"block_{i}_ln_2_bos_std_0": block.ln_2.bos_std_buffer[0],
                    f"block_{i}_ln_2_bos_std_1": block.ln_2.bos_std_buffer[1],
                })

            wandb.log({
                f"ln_f_real_average_std": model.transformer.ln_f.real_average_std_prop,
                f"ln_f_real_bos_std": model.transformer.ln_f.real_bos_std_prop,
                f"ln_f_average_std_0": model.transformer.ln_f.average_std_buffer[0],
                f"ln_f_average_std_1": model.transformer.ln_f.average_std_buffer[1],
                f"ln_f_bos_std_0": model.transformer.ln_f.bos_std_buffer[0],
                f"ln_f_bos_std_1": model.transformer.ln_f.bos_std_buffer[1],
            })

            return control 
    
    class CheckFakeLayerNormStateAfterLoading(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            """Print FakeLayerNorm states after checkpoint is loaded, just before training begins"""
            model = kwargs.get('model')
            if model is None:
                return control
            
            print("\n===== FakeLayerNorm States After Checkpoint Loading =====")
            print(f"Environment settings:")
            print(f"  EXP_CORRECT_BOS: {os.environ.get('EXP_CORRECT_BOS', '0')}")
            print(f"  EXP_RECOMPUTE_STD_ON_FAKE: {os.environ.get('EXP_RECOMPUTE_STD_ON_FAKE', '0')}")
            print(f"  EXP_RECOMPUTE_STD_ON_REAL: {os.environ.get('EXP_RECOMPUTE_STD_ON_REAL', '0')}")
            
            print("\nChecking if layers are fake after loading checkpoint:")
            for i, block in enumerate(model.transformer.h):
                print(f"Block {i}:")
                print(f"  ln_1.is_fake_prop: {block.ln_1.is_fake_prop}")
                print(f"  ln_1.attn_v_is_fake_prop: {block.ln_1.attn_v_is_fake_prop}")
                print(f"  ln_1.bos_special_treatment_prop: {block.ln_1.bos_special_treatment_prop}")
                print(f"  ln_2.is_fake_prop: {block.ln_2.is_fake_prop}")
                print(f"  ln_2.bos_special_treatment_prop: {block.ln_2.bos_special_treatment_prop}")
            
            print(f"\nFinal ln_f.is_fake_prop: {model.transformer.ln_f.is_fake_prop}")
            print(f"Final ln_f.bos_special_treatment_prop: {model.transformer.ln_f.bos_special_treatment_prop}")
            print("========================================\n")
            
            return control

    class SaveAtSpecificStepsCallback(TrainerCallback):
        """Callback to save checkpoints at specific steps specified by the user."""
        
        def __init__(self, save_steps=None):
            """
            Initialize the callback with the steps to checkpoint at.
            
            Args:
                save_steps: List of specific steps at which to save checkpoints
            """
            self.save_steps = save_steps or []
            
        def on_step_end(self, args, state, control, **kwargs):
            """Check if current step is in the list of steps to save checkpoints at."""
            if state.global_step in self.save_steps:
                # Use Trainer's existing checkpoint logic
                control.should_save = True
                
                # Force saving even if we just saved recently
                control.should_save_model = True
                
                print(f"Triggering checkpoint at step {state.global_step}")
                
                # Log to wandb if enabled
                if _USE_WANDB:
                    wandb.log({"custom_checkpoint": state.global_step})
                    
            return control

            
    class StopAfterNStepsCallback(TrainerCallback):
        def __init__(self, early_stop_step):
            self.early_stop_step = early_stop_step
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= self.early_stop_step:
                control.should_training_stop = True
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


    callbacks = [
        LogFakeLayerNormState(),
        CheckFakeLayerNormStateAfterLoading(),
        StopAfterNStepsCallback(config.early_stop_step),
    ]
    if remove_ln:
        callbacks.append(LNRemoverCallback(ln_removers))
    
    # Add custom checkpoint callback if checkpoint_step was provided
    if checkpoint_step:
        callbacks.append(SaveAtSpecificStepsCallback(save_steps=[checkpoint_step]))
        print(f"Will save checkpoint at step: {checkpoint_step}")
    
    # Create multi-dataset dictionary if pile_eval_dataset is provided
    if pile_eval_dataset is not None:
        eval_datasets = {
            # "openwebtext": tokenized["test"],
            "pile": pile_eval_dataset
        }
    else:
        eval_datasets = tokenized["test"]

    aux_loss_weight = config.aux_loss_weight
    
    if aux_loss_weight != 0:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=eval_datasets,
            data_collator=data_collator,
            callbacks=callbacks,
            aux_loss_weight=aux_loss_weight,
        )
    else:
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

def check_checkpoint_path(checkpoint_path):
    """
    Verify if the checkpoint path exists and which format is used.
    Returns a message about the checkpoint status.
    """
    if not os.path.exists(checkpoint_path):
        return f"Warning: Checkpoint path {checkpoint_path} does not exist"
    
    # Check for PyTorch or safetensors format
    pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    safetensors_model_path = os.path.join(checkpoint_path, "model.safetensors")
    
    if os.path.exists(pytorch_model_path):
        return f"Found PyTorch checkpoint at {pytorch_model_path}"
    elif os.path.exists(safetensors_model_path):
        return f"Found safetensors checkpoint at {safetensors_model_path}"
    else:
        return f"Warning: No model file found in {checkpoint_path} (checked for pytorch_model.bin and model.safetensors)"

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
        "--resume_from_checkpoint",
        required=False,
        help="Checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoint_step", 
        type=int,
        help="Step number at which to save a checkpoint",
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
        save_safetensors=False,  # Always use .bin format
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
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

    # Check if the checkpoint exists and print info
    if args.resume_from_checkpoint:
        checkpoint_message = check_checkpoint_path(args.resume_from_checkpoint)
        print(checkpoint_message)
    
    # Initialize model
    model = load_model(model_name, remove_ln=args.mode == "without_ln", grad_acc_steps=config.gradient_accumulation_steps)
    
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
        remove_ln=args.mode == "without_ln",
        checkpoint_step=args.checkpoint_step
    )

if __name__ == "__main__":
    main()
