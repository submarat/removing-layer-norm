"""
Feature flags:

EXP_CORRECT_BOS: Use correct BOS special treatment
EXP_RECOMPUTE_STD_ON_FAKE: Recompute std when FakeLayerNorm is fake - will recompute std on every forward pass
EXP_RECOMPUTE_STD_ON_REAL: Recompute std when FakeLayerNorm is real - will freeze std when FakeLayerNorm goes fake
EXP_NON_BOS_AVERAGE_STD: Compute average std based on input[1:] (i.e. non-BOS) positions
EXP_BOS_SPECIAL_TREATMENT: Whether to use BOS special treatment where FakeLayerNorm will use a special std value for the first position in residual stream
EXP_FLASH_ATTN: Use flash attention
"""
import os

import argparse
import datasets
import gc
import numpy as np
import torch
import torch.nn as nn
import tqdm
import transformers
import wandb
from datetime import datetime
from config import FINETUNE_CONFIGS
from jaxtyping import Float
from prepare_dataset import prepare_dataset
from torch import Tensor
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
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

_USE_WANDB = os.environ.get("USE_WANDB", "1") == "1"

_QWEN_MODEL_TYPES = {"qwen", "qwen2", "qwen3"}


def _get_model_type(model):
    return getattr(getattr(model, "config", None), "model_type", None)


def _is_qwen_model(model):
    return _get_model_type(model) in _QWEN_MODEL_TYPES


def _get_blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Unsupported model architecture for block extraction")


def _get_final_norm_module(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    return None


def _get_context_length_from_config(config):
    for attr in ("n_ctx", "max_position_embeddings", "max_sequence_length"):
        value = getattr(config, attr, None)
        if value is not None:
            return value
    return 1024


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
        if self.aux_loss_weight > 0:
            # Register forward hook to capture activations before ln_f
            def pre_ln_f_hook(module, input):
                # Store the input to ln_f (which is what we want to normalize)
                self.pre_ln_f_activations = input[0]
                return input
            
            # Register the hook on the final layer norm
            final_norm = _get_final_norm_module(self.model)
            if final_norm is not None:
                final_norm.register_forward_pre_hook(pre_ln_f_hook)
        
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
        
        eos_token_id = getattr(model.config, "eos_token_id", 50256)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_tensor = torch.tensor(eos_token_id, device=input_ids.device)
            eos_positions = torch.isin(input_ids, eos_token_tensor)
        else:
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
        # torch.cuda.empty_cache()  # Explicitly clear cache if using CUDA
        # gc.collect() # Collect garbage to free up memory sometimes this does magic
        
        # Add auxiliary loss to main loss
        loss = loss + aux_loss
        
        return (loss, outputs) if return_outputs else loss

class FakeLayerNorm(nn.Module):
    """RMSNorm using a fixed std instead of the actual standard deviation."""

    def __init__(
        self,
        n_embd,
        n_ctx,
        layer,
        bias,
        init_average_std,
        init_bos_std,
        grad_acc_steps=1,
        momentum=0.1,
        bos_special_treatment=False,
        eps=1e-6,
        position_dependent=False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
        self.eps = eps
        self.position_dependent = position_dependent
        
        # Store layer information
        self.layer = layer
        self.n_embd = n_embd
        self.n_ctx = n_ctx
        
        # Register all flags as buffers so they're automatically included in state_dict
        # Using naming without underscore to be compatible with old checkpoints
        self.register_buffer("is_fake", torch.tensor(False))
        self.register_buffer("bos_special_treatment", torch.tensor(bos_special_treatment))
        self.register_buffer("real_average_std", torch.tensor(float(init_average_std)))
        self.register_buffer("real_bos_std", torch.tensor(float(init_bos_std)))
        self.register_buffer("grad_acc_steps", torch.tensor(grad_acc_steps))
        self.register_buffer("momentum", torch.tensor(float(momentum)))

        if self.position_dependent and os.environ.get("EXP_CORRECT_BOS", "0") == "1":
            std_dim = n_ctx
        else:
            std_dim = n_embd
            
        # Register non-parameter tensors as buffers so they're saved in state_dict
        self.register_buffer("average_std_buffer", torch.ones(std_dim, device=device) * init_average_std)
        
        if self.bos_special_treatment.item():
            # Special handling for position 0
            self.average_std_buffer[0] = init_bos_std
            self.register_buffer("bos_std_buffer", torch.ones(std_dim, device=device) * init_bos_std)        
        else:
            self.register_buffer("bos_std_buffer", torch.ones(std_dim, device=device) * init_average_std)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Call parent method to load most of the state
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
        # Remove successfully loaded buffer keys from missing_keys
        for key in [prefix + "is_fake", prefix + "bos_special_treatment"]:
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
        mode = "fake" if is_fake_value else "real"
        bos_treatment = "enabled" if self.bos_special_treatment.item() else "disabled"
        
        # Get short name from layer for cleaner output
        layer_name = self.layer.split('.')[-1] if hasattr(self, 'layer') and self.layer else "unknown"
        
        # Sample a few values from the std tensors for debugging
        avg_std_sample = f"[{self.average_std_buffer[0].item():.4f}, {self.average_std_buffer[1].item():.4f}, ...]"
        bos_std_sample = f"[{self.bos_std_buffer[0].item():.4f}, {self.bos_std_buffer[1].item():.4f}, ...]"
        
        return (f"FakeLayerNorm(layer={layer_name}, mode={mode}, "
                f"real_avg_std={self.real_average_std.item():.4f}, real_bos_std={self.real_bos_std.item():.4f}, "
                f"bos_treatment={bos_treatment}, "
                f"avg_std={avg_std_sample}, bos_std={bos_std_sample})")
    
    def _reshape_std_for_input(self, std, input, use_position_layout):
        view_shape = [1] * input.dim()
        if use_position_layout and input.dim() >= 2:
            view_shape[1] = std.shape[0]
        else:
            view_shape[-1] = std.shape[0]
        return std.view(*view_shape)
    
    def _compute_feature_std(self, x):
        return (x.pow(2).mean(dim=-1) + self.eps).sqrt()
    
    def forward(self, input, std_type="avg"):
        is_fake_value = self.is_fake.item()
        self.recompute_average_std(input)

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
            use_position_layout = (
                self.position_dependent and os.environ.get("EXP_CORRECT_BOS", "0") == "1" and std.numel() == self.n_ctx
            )
            broadcast_std = self._reshape_std_for_input(std.to(input.dtype), input, use_position_layout)
            normalized = input / broadcast_std
            weight = self.weight
            bias = self.bias
            if weight.dtype != input.dtype:
                weight = weight.to(input.dtype)
            normalized = normalized * weight
            if bias is not None:
                if bias.dtype != input.dtype:
                    bias = bias.to(input.dtype)
                normalized = normalized + bias
            return normalized
        else:
            if os.environ.get("EXP_RECOMPUTE_STD_ON_REAL", "0") == "1":
                with torch.no_grad():
                    self.sync_std()

            variance = input.pow(2).mean(dim=-1, keepdim=True)
            normalized = input * torch.rsqrt(variance + self.eps)
            weight = self.weight
            bias = self.bias
            if weight.dtype != input.dtype:
                weight = weight.to(input.dtype)
            normalized = normalized * weight
            if bias is not None:
                if bias.dtype != input.dtype:
                    bias = bias.to(input.dtype)
                normalized = normalized + bias
            return normalized
    
    @torch.no_grad()
    def recompute_average_std(self, x: Float[Tensor, "batch posn d_model"]):
        if self.bos_special_treatment.item():
            # Taking std over model dim
            std = self._compute_feature_std(x)
            # averaging over the batch dimentaion
            std = std.mean(dim=0) 
            # averaging over sequence postions
            if os.environ.get("EXP_NON_BOS_AVERAGE_STD", "0") == "1":
                average_std = std[1:].mean().detach().item()
            else:
                average_std = std.mean().detach().item()
            bos_std = std[0].detach().item()

            self.real_average_std_prop = self.momentum * self.real_average_std_prop + (1-self.momentum) * average_std
            self.real_bos_std_prop = self.momentum * self.real_bos_std_prop + (1-self.momentum) * bos_std
        else:
            # Taking std over model dim
            std = self._compute_feature_std(x)
            # Averaging over everything else
            mean_std = std.mean().detach().item()
            self.real_average_std_prop = self.momentum * self.real_average_std_prop +  (1-self.momentum) * mean_std
            self.real_bos_std_prop = self.momentum * self.real_bos_std_prop +  (1-self.momentum) * mean_std

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


def _std_value(std_dict, key, default=1.0):
    if std_dict:
        return std_dict.get(key, default)
    return default


def _std_pair(std_dict, std_bos_dict, key, default=1.0):
    avg = _std_value(std_dict, key, default)
    bos = _std_value(std_bos_dict, key, avg)
    return avg, bos


def replace_layernorm_with_fake_layernorm(model, std_dict, std_bos_dict, grad_acc_steps=1, momentum=0.1):
    if not _is_qwen_model(model):
        raise ValueError("FakeLayerNorm currently supports Qwen architectures only.")
    _replace_layernorm_with_fake_layernorm_qwen(
        model, std_dict or {}, std_bos_dict or {}, grad_acc_steps=grad_acc_steps, momentum=momentum
    )


def _replace_layernorm_with_fake_layernorm_qwen(model, std_dict, std_bos_dict, grad_acc_steps=1, momentum=0.1):
    layers = model.model.layers
    n_ctx = _get_context_length_from_config(model.config)
    rms_eps = getattr(model.config, "rms_norm_eps", 1e-6)
    registry = {
        "pre_attn": [],
        "post_attn": [],
        "q_norm": [],
        "k_norm": [],
        "final": [],
    }

    for i, block in enumerate(layers):
        input_weight = block.input_layernorm.weight.clone().detach()
        layer_name = f"blocks.{i}.hook_input_layernorm"
        avg_std, bos_std = _std_pair(std_dict, std_bos_dict, layer_name)
        block.input_layernorm = FakeLayerNorm(
            n_embd=block.input_layernorm.weight.shape[0],
            n_ctx=n_ctx,
            layer=layer_name,
            bias=False,
            init_average_std=avg_std,
            init_bos_std=bos_std,
            grad_acc_steps=grad_acc_steps,
            momentum=momentum,
            eps=rms_eps,
            position_dependent=True,
        )
        block.input_layernorm.weight = nn.Parameter(input_weight)
        registry["pre_attn"].append(block.input_layernorm)

        post_weight = block.post_attention_layernorm.weight.clone().detach()
        layer_name = f"blocks.{i}.hook_post_attention_layernorm"
        avg_std, bos_std = _std_pair(std_dict, std_bos_dict, layer_name)
        block.post_attention_layernorm = FakeLayerNorm(
            n_embd=block.post_attention_layernorm.weight.shape[0],
            n_ctx=n_ctx,
            layer=layer_name,
            bias=False,
            init_average_std=avg_std,
            init_bos_std=bos_std,
            grad_acc_steps=grad_acc_steps,
            momentum=momentum,
            eps=rms_eps,
            position_dependent=True,
        )
        block.post_attention_layernorm.weight = nn.Parameter(post_weight)
        registry["post_attn"].append(block.post_attention_layernorm)

        if hasattr(block.self_attn, "q_norm"):
            q_weight = block.self_attn.q_norm.weight.clone().detach()
            layer_name = f"blocks.{i}.hook_q_norm"
            avg_std, bos_std = _std_pair(std_dict, std_bos_dict, layer_name)
            block.self_attn.q_norm = FakeLayerNorm(
                n_embd=block.self_attn.q_norm.weight.shape[0],
                n_ctx=n_ctx,
                layer=layer_name,
                bias=False,
                init_average_std=avg_std,
                init_bos_std=bos_std,
                grad_acc_steps=grad_acc_steps,
                momentum=momentum,
                eps=rms_eps,
                position_dependent=False,
            )
            block.self_attn.q_norm.weight = nn.Parameter(q_weight)
            registry["q_norm"].append(block.self_attn.q_norm)

        if hasattr(block.self_attn, "k_norm"):
            k_weight = block.self_attn.k_norm.weight.clone().detach()
            layer_name = f"blocks.{i}.hook_k_norm"
            avg_std, bos_std = _std_pair(std_dict, std_bos_dict, layer_name)
            block.self_attn.k_norm = FakeLayerNorm(
                n_embd=block.self_attn.k_norm.weight.shape[0],
                n_ctx=n_ctx,
                layer=layer_name,
                bias=False,
                init_average_std=avg_std,
                init_bos_std=bos_std,
                grad_acc_steps=grad_acc_steps,
                momentum=momentum,
                eps=rms_eps,
                position_dependent=False,
            )
            block.self_attn.k_norm.weight = nn.Parameter(k_weight)
            registry["k_norm"].append(block.self_attn.k_norm)

    final_weight = model.model.norm.weight.clone().detach()
    layer_name = "final_norm"
    avg_std, bos_std = _std_pair(std_dict, std_bos_dict, layer_name)
    model.model.norm = FakeLayerNorm(
        n_embd=model.model.norm.weight.shape[0],
        n_ctx=n_ctx,
        layer=layer_name,
        bias=False,
        init_average_std=avg_std,
        init_bos_std=bos_std,
        grad_acc_steps=grad_acc_steps,
        momentum=momentum,
        eps=rms_eps,
        position_dependent=True,
    )
    model.model.norm.weight = nn.Parameter(final_weight)
    registry["final"].append(model.model.norm)
    model.fake_ln_registry = registry

def load_model(model_name="gpt2", remove_ln=False, grad_acc_steps=1, momentum=0.1):
    trust_remote_code = "qwen" in model_name.lower()
    cache_dir = f"{model_name.replace('/', '_')}_cache"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    attn_impl = 'flash_attention_2' if os.environ.get("EXP_FLASH_ATTN", "0") == "1" else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    from_pretrained_kwargs = {
        "cache_dir": cache_dir,
        "config": config,
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_impl and config.model_type == "gpt2":
        from_pretrained_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **from_pretrained_kwargs,
    )
    
    if remove_ln and not _is_qwen_model(model):
        raise ValueError("LayerNorm removal is currently only supported for Qwen models.")

    if remove_ln:
        std_data = std_dicts.get(model_name, {})
        std_dict = std_data.get("std_dict")
        std_bos_dict = std_data.get("std_bos_dict")
        replace_layernorm_with_fake_layernorm(
            model,
            std_dict,
            std_bos_dict,
            grad_acc_steps=grad_acc_steps,
            momentum=momentum,
        )
    
    return model


def finetune(model, training_args, tokenized, data_collator, config, pile_eval_dataset=None, remove_ln=False, checkpoint_step=None):
    """Finetune model with or without layer normalization"""
    blocks = _get_blocks(model)
    num_layers = len(blocks)
    fake_registry = getattr(model, "fake_ln_registry", {})
    pre_attn_norms = fake_registry.get("pre_attn", [])
    post_attn_norms = fake_registry.get("post_attn", [])
    q_norms = fake_registry.get("q_norm", [])
    k_norms = fake_registry.get("k_norm", [])
    final_norms = fake_registry.get("final", [])
    
    def _set_fake(norms, index):
        if 0 <= index < len(norms):
            norms[index].is_fake_prop = True
    
    def disable_ln_2(block_index):
        _set_fake(post_attn_norms, block_index)
        print(f"disabled post-attention rmsnorm for block {block_index}")

    def disable_ln_1qk(block_index):
        _set_fake(q_norms, block_index)
        _set_fake(k_norms, block_index)
        print(f"disabled q/k rmsnorm for block {block_index}")

    def disable_ln_1v(block_index):
        _set_fake(pre_attn_norms, block_index)
        print(f"disabled input rmsnorm for block {block_index}")

    def disable_ln_f():
        if final_norms:
            final_norms[0].is_fake_prop = True
            print("disabled final rmsnorm")

    def disable_eot_std(block_index=None):
        indices = range(len(pre_attn_norms)) if block_index is None else [block_index]
        for idx in indices:
            if 0 <= idx < len(pre_attn_norms):
                pre_attn_norms[idx].disable_eos_special_treatment()
                print(f"disabled eot std for block {idx}")

    def disable_bos_std(block_index=None):
        indices = range(len(pre_attn_norms)) if block_index is None else [block_index]
        for idx in indices:
            if 0 <= idx < len(pre_attn_norms):
                pre_attn_norms[idx].disable_bos_special_treatment()
                print(f"disabled bos std for block {idx}")
    
    # Log the initial state of all FakeLayerNorm instances
    print("\n===== FakeLayerNorm Initial States =====")
    print(f"Environment settings:")
    print(f"  EXP_CORRECT_BOS: {os.environ.get('EXP_CORRECT_BOS', '0')}")
    print(f"  EXP_RECOMPUTE_STD_ON_FAKE: {os.environ.get('EXP_RECOMPUTE_STD_ON_FAKE', '0')}")
    print(f"  EXP_RECOMPUTE_STD_ON_REAL: {os.environ.get('EXP_RECOMPUTE_STD_ON_REAL', '0')}")
    print(f"  EXP_BOS_SPECIAL_TREATMENT: {os.environ.get('EXP_BOS_SPECIAL_TREATMENT', '0')}")

    
    for i, block in enumerate(blocks):
        print(f"\nBlock {i}:")
        if hasattr(block, "input_layernorm"):
            print(f"  input_layernorm: {block.input_layernorm}")
        if hasattr(block, "post_attention_layernorm"):
            print(f"  post_attention_layernorm: {block.post_attention_layernorm}")
        attn = getattr(block, "self_attn", None)
        if attn is not None and hasattr(attn, "q_norm"):
            print(f"  q_norm: {attn.q_norm}")
        if attn is not None and hasattr(attn, "k_norm"):
            print(f"  k_norm: {attn.k_norm}")
    
    final_norm = _get_final_norm_module(model)
    if final_norm is not None:
        print(f"\nFinal norm:")
        print(f"  {final_norm}")
    print("========================================\n")

    class LNRemover:
        """
        Schedules the "removal" of LayerNorms by calling the disable function.
        """

        def __init__(self, start_step, layer_gap_steps, function):
            self.n_layers = num_layers  # Get layers dynamically
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
            blocks = _get_blocks(model)
            if not blocks:
                return control

            def _log_norm(prefix, module):
                if not isinstance(module, FakeLayerNorm):
                    return
                avg0 = module.average_std_buffer[0].item()
                avg1 = module.average_std_buffer[1].item() if module.average_std_buffer.numel() > 1 else avg0
                bos0 = module.bos_std_buffer[0].item()
                bos1 = module.bos_std_buffer[1].item() if module.bos_std_buffer.numel() > 1 else bos0
                wandb.log({
                    f"{prefix}_real_average_std": module.real_average_std_prop,
                    f"{prefix}_real_bos_std": module.real_bos_std_prop,
                    f"{prefix}_average_std_0": avg0,
                    f"{prefix}_average_std_1": avg1,
                    f"{prefix}_bos_std_0": bos0,
                    f"{prefix}_bos_std_1": bos1,
                })

            for i, block in enumerate(blocks):
                if hasattr(block, "input_layernorm"):
                    _log_norm(f"block_{i}_input_ln", block.input_layernorm)
                if hasattr(block, "post_attention_layernorm"):
                    _log_norm(f"block_{i}_post_ln", block.post_attention_layernorm)
                attn = getattr(block, "self_attn", None)
                if attn is not None and hasattr(attn, "q_norm"):
                    _log_norm(f"block_{i}_q_norm", attn.q_norm)
                if attn is not None and hasattr(attn, "k_norm"):
                    _log_norm(f"block_{i}_k_norm", attn.k_norm)

            final_norm = _get_final_norm_module(model)
            if final_norm is not None:
                _log_norm("final_norm", final_norm)

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
            print(f"  EXP_BOS_SPECIAL_TREATMENT: {os.environ.get('EXP_BOS_SPECIAL_TREATMENT', '0')}")

            print("\nChecking if layers are fake after loading checkpoint:")
            blocks = _get_blocks(model)
            if blocks and isinstance(getattr(blocks[0], "input_layernorm", None), FakeLayerNorm):
                for i, block in enumerate(blocks):
                    print(f"Block {i}:")
                    if hasattr(block, "input_layernorm") and isinstance(block.input_layernorm, FakeLayerNorm):
                        print(f"  input_layernorm.is_fake_prop: {block.input_layernorm.is_fake_prop}")
                        print(f"  input_layernorm.bos_special_treatment_prop: {block.input_layernorm.bos_special_treatment_prop}")
                    attn = getattr(block, "self_attn", None)
                    if attn is not None and hasattr(attn, "q_norm") and isinstance(attn.q_norm, FakeLayerNorm):
                        print(f"  q_norm.is_fake_prop: {attn.q_norm.is_fake_prop}")
                    if attn is not None and hasattr(attn, "k_norm") and isinstance(attn.k_norm, FakeLayerNorm):
                        print(f"  k_norm.is_fake_prop: {attn.k_norm.is_fake_prop}")
                    if hasattr(block, "post_attention_layernorm") and isinstance(block.post_attention_layernorm, FakeLayerNorm):
                        print(f"  post_attention_layernorm.is_fake_prop: {block.post_attention_layernorm.is_fake_prop}")
                        print(f"  post_attention_layernorm.bos_special_treatment_prop: {block.post_attention_layernorm.bos_special_treatment_prop}")
                
                final_norm = _get_final_norm_module(model)
                if isinstance(final_norm, FakeLayerNorm):
                    print(f"\nFinal norm.is_fake_prop: {final_norm.is_fake_prop}")
                    print(f"Final norm.bos_special_treatment_prop: {final_norm.bos_special_treatment_prop}")
            else:
                print("Not removing layernorm!")
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
    ]

    if os.environ.get("EXP_CORRECT_BOS", "0") == "1":
        ln_removers.append(LNRemover(training_config.start_bos, training_config.gap_bos, disable_bos_std))

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
        default="qwen3_debug",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override per-device train batch size defined in the config",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        help="Override gradient accumulation steps defined in the config",
    )
    args = parser.parse_args()

    # Forcing to use only 1 GPU. Otherwise, tensors end up on different devices.
    # Fixing this is an open TODO but not a priority.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected ({torch.cuda.device_count()}). Forcing single GPU usage.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Get model name from config
    config = FINETUNE_CONFIGS[args.config]

    config_updates = {}
    if args.batch_size is not None:
        config_updates["base_batch_size"] = args.batch_size
        config_updates["batch_size"] = args.batch_size
        if args.grad_accum_steps is None:
            desired_batch_size = config.target_batch_tokens / config.block_size
            config_updates["gradient_accumulation_steps"] = max(
                1, int(desired_batch_size // max(1, args.batch_size))
            )
    if args.grad_accum_steps is not None:
        config_updates["gradient_accumulation_steps"] = max(1, args.grad_accum_steps)

    if config_updates:
        config = config.model_copy(update=config_updates)

    model_name = config.model_name

    # Prepare datasets
    tokenized, data_collator = prepare_dataset(model_name, ctx_len=config.block_size)
    
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
    model = load_model(model_name, remove_ln=args.mode == "without_ln", grad_acc_steps=config.gradient_accumulation_steps, momentum=config.momentum)
    
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
