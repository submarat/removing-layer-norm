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

from std_dicts import std_dicts
from pydantic import BaseModel, Field
from typing import Dict, Optional, Callable, Any

# TODO: support multi-GPU. If multiple GPUs are available, this will select the first one.
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1337)
# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

_USE_WANDB = True

# Default to gpt2 std_dicts
std_dict = std_dicts["gpt2"]["std_dict"]
std_bos_dict = std_dicts["gpt2"]["std_bos_dict"]

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
        
        # Add auxiliary loss to main loss
        loss = loss + aux_loss
        
        return (loss, outputs) if return_outputs else loss


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

def finetune_with_ln(model, training_args, tokenized, data_collator, config):
    """Finetune model with layer normalization
    
    Args:
        model: The model to finetune
        training_args: Training arguments
        tokenized: Tokenized dataset
        data_collator: Data collator for language modeling
        config: Training configuration
    """
    # Extract auxiliary loss weight from config if available
    aux_loss_weight = getattr(config, "aux_loss_weight", 0.1)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        aux_loss_weight=aux_loss_weight,
    )

    trainer.train()

def finetune_without_ln(model, training_args, tokenized, data_collator, config):

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

    # Extract auxiliary loss weight from config if available
    aux_loss_weight = getattr(config, "aux_loss_weight", 0.1)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        aux_loss_weight=aux_loss_weight,
        callbacks=[LNRemoverCallback(ln_removers)],
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
    
    tokenized, data_collator = prepare_dataset(model_name)
    model = load_model(model_name, remove_ln=args.mode == "without_ln")

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
