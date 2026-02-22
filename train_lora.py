"""
Training script for LayerNorm removal using full-rank LoRA adapters.

The pretrained GPT-2 weights are frozen. Only LoRA adapter matrices on the
Conv1D layers are trained.  The same FakeLayerNorm removal schedule from the
original codebase is reused.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import wandb
import types

from datetime import datetime
from config import FINETUNE_CONFIGS
from prepare_dataset import prepare_dataset
from pile_eval import preprocess_pile_dataset, convert_for_trainer
from lora import inject_lora_adapters, LoRAConv1D
from train import FakeLayerNorm, _USE_WANDB
from std_dicts import std_dicts
from devtools import pprint
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1337)


def replace_layernorm_with_fake_layernorm_lora(model, std_dict, std_bos_dict, grad_acc_steps=1, momentum=0.1):
    """Same as original but the attention monkey-patch calls self.c_attn(x)
    instead of manually accessing .weight, so it works with LoRAConv1D wrappers."""
    n_layers = model.config.n_layer
    n_embd = model.transformer.h[0].ln_1.weight.shape[0]
    n_ctx = model.config.n_ctx

    for i in range(n_layers):
        block = model.transformer.h[i]

        ln_1_weight = block.ln_1.weight.clone().detach()
        ln_1_bias = block.ln_1.bias.clone().detach() if block.ln_1.bias is not None else None
        ln_2_weight = block.ln_2.weight.clone().detach()
        ln_2_bias = block.ln_2.bias.clone().detach() if block.ln_2.bias is not None else None

        layer = f"blocks.{i}.hook_resid_pre"
        block.ln_1 = FakeLayerNorm(
            n_embd=n_embd, n_ctx=n_ctx, layer=layer,
            bias=block.ln_1.bias is not None,
            init_average_std=std_dict[layer],
            init_bos_std=std_bos_dict[layer],
            grad_acc_steps=grad_acc_steps, momentum=momentum)

        layer = f"blocks.{i}.hook_resid_mid"
        block.ln_2 = FakeLayerNorm(
            n_embd=n_embd, n_ctx=n_ctx, layer=layer,
            bias=block.ln_2.bias is not None,
            init_average_std=std_dict[layer],
            init_bos_std=std_bos_dict[layer],
            grad_acc_steps=grad_acc_steps, momentum=momentum)

        block.ln_1.weight = nn.Parameter(ln_1_weight)
        if ln_1_bias is not None:
            block.ln_1.bias = nn.Parameter(ln_1_bias)
        block.ln_2.weight = nn.Parameter(ln_2_weight)
        if ln_2_bias is not None:
            block.ln_2.bias = nn.Parameter(ln_2_bias)

        # Monkey-patch attention forward — calls c_attn(x) so it goes through
        # LoRAConv1D.forward when adapters are injected.
        def make_attn_forward(old_forward):
            def new_forward(self, x_qk, x_v):
                B, T, C = x_qk.size()

                # Full c_attn forward for QK path and V path separately.
                # c_attn produces (Q, K, V) concatenated on last dim.
                qkv_qk = self.c_attn(x_qk)  # (B, T, 3*C)
                qkv_v = self.c_attn(x_v)     # (B, T, 3*C)

                q, k, _ = qkv_qk.split(C, dim=2)
                _, _, v = qkv_v.split(C, dim=2)

                q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
                k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
                v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_dropout.p,
                    is_causal=True
                )
                y = y.transpose(1, 2).contiguous().view(B, T, C)

                y = self.c_proj(y)
                y = self.resid_dropout(y)
                return y

            return types.MethodType(new_forward, block.attn)

        block.attn.forward = make_attn_forward(block.attn.forward)

        def make_forward(old_forward):
            def new_forward(self, x, *args, **kwargs):
                eot_mask = kwargs.pop('eot_mask', None)

                x_qk = self.ln_1(x)
                x_v = self.ln_1(x, attn_v=True)

                if eot_mask is not None:
                    x_v_eot = self.ln_1(x, std_type='bos', attn_v=True)
                    x_v[eot_mask] = x_v_eot[eot_mask]
                    del x_v_eot

                attn_output = self.attn(x_qk, x_v)
                x = x + attn_output
                x = x + self.mlp(self.ln_2(x))
                return x
            return types.MethodType(new_forward, block)

        block.forward = make_forward(block.forward)

    # Replace ln_f
    ln_f = model.transformer.ln_f
    ln_f_weight = ln_f.weight.clone().detach()
    ln_f_bias = ln_f.bias.clone().detach() if ln_f.bias is not None else None

    layer = f"blocks.{n_layers-1}.hook_resid_post"
    model.transformer.ln_f = FakeLayerNorm(
        n_embd=n_embd, n_ctx=n_ctx, layer=layer,
        bias=ln_f.bias is not None,
        init_average_std=std_dict[layer],
        init_bos_std=std_bos_dict[layer],
        grad_acc_steps=grad_acc_steps, momentum=momentum)
    model.transformer.ln_f.weight = nn.Parameter(ln_f_weight)
    if ln_f_bias is not None:
        model.transformer.ln_f.bias = nn.Parameter(ln_f_bias)

    # Monkey-patch transformer forward
    def make_transformer_forward(old_forward):
        def new_forward(self, *args, **kwargs):
            input_ids = kwargs.get('input_ids', args[0] if args else None)

            eot_mask = None
            if input_ids is not None:
                eot_mask = input_ids == 50256

            if args and isinstance(args[0], torch.Tensor):
                kwargs['input_ids'] = args[0]
                args = args[1:]

            hidden_states = self.wte(kwargs['input_ids'])
            position_ids = torch.arange(0, hidden_states.size(1), dtype=torch.long, device=hidden_states.device)
            hidden_states = hidden_states + self.wpe(position_ids)
            hidden_states = self.drop(hidden_states)

            for block in self.h:
                hidden_states = block(hidden_states, eot_mask=eot_mask)

            hidden_states = self.ln_f(hidden_states)
            hidden_states = hidden_states.to(self.wte.weight.dtype)

            return transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=None, hidden_states=None,
                attentions=None, cross_attentions=None,
            )
        return types.MethodType(new_forward, model.transformer)

    model.transformer.forward = make_transformer_forward(model.transformer.forward)


def load_model_lora(model_name="gpt2", grad_acc_steps=1, momentum=0.1):
    """Load GPT-2, inject LoRA adapters, then replace LN with FakeLayerNorm."""
    model = transformers.GPT2LMHeadModel.from_pretrained(
        model_name,
        cache_dir=f"{model_name}_cache",
        config=transformers.GPT2Config.from_pretrained(model_name),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 1. Inject LoRA adapters on all Conv1D layers (while they are still Conv1D)
    trainable, frozen = inject_lora_adapters(model)

    # 2. Replace LayerNorm with FakeLayerNorm (the monkey-patch now calls
    #    through LoRAConv1D.forward)
    std_dict = std_dicts[model_name]["std_dict"]
    std_bos_dict = std_dicts[model_name]["std_bos_dict"]
    replace_layernorm_with_fake_layernorm_lora(
        model, std_dict, std_bos_dict,
        grad_acc_steps=grad_acc_steps, momentum=momentum)

    # 3. Make sure FakeLayerNorm params are frozen too
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"After FakeLayerNorm replacement:")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {frozen:,}")

    return model, trainable, frozen


def finetune_lora(model, training_args, tokenized, data_collator, config,
                  pile_eval_dataset=None, checkpoint_step=None,
                  trainable_params=0, frozen_params=0):
    """Finetune with LoRA adapters and progressive LN removal."""

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

    print("\n===== FakeLayerNorm Initial States =====")
    for i, block in enumerate(model.transformer.h):
        print(f"Block {i}: ln_1={block.ln_1}  ln_2={block.ln_2}")
    print(f"ln_f: {model.transformer.ln_f}")
    print("========================================\n")

    class LNRemover:
        def __init__(self, start_step, layer_gap_steps, function):
            self.n_layers = len(model.transformer.h)
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

        def log_event(self, step, layer_index=None):
            if _USE_WANDB:
                event_name = f"{self.function.__name__}"
                if layer_index is not None:
                    event_name += f"_layer_{layer_index}"
                wandb.log({event_name: float('nan')})

        def log(self, wb):
            if _USE_WANDB:
                wb.log({
                    f"{self.function.__name__}.start_step": self.start_step,
                    f"{self.function.__name__}.layer_gap_steps": self.layer_gap_steps,
                })

    class LNRemoverCallback(TrainerCallback):
        def __init__(self, ln_removers):
            self.ln_removers = ln_removers

        def on_step_begin(self, args, state, control, **kwargs):
            for ln_remover in self.ln_removers:
                ln_remover(state.global_step)
                ln_remover.log(wandb)
            return control

    class StopAfterNStepsCallback(TrainerCallback):
        def __init__(self, early_stop_step):
            self.early_stop_step = early_stop_step

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= self.early_stop_step:
                control.should_training_stop = True
                return control

    class SaveAtSpecificStepsCallback(TrainerCallback):
        def __init__(self, save_steps=None):
            self.save_steps = save_steps or []

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.save_steps:
                control.should_save = True
                control.should_save_model = True
                if _USE_WANDB:
                    wandb.log({"custom_checkpoint": state.global_step})
            return control

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
        LNRemoverCallback(ln_removers),
        StopAfterNStepsCallback(config.early_stop_step),
    ]
    if checkpoint_step:
        callbacks.append(SaveAtSpecificStepsCallback(save_steps=[checkpoint_step]))

    if pile_eval_dataset is not None:
        eval_datasets = {"pile": pile_eval_dataset}
    else:
        eval_datasets = tokenized["test"]

    class LogLoRAParamsCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if _USE_WANDB:
                wandb.log({
                    "lora_trainable_params": trainable_params,
                    "lora_frozen_params": frozen_params,
                })
            return control

    callbacks.append(LogLoRAParamsCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="LoRA-based LayerNorm removal")
    parser.add_argument("--config", choices=list(FINETUNE_CONFIGS.keys()),
                        required=True, help="Training configuration to use")
    parser.add_argument("--resume_from_checkpoint", required=False)
    parser.add_argument("--checkpoint_step", type=int, required=False)
    args = parser.parse_args()

    config = FINETUNE_CONFIGS[args.config]
    model_name = config.model_name

    tokenized, data_collator = prepare_dataset(model_name)

    print("Preparing Pile-apollo evaluation dataset...")
    pile_eval_dataset = None
    if os.environ.get("EVAL", "0") == "1":
        processed_examples, pile_tokenizer = preprocess_pile_dataset(
            "pile-apollo", model_name, num_samples=config.num_eval_samples)
        pile_eval_dataset = convert_for_trainer(
            processed_examples, pile_tokenizer,
            model_name=model_name, num_samples=config.num_eval_samples)

    output_dir = f"results/{model_name}_lora/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
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
        gradient_checkpointing=config.gradient_checkpointing,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        max_grad_norm=getattr(config, 'max_grad_norm', 1.0),
        logging_dir="./logs",
        prediction_loss_only=True,
        lr_scheduler_type=config.lr_scheduler_type,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        report_to="wandb" if _USE_WANDB else "none",
        run_name=f"{args.config}-lora",
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

    model, trainable_params, frozen_params = load_model_lora(
        model_name,
        grad_acc_steps=config.gradient_accumulation_steps,
        momentum=config.momentum)

    print("Begin training")
    print(model)
    print(training_args)
    pprint(config)

    finetune_lora(
        model, training_args, tokenized, data_collator, config,
        pile_eval_dataset,
        checkpoint_step=args.checkpoint_step,
        trainable_params=trainable_params,
        frozen_params=frozen_params)


if __name__ == "__main__":
    main()
