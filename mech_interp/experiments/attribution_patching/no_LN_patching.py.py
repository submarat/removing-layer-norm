#%%
import sys
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import load_nln_model, load_finetuned_model
import torch
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict
import numpy as np
from transformers import AutoTokenizer
import einops
import matplotlib.pyplot as plt
from neel_plotly import scatter, imshow
tokenizer = AutoTokenizer.from_pretrained("gpt2")


model = load_nln_model()
model.set_use_attn_result(True)
model.set_use_attn_in(True)
model.set_use_hook_mlp_in(True)
#%%

prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]
#%%
clean_tokens = model.to_tokens(prompts)

# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor(
    [
        [model.to_single_token(answers[i][j]) for j in range(2)]
        for i in range(len(answers))
    ],
    device=model.cfg.device,
)
print("Answer token indices", answer_token_indices)
# %% #GET THE METRIC

def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff

# normalize the logit diffs
def ioi_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )
        

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")# %%

# %%
### OK NOW WE CAN DO THE ATTRIBUTION PATCHING 
from transformer_lens import (
    # HookedTransformer,
    # HookedTransformerConfig,
    # FactoredMatrix,
    ActivationCache,
)
filter_not_qkv_input = lambda name: "_input" not in name


def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )


clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
    model, clean_tokens, ioi_metric
)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
    model, corrupted_tokens, ioi_metric
)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

# and that's it  
# %% now this is focuses on the residual stream but we have already computed all the cache
def attr_patch_residual(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
):
    clean_residual, residual_labels = clean_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=True
    )
    corrupted_residual = corrupted_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=False
    )
    corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=False
    )
    residual_attr = einops.reduce(
        corrupted_grad_residual * (clean_residual - corrupted_residual),
        "component batch pos d_model -> component pos",
        "sum",
    )
    return residual_attr, residual_labels

residual_attr, residual_labels = attr_patch_residual(
    clean_cache, corrupted_cache, corrupted_grad_cache
)

from neel_plotly import imshow
imshow(
    residual_attr,
    y=residual_labels,
    yaxis="Component",
    xaxis="Position",
    title="Residual Attribution Patching",
)
#%%

def attr_patch_layer_out(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
):
    clean_layer_out, labels = clean_cache.decompose_resid(-1, return_labels=True)
    corrupted_layer_out = corrupted_cache.decompose_resid(-1, return_labels=False)
    corrupted_grad_layer_out = corrupted_grad_cache.decompose_resid(
        -1, return_labels=False
    )
    layer_out_attr = einops.reduce(
        corrupted_grad_layer_out * (clean_layer_out - corrupted_layer_out),
        "component batch pos d_model -> component pos",
        "sum",
    )
    return layer_out_attr, labels


layer_out_attr, layer_out_labels = attr_patch_layer_out(
    clean_cache, corrupted_cache, corrupted_grad_cache
)
imshow(
    layer_out_attr,
    y=layer_out_labels,
    yaxis="Component",
    xaxis="Position",
    title="Layer Output Attribution Patching",
)
# %%
def attr_patch_head_out(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
):
    HEAD_NAMES = [
        f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
    ]
    HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
    HEAD_NAMES_QKV = [
        f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]
    ]
    labels = HEAD_NAMES

    clean_head_out = clean_cache.stack_head_results(-1, return_labels=False)
    corrupted_head_out = corrupted_cache.stack_head_results(-1, return_labels=False)
    corrupted_grad_head_out = corrupted_grad_cache.stack_head_results(
        -1, return_labels=False
    )
    head_out_attr = einops.reduce(
        corrupted_grad_head_out * (clean_head_out - corrupted_head_out),
        "component batch pos d_model -> component pos",
        "sum",
    )
    return head_out_attr, labels


head_out_attr, head_out_labels = attr_patch_head_out(
    clean_cache, corrupted_cache, corrupted_grad_cache
)
imshow(
    head_out_attr,
    y=head_out_labels,
    yaxis="Component",
    xaxis="Position",
    title="Head Output Attribution Patching",
)
sum_head_out_attr = einops.reduce(
    head_out_attr,
    "(layer head) pos -> layer head",
    "sum",
    layer=model.cfg.n_layers,
    head=model.cfg.n_heads,
)
imshow(
    sum_head_out_attr,
    yaxis="Layer",
    xaxis="Head Index",
    title="Head Output Attribution Patching Sum Over Pos",
)
# %% COMPARE ATTRIBUTION PATCHING TO ACTIVATION PATCHING

attribution_cache_dict = {}
for key in corrupted_grad_cache.cache_dict.keys():
    attribution_cache_dict[key] = corrupted_grad_cache.cache_dict[key] * (
        clean_cache.cache_dict[key] - corrupted_cache.cache_dict[key]
    )
attr_cache = ActivationCache(attribution_cache_dict, model)
# %%
str_tokens = model.to_str_tokens(clean_tokens[0])
context_length = len(str_tokens)
# %%
from transformer_lens import patching
every_block_act_patch_result = patching.get_act_patch_block_every(
    model, corrupted_tokens, clean_cache, ioi_metric
)
imshow(
    every_block_act_patch_result,
    facet_col=0,
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
    title="Activation Patching Per Block (no LN model)",
    xaxis="Position",
    yaxis="Layer",
    zmax=1,
    zmin=-1,
    x=[f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
)

# %%
def get_attr_patch_block_every(attr_cache):
    resid_pre_attr = einops.reduce(
        attr_cache.stack_activation("resid_pre"),
        "layer batch pos d_model -> layer pos",
        "sum",
    )
    attn_out_attr = einops.reduce(
        attr_cache.stack_activation("attn_out"),
        "layer batch pos d_model -> layer pos",
        "sum",
    )
    mlp_out_attr = einops.reduce(
        attr_cache.stack_activation("mlp_out"),
        "layer batch pos d_model -> layer pos",
        "sum",
    )

    every_block_attr_patch_result = torch.stack(
        [resid_pre_attr, attn_out_attr, mlp_out_attr], dim=0
    )
    return every_block_attr_patch_result


every_block_attr_patch_result = get_attr_patch_block_every(attr_cache)
imshow(
    every_block_attr_patch_result,
    facet_col=0,
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
    title="Attribution Patching Per Block (no LN model)",
    xaxis="Position",
    yaxis="Layer",
    zmax=1,
    zmin=-1,
    x=[f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
)

# %%
scatter(
    y=every_block_attr_patch_result.reshape(3, -1),
    x=every_block_act_patch_result.reshape(3, -1),
    facet_col=0,
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
    title="Attribution vs Activation Patching Per Block (no LN model)",
    xaxis="Activation Patch",
    yaxis="Attribution Patch",
    hover=[
        f"Layer {l}, Position {p}, |{str_tokens[p]}|"
        for l in range(model.cfg.n_layers)
        for p in range(context_length)
    ],
    color=einops.repeat(
        torch.arange(model.cfg.n_layers), "layer -> (layer pos)", pos=context_length
    ),
    color_continuous_scale="Portland",
)

# %%
plt.scatter(every_block_act_patch_result[0].cpu().numpy().flatten(), every_block_attr_patch_result[0].cpu().numpy().flatten(), c=einops.repeat(
        torch.arange(model.cfg.n_layers), "layer -> (layer pos)", pos=context_length
    ).flatten(), cmap='copper')
min = np.min([every_block_act_patch_result[0].cpu().numpy().min(), every_block_attr_patch_result[0].cpu().numpy().min()])
max = np.max([every_block_act_patch_result[0].cpu().numpy().max(), every_block_attr_patch_result[0].cpu().numpy().max()])
plt.axvline(0, color='black', linestyle='--')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Activation Patching")
plt.ylabel("Attribution Patching")
plt.plot(np.linspace(min, max, 100), np.linspace(min, max, 100), color='black', linestyle='--')
plt.colorbar(label='Layer')
plt.title("NO LN MODEL: Patching comparison in each block residual stream")
plt.show()
# %%
