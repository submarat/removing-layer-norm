#%%
import sys
sys.path.append("/workspace/removing-layer-norm/")
# from mech_interp.load_models import load_nln_model, load_finetuned_model, load_baseline, load_noLN_model, load_vanilla_small
from mech_interp.load_models import load_vanilla_small, load_baseline_small, load_noLN_small
import torch
import numpy as np
from transformers import AutoTokenizer
import einops
import matplotlib.pyplot as plt
from neel_plotly import imshow, scatter
from mech_interp.experiments.attribution_patching.attribution_patching_utils import get_cache_fwd_and_bwd, get_attr_patch_attn_head_all_pos_every, get_attr_patch_block_every 
from tqdm import tqdm
import custom_patching

from functools import partial
from neel_plotly import imshow
import pickle
import os
from transformer_lens import ActivationCache

store_attn_patch_results_all_pos = False
store_attn_patch_results_by_pos = False
compute_activation_patching = True
show_visualizations = False

def process_batch_for_act_patching_resid_pre(
    model, 
    clean_tokens,
    corrupted_tokens,
    metric,
    store_location
): 
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

        corrupted_logit_diff = metric(corrupted_logits)
        resid_pre_act_patch_result = custom_patching.get_act_patch_resid_pre(
            model, corrupted_tokens, clean_cache, metric
        )
        resid_pre_act_patch_result = resid_pre_act_patch_result - corrupted_logit_diff.reshape(1,1,-1)
        os.makedirs(os.path.dirname(store_location), exist_ok=True)
        with open(store_location, "wb") as f:
            pickle.dump(resid_pre_act_patch_result, f)
    return resid_pre_act_patch_result

def get_attr_patch_resid_pre(attribution_cache, reduce_over_batch=True):
    if reduce_over_batch:
        resid_pre_attr = einops.reduce(
            attribution_cache.stack_activation("resid_pre"),
            "layer batch pos d_model -> layer pos",
            "sum",
        )
    else:
        resid_pre_attr = einops.reduce(
            attribution_cache.stack_activation("resid_pre"),
            "layer batch pos d_model -> batch layer pos",
            "sum",
        )
    return resid_pre_attr

def process_batch_for_attr_patching_resid_pre(
    model,
    clean_tokens,
    corrupted_tokens,
    answer_token_indices,
    logit_diff_metric,
    store_location
):
    attribution_cache_dict = {}
    num_examples = clean_tokens.shape[0]
    for i, (single_clean_tokens, single_corrupted_tokens, single_answer_token_indices) in tqdm(enumerate(zip(clean_tokens, corrupted_tokens, answer_token_indices)), total=num_examples):
        metric = partial(logit_diff_metric, answer_token_indices=single_answer_token_indices.unsqueeze(0), mean=True)
        clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
            model, single_clean_tokens, metric
        )
        corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
            model, single_corrupted_tokens, metric
        )
        if i == 0:
            for key in corrupted_grad_cache.cache_dict.keys():
                attribution_cache_dict[key] = []
        for key in corrupted_grad_cache.cache_dict.keys():
            attribution_cache_dict[key].append(corrupted_grad_cache.cache_dict[key] * (
                clean_cache.cache_dict[key] - corrupted_cache.cache_dict[key]
            ))
    for key in attribution_cache_dict.keys():
        attribution_cache_dict[key] = torch.cat(attribution_cache_dict[key], dim=0)
    attribution_cache = ActivationCache(attribution_cache_dict, model)
    resid_pre_attr_patch_result = get_attr_patch_resid_pre(attribution_cache, reduce_over_batch=False).permute(1,2,0)
    os.makedirs(os.path.dirname(store_location), exist_ok=True)
    with open(store_location, "wb") as f:
        pickle.dump(resid_pre_attr_patch_result, f)
    return resid_pre_attr_patch_result

def get_logit_diff(logits, answer_token_indices, mean=True):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    else: 
        raise AssertionError("Logits shape is not 2 or 3")
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    if mean:
        return (correct_logits - incorrect_logits).mean()
    else:
        return (correct_logits - incorrect_logits).squeeze(-1)

def pickleload(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# prepare model
for PROMPTS_SET in range(0, 30):
    prompts_and_answers = pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{PROMPTS_SET}/prompts.pkl")
    prompts = prompts_and_answers['prompts']
    answers = prompts_and_answers['answers']
    for MODEL_NAME in ['baseline', 'vanilla', 'noLN']: # 'baseline', 'finetuned', 'nln'
        print(f"Running {MODEL_NAME} model for prompts set {PROMPTS_SET}")
        # if MODEL_NAME == 'finetuned':
        #     model = load_finetuned_model()
        # if MODEL_NAME == 'nln':
        #     model = load_nln_model()
        if MODEL_NAME == 'baseline':
            model = load_baseline_small()
        if MODEL_NAME == 'noLN':
            model = load_noLN_small()
        if MODEL_NAME == 'vanilla':
            model = load_vanilla_small()

        # not sure why this is needed
        model.set_use_attn_result(True)
        model.set_use_attn_in(True)
        model.set_use_hook_mlp_in(True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # tokenize prompts
        clean_tokens = model.to_tokens(prompts)
        corrupted_tokens = clean_tokens[
            [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
        ]
        answer_token_indices = torch.tensor(
            [
                [model.to_single_token(answers[i][j]) for j in range(2)]
                for i in range(len(answers))
            ],
            device=model.cfg.device,
        )
        
        # ACTIVATION PATCHING
        metric = partial(get_logit_diff, answer_token_indices=answer_token_indices, mean=False)
        store_location = f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{PROMPTS_SET}/new_resid_pre_act_patch_result_{MODEL_NAME}.pkl"
        process_batch_for_act_patching_resid_pre(
            model, 
            clean_tokens, 
            corrupted_tokens, 
            metric, 
            store_location
        )

        # ATTRIBUTION PATCHING 
        store_location = f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{PROMPTS_SET}/new_resid_pre_attr_patch_result_{MODEL_NAME}.pkl"
        process_batch_for_attr_patching_resid_pre(
            model,
            clean_tokens,
            corrupted_tokens,
            answer_token_indices,
            metric,
            store_location
        )
        # clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
        #     model, clean_tokens, ioi_metric_wrapper
        # )
        # corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
        #     model, corrupted_tokens, ioi_metric_wrapper
        # )
        # print("Clean Value:", clean_value) #should be 1
        # print("Clean Activations Cached:", len(clean_cache))
        # print("Clean Gradients Cached:", len(clean_grad_cache))
        # print("Corrupted Value:", corrupted_value) #should be 0
        # print("Corrupted Activations Cached:", len(corrupted_cache))
        # print("Corrupted Gradients Cached:", len(corrupted_grad_cache))

        # attribution_cache_dict = {}
        # for key in corrupted_grad_cache.cache_dict.keys():
        #     attribution_cache_dict[key] = corrupted_grad_cache.cache_dict[key] * (
        #         clean_cache.cache_dict[key] - corrupted_cache.cache_dict[key]
        #     )
        # attr_cache = ActivationCache(attribution_cache_dict, model)
        # _
       
        # with open(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{PROMPTS_SET}/every_block_act_attr_patch_result_{MODEL_NAME}.pkl", "wb") as f:
        #     pickle.dump(every_block_attr_patch_result, f)
        
        # if store_attn_patch_results_all_pos:
        #     every_attn_attr_patch_result = get_attr_patch_attn_head_all_pos_every(attr_cache)
        #     with open(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{PROMPTS_SET}/every_attn_attr_patch_result_{MODEL_NAME}.pkl", "wb") as f:
        #         pickle.dump(every_attn_attr_patch_result, f)
        
        # if show_visualizations:
        #     imshow(
        #         every_block_attr_patch_result,
        #         facet_col=0,
        #         facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
        #         title=f"Attribution Patching Per Block ({MODEL_NAME} model, prompts set {PROMPTS_SET})",
        #         xaxis="Position",
        #         yaxis="Layer",
        #         zmax=1,
        #         zmin=-1,
        #     )
        #     if store_attn_patch_results_all_pos:
        #         imshow(
        #             every_attn_attr_patch_result,
        #             facet_col=0,
        #             facet_labels=["output", "query", "key", "value", "pattern"],
        #             title=f"Attribution Patching Per Head ({MODEL_NAME} model, prompts set {PROMPTS_SET})",
        #             xaxis="Head",
        #             yaxis="Layer",
        #             zmax=1,
        #             zmin=-1,
        #         )


# #%%
# prompts_set = 1
# def l1(act, attr):
#     return torch.abs(act - attr)

# def get_results_dict(model_name, prompts_set):
#     results = {
#         'every_block': {
#             'attribution' :pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{prompts_set}/every_block_act_attr_patch_result_{model_name}.pkl"),
#             'activation' :pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{prompts_set}/every_block_act_patch_result_{model_name}.pkl"),
#         },
#         'every_attn': {
#             'attribution' :pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{prompts_set}/every_attn_attr_patch_result_{model_name}.pkl"),
#             'activation' :pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{prompts_set}/every_attn_patch_result_{model_name}.pkl"),
#         },
#     }
#     return results

# finetuned_results = get_results_dict('finetuned', prompts_set)
# nln_results = get_results_dict('nln', prompts_set)
# baseline_results = get_results_dict('baseline', prompts_set)

# index = 0
# baseline = l1(baseline_results['every_block']['activation'][index], baseline_results['every_block']['attribution'][index]).sum(dim=-1).cpu().numpy()
# finetuned = l1(finetuned_results['every_block']['activation'][index], finetuned_results['every_block']['attribution'][index]).sum(dim=-1).cpu().numpy()   
# nln = l1(nln_results['every_block']['activation'][index], nln_results['every_block']['attribution'][index]).sum(dim=-1).cpu().numpy()

# plt.plot(baseline, label='baseline')
# plt.plot(finetuned, label='finetuned')
# plt.plot(nln, label='nln')
# plt.xlabel('Layer')
# plt.ylabel('L1 Loss (summed over positions)')
# plt.title('L1 Loss of Activation vs Attribution Patching')
# plt.legend()
# plt.show()

# #%%
# batch_size = 8
# with torch.no_grad():
#     from transformer_lens import patching
#     for i in tqdm(range(0, len(clean_tokens), batch_size)):
#         def ioi_metric_wrapper(logits, answer_token_indices=answer_token_indices[i:i+batch_size]):
#             return ioi_metric(logits, answer_token_indices)
#         _, clean_cache = model.run_with_cache(clean_tokens[i:i+batch_size])
#         attn_patch_result = patching.get_act_patch_attn_head_by_pos_every(
#             model, corrupted_tokens[i:i+batch_size], clean_cache, ioi_metric_wrapper
#         )
#         break
# #%%
# attn_patch_result.shape

# #%%
# batch_size = 8
# with torch.no_grad():
#     from transformer_lens import patching
#     for i in tqdm(range(0, len(clean_tokens), batch_size)):
#         def ioi_metric_wrapper(logits, answer_token_indices=answer_token_indices[i:i+batch_size]):
#             return ioi_metric(logits, answer_token_indices)
#         _, clean_cache = model.run_with_cache(clean_tokens[i:i+batch_size])
#         attn_patch_result = patching.get_act_patch_attn_head_all_pos_every(
#             model, corrupted_tokens[i:i+batch_size], clean_cache, ioi_metric_wrapper
#         )
#         break

# #%%
# every_block_act_patch_result
# #%%
# ######## ACTIVATION PATCHING ########
# from transformer_lens import patching
# every_block_act_patch_result = patching.get_act_patch_block_every(
#     model, corrupted_tokens, clean_cache, ioi_metric
# )

# from neel_plotly import imshow
# imshow(
#     every_block_act_patch_result[0],
#     title="Activation Patching Per Block",
#     xaxis="Position",
#     yaxis="Layer",
# )

# ### OK NOW WE CAN DO THE ATTRIBUTION PATCHING
# from transformer_lens import (
#     # HookedTransformer,
#     # HookedTransformerConfig,
#     # FactoredMatrix,
#     ActivationCache,
# )
# filter_not_qkv_input = lambda name: "_input" not in name

# def get_cache_fwd_and_bwd(model, tokens, metric):
#     model.reset_hooks()
#     cache = {}

#     def forward_cache_hook(act, hook):
#         cache[hook.name] = act.detach()

#     model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")

#     grad_cache = {}

#     def backward_cache_hook(act, hook):
#         grad_cache[hook.name] = act.detach()

#     model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

#     value = metric(model(tokens))
#     value.backward()
#     model.reset_hooks()
#     return (
#         value.item(),
#         ActivationCache(cache, model),
#         ActivationCache(grad_cache, model),
#     )


# clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(
#     model, clean_tokens, ioi_metric
# )
# print("Clean Value:", clean_value)
# print("Clean Activations Cached:", len(clean_cache))
# print("Clean Gradients Cached:", len(clean_grad_cache))
# corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(
#     model, corrupted_tokens, ioi_metric
# )
# print("Corrupted Value:", corrupted_value)
# print("Corrupted Activations Cached:", len(corrupted_cache))
# print("Corrupted Gradients Cached:", len(corrupted_grad_cache))


# def attr_patch_residual(
#     clean_cache: ActivationCache,
#     corrupted_cache: ActivationCache,
#     corrupted_grad_cache: ActivationCache,
# ):
#     clean_residual, residual_labels = clean_cache.accumulated_resid(
#         -1, incl_mid=True, return_labels=True
#     )
#     corrupted_residual = corrupted_cache.accumulated_resid(
#         -1, incl_mid=True, return_labels=False
#     )
#     corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(
#         -1, incl_mid=True, return_labels=False
#     )
#     residual_attr = einops.reduce(
#         corrupted_grad_residual * (clean_residual - corrupted_residual),
#         "component batch pos d_model -> component pos",
#         "sum",
#     )
#     return residual_attr, residual_labels


# residual_attr, residual_labels = attr_patch_residual(
#     clean_cache, corrupted_cache, corrupted_grad_cache
# )
# from neel_plotly import imshow
# imshow(
#     residual_attr,
#     y=residual_labels,
#     yaxis="Component",
#     xaxis="Position",
#     title="Residual Attribution Patching",
# )
# def attr_patch_layer_out(
#     clean_cache: ActivationCache,
#     corrupted_cache: ActivationCache,
#     corrupted_grad_cache: ActivationCache,
# ):
#     clean_layer_out, labels = clean_cache.decompose_resid(-1, return_labels=True)
#     corrupted_layer_out = corrupted_cache.decompose_resid(-1, return_labels=False)
#     corrupted_grad_layer_out = corrupted_grad_cache.decompose_resid(
#         -1, return_labels=False
#     )
#     layer_out_attr = einops.reduce(
#         corrupted_grad_layer_out * (clean_layer_out - corrupted_layer_out),
#         "component batch pos d_model -> component pos",
#         "sum",
#     )
#     return layer_out_attr, labels


# layer_out_attr, layer_out_labels = attr_patch_layer_out(
#     clean_cache, corrupted_cache, corrupted_grad_cache
# )
# imshow(
#     layer_out_attr,
#     y=layer_out_labels,
#     yaxis="Component",
#     xaxis="Position",
#     title="Layer Output Attribution Patching",
# )
# def attr_patch_head_out(
#     clean_cache: ActivationCache,
#     corrupted_cache: ActivationCache,
#     corrupted_grad_cache: ActivationCache,
# ):
#     HEAD_NAMES = [
#         f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
#     ]
#     HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
#     HEAD_NAMES_QKV = [
#         f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]
#     ]
#     labels = HEAD_NAMES

#     clean_head_out = clean_cache.stack_head_results(-1, return_labels=False)
#     corrupted_head_out = corrupted_cache.stack_head_results(-1, return_labels=False)
#     corrupted_grad_head_out = corrupted_grad_cache.stack_head_results(
#         -1, return_labels=False
#     )
#     head_out_attr = einops.reduce(
#         corrupted_grad_head_out * (clean_head_out - corrupted_head_out),
#         "component batch pos d_model -> component pos",
#         "sum",
#     )
#     return head_out_attr, labels


# head_out_attr, head_out_labels = attr_patch_head_out(
#     clean_cache, corrupted_cache, corrupted_grad_cache
# )
# imshow(
#     head_out_attr,
#     y=head_out_labels,
#     yaxis="Component",
#     xaxis="Position",
#     title="Head Output Attribution Patching",
# )
# sum_head_out_attr = einops.reduce(
#     head_out_attr,
#     "(layer head) pos -> layer head",
#     "sum",
#     layer=model.cfg.n_layers,
#     head=model.cfg.n_heads,
# )
# imshow(
#     sum_head_out_attr,
#     yaxis="Layer",
#     xaxis="Head Index",
#     title="Head Output Attribution Patching Sum Over Pos",
# )
# # COMPARE ATTRIBUTION PATCHING TO ACTIVATION PATCHING
# def get_attr_patch_block_every(attr_cache):
#     resid_pre_attr = einops.reduce(
#         attr_cache.stack_activation("resid_pre"),
#         "layer batch pos d_model -> layer pos",
#         "sum",
#     )
#     attn_out_attr = einops.reduce(
#         attr_cache.stack_activation("attn_out"),
#         "layer batch pos d_model -> layer pos",
#         "sum",
#     )
#     mlp_out_attr = einops.reduce(
#         attr_cache.stack_activation("mlp_out"),
#         "layer batch pos d_model -> layer pos",
#         "sum",
#     )

#     every_block_attr_patch_result = torch.stack(
#         [resid_pre_attr, attn_out_attr, mlp_out_attr], dim=0
#     )
#     return every_block_attr_patch_result

# attribution_cache_dict = {}
# for key in corrupted_grad_cache.cache_dict.keys():
#     attribution_cache_dict[key] = corrupted_grad_cache.cache_dict[key] * (
#         clean_cache.cache_dict[key] - corrupted_cache.cache_dict[key]
#     )
# attr_cache = ActivationCache(attribution_cache_dict, model)

# every_block_attr_patch_result = get_attr_patch_block_every(attr_cache)
# imshow(
#     every_block_attr_patch_result,
#     facet_col=0,
#     facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
#     title="Attribution Patching Per Block (finetuned model)",
#     xaxis="Position",
#     yaxis="Layer",
#     zmax=1,
#     zmin=-1,
#     x=[f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
# )

# from transformer_lens import patching
# str_tokens = model.to_str_tokens(clean_tokens[0])
# context_length = len(str_tokens)
# every_block_act_patch_result = patching.get_act_patch_block_every(
#     model, corrupted_tokens, clean_cache, ioi_metric
# )
# imshow(
#     every_block_act_patch_result,
#     facet_col=0,
#     facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
#     title="Activation Patching Per Block (finetuned model)",
#     xaxis="Position",
#     yaxis="Layer",
#     zmax=1,
#     zmin=-1,
#     x=[f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))],
# )
# from neel_plotly import scatter
# scatter(
#     y=every_block_attr_patch_result.reshape(3, -1),
#     x=every_block_act_patch_result.reshape(3, -1),
#     facet_col=0,
#     facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
#     title="Attribution vs Activation Patching Per Block (finetuned model)",
#     xaxis="Activation Patch",
#     yaxis="Attribution Patch",
#     hover=[
#         f"Layer {l}, Position {p}, |{str_tokens[p]}|"
#         for l in range(model.cfg.n_layers)
#         for p in range(context_length)
#     ],
#     color=einops.repeat(
#         torch.arange(model.cfg.n_layers), "layer -> (layer pos)", pos=context_length
#     ),
#     color_continuous_scale="Portland",
# )


# plt.scatter(every_block_act_patch_result[0].cpu().numpy().flatten(), every_block_attr_patch_result[0].cpu().numpy().flatten(), c=einops.repeat(
#         torch.arange(model.cfg.n_layers), "layer -> (layer pos)", pos=context_length
#     ).flatten(), cmap='copper')
# min = np.min([every_block_act_patch_result[0].cpu().numpy().min(), every_block_attr_patch_result[0].cpu().numpy().min()])
# max = np.max([every_block_act_patch_result[0].cpu().numpy().max(), every_block_attr_patch_result[0].cpu().numpy().max()])
# plt.axvline(0, color='black', linestyle='--')
# plt.axhline(0, color='black', linestyle='--')
# plt.xlabel("Activation Patching")
# plt.ylabel("Attribution Patching")
# plt.plot(np.linspace(min, max, 100), np.linspace(min, max, 100), color='black', linestyle='--')
# plt.colorbar(label='Layer')
# plt.title(f"{model_name} model: Patching comparison in each block residual stream")
# plt.show()

# # save results
# torch.save(every_block_act_patch_result, f"every_block_act_patch_result_{model_name}_set_{set_prompts}.pt")
# torch.save(every_block_attr_patch_result, f"every_block_attr_patch_result_{model_name}_set_{set_prompts}.pt")
