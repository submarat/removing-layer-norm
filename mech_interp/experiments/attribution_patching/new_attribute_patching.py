#%%
import sys
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import load_vanilla_small, load_noLN_small, load_baseline_small
import torch
import numpy as np
from transformers import AutoTokenizer
import einops
import matplotlib.pyplot as plt
from neel_plotly import imshow, scatter
# from ioi_utils import get_logit_diff, ioi_metric
# from mech_interp.experiments.attribution_patching.attribution_patching_utils import get_cache_fwd_and_bwd, get_attr_patch_attn_head_all_pos_every, get_attr_patch_block_every 
import pickle
from functools import partial
# from custom_patching import get_act_patch_block_every_per_prompt
import transformer_lens.patching as patching
import custom_patching
from tqdm import tqdm
from mech_interp.experiments.attribution_patching.attribution_patching_utils import get_cache_fwd_and_bwd, get_attr_patch_block_every
from transformer_lens import ActivationCache

def pickleload(path):
    with open(path, "rb") as f:
        return pickle.load(f)

prompts_and_answers = pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/final/prompts.pkl")
prompts = prompts_and_answers["prompts"]
answers = prompts_and_answers["answers"]

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


model_names = ["baseline", "vanilla", "noLN"]
models = [load_baseline_small(), load_vanilla_small(), load_noLN_small()]

for model in models:
    model.set_use_attn_result(True)
    model.set_use_attn_in(True)
    model.set_use_hook_mlp_in(True)
    
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenize prompts and get answer_tokens
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


BATCH_SIZE = 16 # Adjust as needed. 16 is a conservative choice based on other cell's successful run with this batch size.
num_examples = clean_tokens.shape[0]

for model_name, model in zip(model_names,models):
    model_results = {
        'clean_logit_diff_batch_results': [],
        'corrupted_logit_diff_batch_results': [],
        'act_patch_batch_results': [],
        'attr_patch_batch_results': [],
        'attr_patch_single_example_results': []
    }
    # with torch.no_grad():
    #     clean_logit_diff_batch_results = []
    #     corrupted_logit_diff_batch_results = []
    #     act_patch_batch_results = []

    #     for i in tqdm(range(0, num_examples, BATCH_SIZE)):
    #         # get batch indices
    #         batch_start = i
    #         batch_end = min(i + BATCH_SIZE, num_examples)
    #         # get batch of clean and corrupted tokens
    #         current_clean_tokens_batch = clean_tokens[batch_start:batch_end]
    #         current_corrupted_tokens_batch = corrupted_tokens[batch_start:batch_end]
    #         # get logits and diffs
    #         current_clean_logits, current_clean_cache = model.run_with_cache(current_clean_tokens_batch)
    #         current_corrupted_logits, _ = model.run_with_cache(current_corrupted_tokens_batch)
    #         current_clean_logits_diff = get_logit_diff(current_clean_logits, answer_token_indices[batch_start:batch_end], mean=False)
    #         current_corrupted_logits_diff = get_logit_diff(current_corrupted_logits, answer_token_indices[batch_start:batch_end], mean=False)
    #         diff_logit_diff = current_clean_logits_diff - current_corrupted_logits_diff
    #         # get answer token indices
    #         current_answer_token_indices_batch = answer_token_indices[batch_start:batch_end]
    #         # get metric
    #         metric = partial(get_logit_diff, answer_token_indices=current_answer_token_indices_batch, mean=False)
    #         # get resid pre act patch result
    #         resid_pre_act_patch_result = custom_patching.get_act_patch_resid_pre(
    #             model, current_corrupted_tokens_batch, current_clean_cache, metric
    #             )
    #         # subtract logit diff of current corrupted tokens from patch logit diff
    #         resid_pre_act_patch_result = resid_pre_act_patch_result - current_corrupted_logits_diff.reshape(1,1,-1)
    #         # append to list
    #         model_results['act_patch_batch_results'].append(resid_pre_act_patch_result)
    #         model_results['clean_logit_diff_batch_results'].append(current_clean_logits_diff)
    #         model_results['corrupted_logit_diff_batch_results'].append(current_corrupted_logits_diff)

    #     model_results['act_patch_batch_results'] = torch.cat(model_results['act_patch_batch_results'], dim=-1).permute(2,0,1)
    #     model_results['clean_logit_diff_batch_results'] = torch.cat(model_results['clean_logit_diff_batch_results'], dim=-1)
    #     model_results['corrupted_logit_diff_batch_results'] = torch.cat(model_results['corrupted_logit_diff_batch_results'], dim=-1)

    # # attribution patching:
    attribution_cache_dict = {}
    num_examples = clean_tokens.shape[0]
    for i, (single_clean_tokens, single_corrupted_tokens, single_answer_token_indices) in tqdm(enumerate(zip(clean_tokens, corrupted_tokens, answer_token_indices)), total=num_examples):
        metric = partial(get_logit_diff, answer_token_indices=single_answer_token_indices.unsqueeze(0), mean=True)
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
            attr_cache_item = corrupted_grad_cache.cache_dict[key] * (
                clean_cache.cache_dict[key] - corrupted_cache.cache_dict[key]
            )
            attribution_cache_dict[key].append(attr_cache_item)
    for key in attribution_cache_dict.keys():
        attribution_cache_dict[key] = torch.cat(attribution_cache_dict[key], dim=0)/len(attribution_cache_dict[key])
    attribution_cache = ActivationCache(attribution_cache_dict, model)
    model_results['attr_patch_single_example_results'] = get_attr_patch_resid_pre(attribution_cache, reduce_over_batch=False) 

    import os
    os.makedirs(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/final/{model_name}", exist_ok=True)
    with open(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/final/{model_name}/attribute_patching.pkl", "wb") as f:
        pickle.dump(model_results, f)


    # # attribution patching:
    attribution_cache_dict = {}
    num_examples = clean_tokens.shape[0]
    for i, (single_clean_tokens, single_corrupted_tokens, single_answer_token_indices) in tqdm(enumerate(zip(clean_tokens, corrupted_tokens, answer_token_indices)), total=num_examples):
        metric = partial(get_logit_diff, answer_token_indices=single_answer_token_indices.unsqueeze(0), mean=True)
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
            attr_cache_item = corrupted_grad_cache.cache_dict[key] * (
                clean_cache.cache_dict[key] - corrupted_cache.cache_dict[key]
            )
            attribution_cache_dict[key].append(attr_cache_item)
    for key in attribution_cache_dict.keys():
        attribution_cache_dict[key] = torch.cat(attribution_cache_dict[key], dim=0)/len(attribution_cache_dict[key])
    attribution_cache = ActivationCache(attribution_cache_dict, model)
    model_results['attr_patch_single_example_results'] = get_attr_patch_resid_pre(attribution_cache, reduce_over_batch=False) 

    import os
    os.makedirs(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/final/{model_name}", exist_ok=True)
    with open(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/final/{model_name}/attribute_patching.pkl", "wb") as f:
        pickle.dump(model_results, f)
# %%
