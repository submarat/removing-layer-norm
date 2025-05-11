#%%
import sys
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import ModelFactory
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

store_attn_patch_results_all_pos = False
store_attn_patch_results_by_pos = False
compute_activation_patching = True
show_visualizations = False

factory = ModelFactory( 
    ['baseline', 'finetuned', 'noLN'],
    model_dir="models",
    model_size="small"
)

# prepare model
for PROMPTS_SET in range(0, 30):
    prompts_and_answers = pickleload(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{PROMPTS_SET}/prompts.pkl")
    prompts = prompts_and_answers['prompts']
    answers = prompts_and_answers['answers']
    for MODEL_NAME in ['baseline', 'vanilla', 'noLN']: 
        print(f"Running {MODEL_NAME} model for prompts set {PROMPTS_SET}")

        if MODEL_NAME == 'baseline':
            model = factory.models['baseline']
        if MODEL_NAME == 'noLN':
            model = factory.models['noLN']
        if MODEL_NAME == 'vanilla':
            model = factory.models['finetuned']

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
