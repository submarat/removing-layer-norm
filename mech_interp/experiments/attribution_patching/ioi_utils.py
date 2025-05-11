#%%
from ioi_dataset import IOIDataset
import sys
import pickle
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import load_nln_model, load_finetuned_model

def reorder_list(items, n, order):
    result = []
    for i in range(0, len(items), n):
        group = items[i:i+n]
        if len(group) == n:  # Only reorder complete groups
            result.extend([group[j] for j in order])
        else:
            result.extend(group)  # Keep partial groups as is
    return result

def get_bisymmetric_prompts_and_answers(total_N, nb_templates):
    d1 = IOIDataset(
            prompt_type="BABA",
            N=total_N*2,
            symmetric=True, 
            nb_templates=nb_templates, 
        )
    d2 = d1.gen_flipped_prompts(("IO", "S1"))
    c = [item for pair in zip(d1.ioi_prompts, d2.ioi_prompts) for item in pair]
    prompts = [metadata["text"].rsplit(' ', 1)[0] for metadata in c]
    answers = [(' ' + metadata["IO"], ' ' + metadata["S"]) for metadata in c]
    prompts = reorder_list(prompts, 4, [0, 3, 1, 2])
    answers = reorder_list(answers, 4, [0, 3, 1, 2])
    return prompts, answers

def get_logit_diff(logits, answer_token_indices, mean=True):
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    if mean:
        return (correct_logits - incorrect_logits).mean()
    else:
        return correct_logits - incorrect_logits

def ioi_metric(logits, answer_token_indices, mean=None, corrupted_baseline=-3.6164, clean_baseline= 3.6164):
    if mean != None:
        return (get_logit_diff(logits, answer_token_indices, mean=mean) - corrupted_baseline) / (clean_baseline - corrupted_baseline)
    else:
        return (get_logit_diff(logits, answer_token_indices) - corrupted_baseline) / (clean_baseline - corrupted_baseline)


# # # %%
# for i in range(0, 30):
#     # generate prompts
#     prompts, answers = get_bisymmetric_prompts_and_answers(
#         total_N=4,
#         nb_templates=1
#     )
#     print(prompts)
#     print(answers)
#     import pickle
#     import os
#     os.makedirs(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{i}", exist_ok=True)
#     with open(f"/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_{i}/prompts.pkl", "wb") as f:
#         pickle.dump({"prompts": prompts, "answers": answers}, f)

# import os
# prompts, answers = get_bisymmetric_prompts_and_answers(500, 1)
# path = "/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/final/prompts.pkl"
# os.makedirs(os.path.dirname(path), exist_ok=True)
# with open(path, "wb") as f:
#     pickle.dump({"prompts": prompts, "answers": answers}, f)

# import os
# prompts, answers = get_bisymmetric_prompts_and_answers(16, 1)
# path = "/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/final/prompts_small.pkl"
# os.makedirs(os.path.dirname(path), exist_ok=True)
# with open(path, "wb") as f:
#     pickle.dump({"prompts": prompts, "answers": answers}, f)
# # %%

# %%
