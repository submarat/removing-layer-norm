# %%
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append("../..")
from mech_interp.load_models import ModelFactory
from mech_interp.load_dataset import DataLoader

# parameters
layers_list = [2, 4, 6, 8, 10, 11]
n_seqs = 10
batch_size = 5
max_context = 1024
d = 768
prepend_bos = False
use_first_token = True
fold_ln = False
center_unembed=False
center_writing_weights=False

# functions
def kurtosis(x):
    return np.mean(x**4) / np.mean(x**2)**2

# initialize model factory
model_factory = ModelFactory(
    model_names=['baseline', 'noLN'],
    model_dir='../models',
    device='cuda',
    fold_ln=fold_ln,
    center_unembed=center_unembed,
    center_writing_weights=center_writing_weights,
    eval_mode=True,
    model_size="small",
)

# initialize dataloader
dataloader = DataLoader(
    dataset_name='apollo-owt',
    batch_size=batch_size,
    max_context=max_context,
    num_samples=n_seqs,
    prepend_bos=prepend_bos,
).create_dataloader()

# compute s values
all_s_values = {
    'baseline': {},
    'noLN': {}
}
max_len_seq = max_context - 1 if use_first_token == False else max_context
with torch.no_grad():
    all_acts = np.zeros(( n_seqs, max_len_seq, d))
    for model_name, model in model_factory.models.items():
        for layer in layers_list:
            for i, batch in tqdm(enumerate(dataloader), total=n_seqs//batch_size):
                batch = batch.to(model.cfg.device)
                _, cache = model.run_with_cache(batch)
                acts = cache[f'blocks.{layer}.hook_resid_pre'].cpu().numpy()
                if use_first_token == False:
                    acts = acts[:, 1:, :] 
                all_acts[i*batch_size:(i+1)*batch_size, :, :] = acts
            all_s_values[model_name][layer] = np.sqrt(np.mean(all_acts.reshape(-1, d)**2, axis=0))

# compute kurtosis
all_kurtosis = {
    'baseline': {},
    'noLN': {}
}
for model_name in all_s_values.keys():
    for layer in all_s_values[model_name].keys():
        all_kurtosis[model_name][layer] = kurtosis(all_s_values[model_name][layer])

# plot kurtosis
for model in model_factory.models.keys():
    plt.plot(layers_list, [all_kurtosis[model][layer] for layer in layers_list], '*-', label=model)
    plt.title(f'kurtosis use_first_token={use_first_token}')
    plt.xlabel('layer')
    plt.ylabel('kurtosis')
# plt.ylim(0, 300)
plt.legend()
plt.show()

# %%
