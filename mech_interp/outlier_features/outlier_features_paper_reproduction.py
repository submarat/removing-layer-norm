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
n_seqs = 5
batch_size = 5
max_context = 1024
d = 768
prepend_bos = False
# use_first_token = True
# fold_ln = False
center_unembed=False
center_writing_weights=False

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for use_first_token in [True, False]:
    for fold_ln in [False, True]:
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


        hidden_states_rms = {
            'baseline': {},
            'noLN': {}
        }
        hidden_states_kurtosis = {
            'baseline': {},
            'noLN': {}
        }

        with torch.no_grad():
            for i in range(len(layers_list)):
                layer = layers_list[i]
                for model_name, model in model_factory.models.items():
                    
                    for i, batch in tqdm(enumerate(dataloader), total=n_seqs//batch_size):
                        batch = batch.to(model.cfg.device)
                        _, cache = model.run_with_cache(batch)
                        acts = cache[f'blocks.{layer}.hook_resid_pre']
                    
                        x = acts
                        if use_first_token == False:
                            x = x[:, 1:, :]

                        # Convert to float64 for numerical stability
                        x = x.to(torch.float64) # i added this

                        # centre across width (optional)
                        x = x - x.mean(dim=-1, keepdim=True)

                        # compute rms
                        x_rms = (x**2).mean().sqrt() 
                        hidden_states_rms[model_name][layer] = x_rms.cpu().numpy()

                        # normalise and flatten
                        flat_x = x.view(-1, x.shape[-1]) / x_rms # (Batch * Seqlen, Width)
                        bs, dim = flat_x.shape

                        # Feature Gram Statistics
                        feat_gram = flat_x.T @ flat_x / bs
                        feat_rms_vec = torch.diag(feat_gram)
                        hidden_states_kurtosis[model_name][layer]= feat_rms_vec.var().cpu().numpy()
        
        for model_name in hidden_states_kurtosis.keys():
            axs[int(use_first_token), int(fold_ln)].plot(layers_list, [hidden_states_kurtosis[model_name][layer] for layer in layers_list], '*-', label=model_name)
        axs[int(use_first_token), int(fold_ln)*-1 + 1].set_xlabel('layer')
        axs[int(use_first_token), int(fold_ln)*-1 + 1].set_ylabel('kurtosis')
        axs[int(use_first_token), int(fold_ln)*-1 + 1].set_title(f'kurtosis (use_first_token={use_first_token}, fold_ln={fold_ln})')
        axs[int(use_first_token), int(fold_ln)*-1 + 1].set_ylim(0, d+10)
        
plt.legend()
plt.savefig(f'kurtosis combinations.png')
plt.show()
# %%
 