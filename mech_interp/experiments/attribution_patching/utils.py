# useful functions 
import numpy as np
import pickle
import sys
import os
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_dataset import DataManager
from tqdm import tqdm

#%%
def get_mean_activations(model, positions, num_samples=10000, batch_size=100, max_context=50):
    assert num_samples % batch_size == 0
    num_batches = num_samples // batch_size
    dm = DataManager(dataset_name='luca-pile', num_samples=num_samples, batch_size=batch_size, max_context=max_context)
    dataloader = dm.create_dataloader()
    mean_activations = {position: 0 for position in positions}
    for tokens in tqdm(dataloader, total=num_batches):
        model.reset_hooks()
        _, cache = model.run_with_cache(tokens)
        for position in positions:
            mean_activations[position] += cache[position].mean(dim=(0,1))
    for position in positions:
        mean_activations[position] /= num_batches
    return mean_activations