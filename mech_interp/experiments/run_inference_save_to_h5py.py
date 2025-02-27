#%%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import h5py
from mech_interp.models.load_models import load_baseline, load_finetuned_model, load_nln_model
import numpy as np
from tqdm import tqdm
import os
import torch.nn as nn
import itertools

class JSDivergence(nn.Module):
    def __init__(self, eps=1e-8):  # Use 1e-8 to match precision
        super(JSDivergence, self).__init__()
        self.eps = eps

    def kl_divergence(self, p, q):
        """Compute KL divergence between p and q distributions."""
        # Add epsilon to avoid numerical issues with log(0)
        p_safe = p + self.eps
        q_safe = q + self.eps
        kl_div = torch.sum(p_safe * torch.log(p_safe / q_safe), dim=-1)
        return kl_div
    
    def forward(self, p, q):
        """Compute Jensen-Shannon divergence between p and q distributions."""
        m = 0.5 * (p + q)  # Mixture distribution
        jsd = 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)
        return jsd

class ModelManager:
    def __init__(self, model_names, device):
        self.device = device
        self.models = self._load_models(model_names)
        self.model_pairs = list(itertools.combinations(model_names, 2))
        
    def _load_models(self, model_names):
        models_dict = {}
        for name in model_names:
            if name == 'baseline':
                models_dict[name] = load_baseline()
            elif name == 'finetuned':
                models_dict[name] = load_finetuned_model()
            elif name == 'noLN':
                models_dict[name] = load_nln_model()
            else:
                raise ValueError(f"Unknown model: {name}")
            models_dict[name] = models_dict[name].to(self.device)
        return models_dict

class DatasetManager:
    @staticmethod
    def get_dataset(dataset_name, num_samples):
        if dataset_name == 'apollo-pile':
            dataset = load_dataset('apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2', 
                                 streaming=True, split="train")
        elif dataset_name == 'apollo-owt':
            dataset = load_dataset('apollo-research/Skylion007-openwebtext-tokenizer-gpt2', 
                                 streaming=True, split="train")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return dataset.shuffle(seed=42).take(num_samples)

class H5FileManager:
    def __init__(self, output_file, num_samples, sequence_length, models):
        self.output_file = output_file
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.models = models
        self.model_pairs = list(itertools.combinations(models, 2))
        
    def initialize_file(self):
        with h5py.File(self.output_file, 'w') as f:
            # Create token dataset
            f.create_dataset('tokens', 
                           shape=(self.num_samples, self.sequence_length),
                           dtype=np.int32,
                           chunks=True)
            
            # Create CE loss datasets
            ce_loss_group = f.create_group('CE_loss')
            for model_name in self.models:
                ce_loss_group.create_dataset(model_name,
                                          shape=(self.num_samples, self.sequence_length-1),
                                          dtype=np.float16,
                                          chunks=True)
                
            # Create JSD dataset
            jsd_group = f.create_group('JSD')
            for model1, model2 in self.model_pairs:
                jsd_group.create_dataset(f'{model1}_vs_{model2}',
                                           shape=(self.num_samples, self.sequence_length),
                                           dtype=np.float16,
                                           chunks=True)
            
            # Create CE loss difference dataset
            ce_diff_group = f.create_group('CE_diff')
            for model1, model2 in self.model_pairs:
                ce_diff_group.create_dataset(f'{model1}_vs_{model2}',
                                          shape=(self.num_samples, self.sequence_length-1),
                                          dtype=np.float16,
                                          chunks=True)

class InferenceRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize managers
        self.model_manager = ModelManager(config['models'], self.device)
        self.h5_manager = H5FileManager(config['output_file'], 
                                      config['num_samples'],
                                      config['sequence_length'],
                                      config['models'])
        
        # Initialize loss functions
        self.jsd_loss_fn = JSDivergence().to(self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
    def run_inference(self):
        # Prepare dataset
        dataset = DatasetManager.get_dataset(self.config['dataset'], self.config['num_samples'])
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Calculate total number of batches
        total_batches = (self.config['num_samples'] + self.config['batch_size'] - 1) // self.config['batch_size']
        
        # Initialize H5 file
        self.h5_manager.initialize_file()
        
        # Run inference
        with h5py.File(self.config['output_file'], 'r+') as f:
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, total=total_batches, desc="Processing batches")):
                    self._process_batch(i, batch, f)
    
    def _process_batch(self, batch_idx, batch, h5_file):
        # Get and process tokens
        input_ids = torch.stack(batch['input_ids'], 1)[:, :self.config['sequence_length']]
        num_samples_in_batch = input_ids.shape[0]
        batch_start = batch_idx * self.config['batch_size']
        batch_end = batch_start + num_samples_in_batch
        
        # Save tokens
        h5_file['tokens'][batch_start:batch_end] = input_ids.cpu().numpy()
        
        # Get model outputs and compute losses
        input_ids = input_ids.to(self.device)
        logits = {name: model(input_ids) for name, model in self.model_manager.models.items()}
        
        # Compute and save CE loss
        ce_losses = {}
        for model_name, model_logits in logits.items():
            ce_loss = self._compute_ce_loss(model_logits, input_ids, num_samples_in_batch)
            h5_file['CE_loss'][model_name][batch_start:batch_end] = ce_loss.cpu().numpy()
            ce_losses[model_name] = ce_loss
        
        # Compute and save CE loss differences
        for model1, model2 in self.model_manager.model_pairs:
            ce_diff = ce_losses[model1] - ce_losses[model2]
            h5_file['CE_diff'][f'{model1}_vs_{model2}'][batch_start:batch_end] = ce_diff.cpu().numpy()
        
        # Compute and save JSD
        for model1, model2 in self.model_manager.model_pairs:
            jsd_loss = self._compute_jsd_loss(logits[model1], logits[model2], num_samples_in_batch)
            h5_file['JSD'][f'{model1}_vs_{model2}'][batch_start:batch_end] = jsd_loss.cpu().numpy()
    
    def _compute_ce_loss(self, logits, input_ids, num_samples_in_batch):
        ce_loss = self.ce_loss_fn(logits[:, :-1, :].flatten(end_dim=-2), input_ids[:, 1:].flatten())
        return ce_loss.reshape(num_samples_in_batch, -1)
    
    def _compute_jsd_loss(self, logits1, logits2, num_samples_in_batch):
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
        jsd_loss = self.jsd_loss_fn(probs1, probs2)
        return jsd_loss.reshape(num_samples_in_batch, self.config['sequence_length'])

def main():
    for dataset in ['apollo-pile', 'apollo-owt']:
        config = {
            'dataset': dataset,
            'models': ['baseline', 'finetuned', 'noLN'],
            'num_samples': 10000,
            'sequence_length': 50,
            'batch_size': 50,
            'folder': '/workspace/removing-layer-norm/mech_interp/data/inference_logs/',
        }
        config['folder'] = os.path.join(config['folder'], f"dataset_{config['dataset']}_samples_{config['num_samples']}_seqlen_{config['sequence_length']}")
        
        # Setup output path
        os.makedirs(config['folder'], exist_ok=True)
        config['output_file'] = f"{config['folder']}/inference_results.h5"
        
        # Run inference
        runner = InferenceRunner(config)
        runner.run_inference()

if __name__ == "__main__":
    main()# %%

# %%
