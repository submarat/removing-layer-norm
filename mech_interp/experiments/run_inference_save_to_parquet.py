#%%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
from mech_interp.models.load_models import load_baseline, load_finetuned_model, load_nln_model
import numpy as np
from tqdm import tqdm
import os
import torch.nn as nn
import itertools
from transformers import GPT2Tokenizer

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
    

class TopKJSDivergence(nn.Module):
    def __init__(self, k=50, eps=1e-8):
        super(TopKJSDivergence, self).__init__()
        self.k = k
        self.eps = eps

    def kl_divergence(self, p, q):
        """Compute KL divergence between p and q distributions."""
        # Add epsilon to avoid numerical issues with log(0)
        p_safe = p + self.eps
        q_safe = q + self.eps
        kl_div = torch.sum(p_safe * torch.log(p_safe / q_safe), dim=-1)
        return kl_div
    
    def forward(self, p, q):
        """Compute Jensen-Shannon divergence between p and q using only top-k tokens from each."""
        # Get the indices of the top-k values in each distribution
        _, p_indices = torch.topk(p, self.k, dim=-1)
        _, q_indices = torch.topk(q, self.k, dim=-1)
        
        # Create a mask for the union of top-k indices
        vocab_size = p.size(-1)
        batch_size = p.size(0)
        seq_len = p.size(1) if p.dim() > 2 else 1
        
        # Reshape if needed for proper indexing
        if p.dim() > 2:
            p_indices = p_indices.reshape(batch_size * seq_len, self.k)
            q_indices = q_indices.reshape(batch_size * seq_len, self.k)
            p_flat = p.reshape(batch_size * seq_len, vocab_size)
            q_flat = q.reshape(batch_size * seq_len, vocab_size)
        else:
            p_flat = p
            q_flat = q
            
        # Create masks for top-k tokens
        mask = torch.zeros_like(p_flat, dtype=torch.bool)
        
        # Fill in the mask with the union of top-k indices
        batch_indices = torch.arange(mask.size(0)).unsqueeze(1).expand(-1, self.k)
        mask.scatter_(1, p_indices, True)
        mask.scatter_(1, q_indices, True)
        
        # Apply the mask to get only the top-k probabilities
        # Zero out probabilities not in the top-k union
        p_masked = p_flat.clone()
        q_masked = q_flat.clone()
        
        p_masked[~mask] = 0.0
        q_masked[~mask] = 0.0
        
        # Renormalize the masked distributions to sum to 1
        p_sum = p_masked.sum(dim=-1, keepdim=True)
        q_sum = q_masked.sum(dim=-1, keepdim=True)
        
        p_masked = p_masked / (p_sum + self.eps)
        q_masked = q_masked / (q_sum + self.eps)
        
        # Compute JS divergence on the masked distributions
        m_masked = 0.5 * (p_masked + q_masked)
        jsd = 0.5 * self.kl_divergence(p_masked, m_masked) + 0.5 * self.kl_divergence(q_masked, m_masked)
        
        # Reshape back to original dimensions if needed
        if p.dim() > 2:
            jsd = jsd.reshape(batch_size, seq_len)
            
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

class DataLoaderManager:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    def get_dataset(self, dataset_name, num_samples):
        if dataset_name == 'apollo-pile':
            dataset = load_dataset('apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2', 
                                 streaming=True, split="train")
        elif dataset_name == 'apollo-owt':
            dataset = load_dataset('apollo-research/Skylion007-openwebtext-tokenizer-gpt2', 
                                 streaming=True, split="train")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return dataset.shuffle(seed=42).take(num_samples)

    def get_dataloader(self, dataset_name, num_samples, batch_size, sequence_length):
        dataset = self.get_dataset(dataset_name, num_samples)
        dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=False,
                        collate_fn=lambda x: {'input_ids': torch.tensor([item['input_ids'][:sequence_length] for item in x])})  
        return dataloader
        


class ParquetManager:
    def __init__(self, output_file, models, save_text=False):
        self.output_file = output_file
        self.models = models
        self.model_pairs = list(itertools.combinations(models, 2))
        self.data = []
        self.save_text = save_text
        # Load tokenizer for text conversion
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    def add_batch_data(self, input_ids, ce_losses, ce_diffs, jsd_losses, topk_jsd_losses, batch_idx, batch_size, save_text):
        """Add batch data to the list that will be converted to a DataFrame."""
        batch_size, seq_len = input_ids.shape
        
        # For each position in each sequence in the batch
        for seq_idx in range(batch_size):
            for pos in range(seq_len - 1):  # -1 because we need next token for each context
                # Get context and target tokens
                context = input_ids[seq_idx, :pos+1].tolist()
                next_token = input_ids[seq_idx, pos+1].item()
                last_token = context[-1] if context else None
                
                # Create base record with grouped columns
                record = {
                    'sequence_idx': batch_idx * batch_size + seq_idx,
                    'context_length': len(context),
                    'full_context': context,
                    'last_token': last_token,
                    'following_token': next_token,
                }
                if save_text:
                    context_text = self.tokenizer.decode(context)
                    last_token_text = self.tokenizer.decode(last_token)
                    next_token_text = self.tokenizer.decode(next_token)
                    record['full_context_text'] = context_text
                    record['last_token_text'] = last_token_text
                    record['following_token_text'] = next_token_text
                
                # Add CE losses for each model
                for model_name in self.models:
                    record[f'ce_{model_name}'] = ce_losses[model_name][seq_idx, pos].item()
                
                # Add CE differences for each model pair
                for model1, model2 in self.model_pairs:
                    pair_name = f'{model1}_vs_{model2}'
                    record[f'ce_diff_{pair_name}'] = ce_diffs[pair_name][seq_idx, pos].item()
                    record[f'jsd_{pair_name}'] = jsd_losses[pair_name][seq_idx, pos].item()
                    record[f'topk_jsd_{pair_name}'] = topk_jsd_losses[pair_name][seq_idx, pos].item()
                
                self.data.append(record)
    
    def save_to_parquet(self):
        """Convert accumulated data to DataFrame and save as parquet."""
        df = pd.DataFrame(self.data)
        df.to_parquet(self.output_file, index=False)
        print(f"Saved data to {self.output_file}")

class InferenceRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize managers
        self.model_manager = ModelManager(config['models'], self.device)
        self.dataloader_manager = DataLoaderManager()
        self.parquet_manager = ParquetManager(
            config['output_file'], 
            config['models']
        )
        
        # Initialize loss functions
        self.jsd_loss_fn = JSDivergence().to(self.device)
        self.topk_jsd_loss_fn = TopKJSDivergence(k=50).to(self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device)
        
    def run_inference(self):
        # Prepare dataset
        dataloader = self.dataloader_manager.get_dataloader(
            dataset_name=self.config['dataset'],
            num_samples=self.config['num_samples'],
            batch_size=self.config['batch_size'],
            sequence_length=self.config['sequence_length']
        )
        # Calculate total number of batches
        total_batches = (self.config['num_samples'] + self.config['batch_size'] - 1) // self.config['batch_size']
        
        # Run inference
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=total_batches, desc="Processing batches")):
                self._process_batch(i, batch)
        
        # Save all data to parquet file
        self.parquet_manager.save_to_parquet()
    
    def _process_batch(self, batch_idx, batch):
        # Get and process tokens
        input_ids = batch['input_ids']
        num_samples_in_batch = input_ids.shape[0]
        
        # Get model outputs and compute losses
        input_ids = input_ids.to(self.device)
        logits = {name: model(input_ids) for name, model in self.model_manager.models.items()}
        
        # Compute CE loss
        ce_losses = {}
        for model_name, model_logits in logits.items():
            ce_loss = self._compute_ce_loss(model_logits, input_ids, num_samples_in_batch)
            ce_losses[model_name] = ce_loss
        
        # Compute CE loss differences
        ce_diffs = {}
        for model1, model2 in self.model_manager.model_pairs:
            pair_name = f'{model1}_vs_{model2}'
            ce_diffs[pair_name] = ce_losses[model1] - ce_losses[model2]
        
        # Compute JSD and TopK JSD
        jsd_losses = {}
        topk_jsd_losses = {}
        for model1, model2 in self.model_manager.model_pairs:
            pair_name = f'{model1}_vs_{model2}'
            jsd_losses[pair_name] = self._compute_jsd_loss(logits[model1], logits[model2], num_samples_in_batch)
            topk_jsd_losses[pair_name] = self._compute_topk_jsd_loss(logits[model1], logits[model2], num_samples_in_batch)
        
        # Add batch data to parquet manager
        self.parquet_manager.add_batch_data(
            input_ids.cpu(),
            ce_losses,
            ce_diffs,
            jsd_losses,
            topk_jsd_losses,
            batch_idx,
            self.config['batch_size'],
            self.config['save_text']
        )
    
    def _compute_ce_loss(self, logits, input_ids, num_samples_in_batch):
        ce_loss = self.ce_loss_fn(logits[:, :-1, :].flatten(end_dim=-2), input_ids[:, 1:].flatten())
        return ce_loss.reshape(num_samples_in_batch, -1)
    
    def _compute_jsd_loss(self, logits1, logits2, num_samples_in_batch):
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
        jsd_loss = self.jsd_loss_fn(probs1, probs2)
        return jsd_loss.reshape(num_samples_in_batch, self.config['sequence_length'])
    
    def _compute_topk_jsd_loss(self, logits1, logits2, num_samples_in_batch):
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
        topk_jsd_loss = self.topk_jsd_loss_fn(probs1, probs2)
        return topk_jsd_loss.reshape(num_samples_in_batch, self.config['sequence_length'])

def main():
    for dataset in ['apollo-pile']:
        config = {
            'dataset': dataset,
            'models': ['baseline', 'finetuned', 'noLN'],
            'num_samples': 10000,
            'sequence_length': 512,
            'batch_size': 10,
            'save_text': False,
            'folder': '/workspace/removing-layer-norm/mech_interp/data/inference_logs/',
        }
        config['folder'] = os.path.join(config['folder'], f"dataset_{config['dataset']}_samples_{config['num_samples']}_seqlen_{config['sequence_length']}")
        
        # Setup output path
        os.makedirs(config['folder'], exist_ok=True)
        config['output_file'] = f"{config['folder']}/inference_results.parquet"
        
        # Run inference
        runner = InferenceRunner(config)
        runner.run_inference()

if __name__ == "__main__":
    main()

