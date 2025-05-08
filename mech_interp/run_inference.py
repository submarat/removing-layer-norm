from dataclasses import dataclass, field
from typing import List, Optional, Union
import os
from pathlib import Path
import multiprocessing
import itertools
import torch
import torch.nn.functional as F
from tqdm import tqdm

from load_models import ModelFactory
from load_dataset import DataLoader
from format_results import FormatInference 
from metrics import JSDivergence


@dataclass
class InferenceConfig:
    """Configuration for model inference runs."""
    dataset: str
    models: List[str]
    num_samples: int
    max_sequence_length: int
    batch_size: int
    model_dir: str = '/workspace/removing-layer-norm/mech_interp/models'
    prepend_bos: bool = False
    save_text: bool = False
    folder: Union[str, Path] = field(default="/workspace/removing-layer-norm/mech_interp/inference_logs/")
    output_file: Optional[str] = None
    num_threads: Optional[int] = None

    
    def __post_init__(self):
        # Set up derived paths after initialization to correctly assign self.output_file
        self.folder = os.path.join(
            self.folder, 
            f"dataset_{self.dataset}_samples_{self.num_samples}_" + \
            f"seqlen_{self.max_sequence_length}_prepend_{self.prepend_bos}"
        )
        
        # Ensure the directory exists
        os.makedirs(self.folder, exist_ok=True)
        
        # Set the output file path
        self.output_file = f"{self.folder}/inference_results.parquet"

        if self.num_threads is None:
            self.num_threads = min(32, multiprocessing.cpu_count() * 2)


class InferenceRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize various managers
        self.model_manager = ModelFactory(
                model_names=self.config.models,
                model_dir=self.config.model_dir,
                device=self.device)
        self.dataloader = DataLoader(
                dataset_name=self.config.dataset,
                batch_size=self.config.batch_size,
                max_context=self.config.max_sequence_length,
                num_samples=self.config.num_samples,
                prepend_bos=self.config.prepend_bos
                )
        self.results_formatter = FormatInference(
                self.config.output_file,
                save_text=self.config.save_text,
                num_threads=self.config.num_threads

                )
        self.JSD = JSDivergence()


    def _compute_metrics_from_logits(self, logits, targets):
        """
        Compute all metrics (probs, CE loss, entropy) from logits in a single pass.
        
        Args:
            logits: Model logits with shape [batch_size, sequence_length, vocab_size]
            targets: Optional target tokens with shape [batch_size, sequence_length]
            
        Returns:
            dict: Dictionary containing all computed metrics
                - 'probs': Softmax probabilities [batch_size, sequence_length-1, vocab_size]
                - 'ce_loss': Cross-entropy loss [batch_size, sequence_length-1]
                - 'entropy': Entropy of probability distribution [batch_size, sequence_length-1]
        """        
        # Get sequence-aligned logits (skip last position)
        seq_logits = logits[:, :-1, :]
        
        # Apply softmax to get probabilities (only once!)
        probs = F.softmax(seq_logits, dim=-1)
    
        # Find maximum probabilities and corresponding tokens
        max_probs, max_tokens = torch.max(probs, dim=-1)
        
        # Initialize result dictionary
        results = {
            'probs': probs,
            'max_prob': max_probs,
            'max_token': max_tokens
        }
        
        # Calculate entropy: -sum(p * log(p))
        epsilon = 1e-10  # Small epsilon to avoid log(0)
        log_probs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        results['entropy'] = entropy
        
        # Calculate cross-entropy loss
        # Target tokens - shape: [batch_size, sequence_length-1]
        target_tokens = targets[:, 1:]
        
        # Use log_probs we already computed
        ce_loss = -torch.gather(
            log_probs,
            dim=-1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1).contiguous()
        
        results['ce_loss'] = ce_loss
        
        return results        


    def run_inference(self):
        # Calculate total number of batches
        total_batches = (self.config.num_samples + self.config.batch_size - 1) // self.config.batch_size
        batched_data = self.dataloader.create_dataloader()

        # Run inference
        with torch.no_grad():  # Ensure no gradients are computed, this scope applies to all within
            for batch_idx, batch in enumerate(tqdm(batched_data, total=total_batches, desc="Processing batches")):
                batch = batch.to(self.device)
                self._process_batch(batch_idx, batch)
                
                # Clear GPU cache periodically to avoid OOM issues
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

        # Save all data to parquet file
        self.results_formatter.save_to_parquet()

    def _process_batch(self, batch_idx, batch):
        # Compute logits for all models in one pass
        logits = {name: model(batch) for name, model in self.model_manager.models.items()}

        # Compute metrics
        model_metrics = {}
        for model_name, model_logits in logits.items():
            model_metrics[model_name] = self._compute_metrics_from_logits(model_logits, batch)

        # Extract metrics for parquet storage
        ce_losses = {name: metrics['ce_loss'].cpu().numpy() for name, metrics in model_metrics.items()}
        entropies = {name: metrics['entropy'].cpu().numpy() for name, metrics in model_metrics.items()}
        max_probs = {name: metrics['max_prob'].cpu().numpy() for name, metrics in model_metrics.items()}
        max_tokens = {name: metrics['max_token'].cpu().numpy() for name, metrics in model_metrics.items()}

        # Compute JSD using pre-computed probabilities
        jsd_losses = {}
        for model1, model2 in self.model_manager.model_pairs:
            pair_name = f'{model1}_vs_{model2}'
            jsd = self.JSD(
                model_metrics[model1]['probs'],
                model_metrics[model2]['probs']
            )
            jsd_losses[pair_name] = jsd.cpu().numpy()

        # Add batch data to parquet manager
        self.results_formatter.add_batch_data(
            batch_idx * self.config.batch_size, # To keep track of original sequence indices
            batch.cpu().numpy(),
            ce_losses,
            jsd_losses,
            entropies,
            max_probs,
            max_tokens
        )


if __name__ == "__main__":
    # Create a config for the Apollo Pile dataset
    config = InferenceConfig(
        dataset="luca-pile",
        models=["baseline", "finetuned", "noLN"],
        num_samples=1000,
        max_sequence_length=512,
        batch_size=10,
        prepend_bos=False
    )

    runner = InferenceRunner(config)
    runner.run_inference()
