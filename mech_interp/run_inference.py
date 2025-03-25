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
    jsd_topk: int = 50

    
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

        # Initialize various manag
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
        self.JSD_TopK = JSDivergence(topk=self.config.jsd_topk)


    def _compute_ce_loss(self, logits, targets):
        """
        Compute optimised verison of CE loss using native log probs and gather
        """
        # Apply log_softmax along vocabulary dimension (dim=-1)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

        # Target tokens - shape: [batch_size, sequence_length]
        target_tokens = targets[:, 1:]

        # Find probability for ground truth token. Using .contiguous() to ensure memory layout is optimal
        ce_loss = -torch.gather(
            log_probs,
            dim=-1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1).contiguous()

        return ce_loss

    def _compute_jsd_loss(self, logits_p, logits_q, topk=None):
        """
        Compute Jensen-Shannon Divergence between two sets of logits.
        """
        # Truncate to match the same sequence alignment as in CE loss
        logits_p = logits_p[:, :-1, :]
        logits_q = logits_q[:, :-1, :]
        if not topk:
            return self.JSD(logits_p, logits_q)
        else:
           return  self.JSD_TopK(logits_p, logits_q)


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

        # Compute CE loss
        ce_losses = {}
        for model_name, model_logits in logits.items():
            ce_loss = self._compute_ce_loss(model_logits, batch)
            ce_losses[model_name] = ce_loss.cpu().numpy()

        # Compute CE loss differences
        ce_diffs = {}
        for model1, model2 in self.model_manager.model_pairs:
            pair_name = f'{model1}_vs_{model2}'
            ce_diffs[pair_name] = ce_losses[model1] - ce_losses[model2]

        # Compute JSD
        jsd_losses = {}
        topk_jsd_losses = {}
        for model1, model2 in self.model_manager.model_pairs:
            pair_name = f'{model1}_vs_{model2}'
            jsd_losses[pair_name] = self._compute_jsd_loss(logits[model1], logits[model2]).cpu().numpy()
            topk_jsd_losses[pair_name] = self._compute_jsd_loss(
                    logits[model1], logits[model2], topk=self.config.jsd_topk).cpu().numpy()

        # Add batch data to parquet manager
        self.results_formatter.add_batch_data(
            batch_idx * self.config.batch_size, # To keep track of original sequence indices
            batch.cpu().numpy(),
            ce_losses,
            ce_diffs,
            jsd_losses,
            topk_jsd_losses,
        )


if __name__ == "__main__":
    # Create a config for the Apollo Pile dataset
    config = InferenceConfig(
        dataset="apollo-pile",
        models=["baseline", "finetuned", "noLN"],
        num_samples=5000,
        max_sequence_length=512,
        batch_size=10,
        prepend_bos=True
    )

    runner = InferenceRunner(config)
    runner.run_inference()
