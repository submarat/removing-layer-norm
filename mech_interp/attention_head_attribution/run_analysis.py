from functools import partial
from typing import List, Optional, Union, Dict, Tuple, Any

import os
import sys
import numpy as np
import torch as t
import einops
from tqdm import tqdm
from transformer_lens import utils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add parent directory to path for custom imports
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_models import ModelFactory
from load_dataset import DataLoader


class AttentionAttributionAnalysis:
    def __init__(
        self,
        model_types: List[str] = ['baseline', 'finetuned', 'noLN'],
        model_dir: str = "../models",
        model_size: str = "small",
        center_unembed: bool = True,
        center_writing_weights: bool = True,
        fold_ln: bool = True,
        batch_size: int = 5,
        dataset_name: str = "luca-pile",
        max_context: int = 512,
        num_samples: int = 1000,
        prepend_bos: bool = False
    ):
        """
        Initialize the Attention Attribution Analysis class.
        
        Args:
            model_types: List of model types to analyze
            model_dir: Directory containing the models
            model_size: Size of the models to use
            center_unembed: Whether to center the unembedding weights
            center_writing_weights: Whether to center the writing weights
            fold_ln: Whether to fold layer normalization
            batch_size: Batch size for dataset loading
            dataset_name: Name of the dataset to use
            max_context: Maximum context length
            num_samples: Number of samples to use
            prepend_bos: Whether to prepend BOS token
        """
        self.model_types = model_types
        self.model_dir = model_dir
        self.model_size = model_size
        self.center_unembed = center_unembed
        self.center_writing_weights = center_writing_weights
        self.fold_ln = fold_ln
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.max_context = max_context
        self.num_samples = num_samples
        self.prepend_bos = prepend_bos
        
        self.device = utils.get_device()
        self.model_results = {model: {'absolute': [], 'relative': []} for model in model_types}
        
        # Load dataloader
        self._load_dataloader()
        
    def _load_dataloader(self) -> None:
        """Load the dataloader for analysis."""
        self.dataloader = DataLoader(
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            max_context=self.max_context,
            num_samples=self.num_samples,
            prepend_bos=self.prepend_bos
        ).create_dataloader()
        
    def _load_model(self, model_type: str):
        """Load a specific model type."""
        factory = ModelFactory(
            [model_type],
            model_dir=self.model_dir,
            model_size=self.model_size,
            center_unembed=self.center_unembed,
            center_writing_weights=self.center_writing_weights,
            fold_ln=self.fold_ln,
        )
        
        model = factory.models[model_type]
        model = model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        model.cfg.use_attn_result = True  # Enable storing of attention results
        return model
    
    @staticmethod
    def apply_layer_norm(input: t.Tensor) -> t.Tensor:
        """Apply layer normalization manually."""
        eps = 1e-5
        mean = input.mean(dim=-1, keepdim=True)
        var = ((input - mean) ** 2).mean(dim=-1, keepdim=True)
        norm = (input - mean) / t.sqrt(var + eps)
        
        return norm
    
    def run_single_model_analysis(self, model_type: str) -> None:
        """
        Run the analysis for a single model type.
        
        Args:
            model_type: The model type to analyze
        """
        model = self._load_model(model_type)
        model = model.to(self.device)
        
        with t.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Running DLA vs direct ablation for {model_type}"):
                batch = batch.to(self.device)
                original_logits, cache = model.run_with_cache(batch)
                target_tokens = batch[:, -1]
                
                # Get the unembedding vectors for our target tokens
                target_unembed_W = model.W_U[:, target_tokens].T  # Shape: [batch, d_model]
                target_unembed_b = model.b_U[target_tokens] # Shape [batch] 
                
                # Transform finally residual stream to correct output logit
                final_residual_stream = cache["resid_post", -1]
                final_token_residual_stream = final_residual_stream[:, -1, :]
                
                # Apply layer norm based on model type
                if model_type == 'noLN':
                    scaled_final_token_residual_stream = final_token_residual_stream
                else:
                    scaled_final_token_residual_stream = cache.apply_ln_to_stack(
                        final_token_residual_stream, 
                        layer=-1, 
                        pos_slice=-1
                    )
                    
                logits = einops.reduce(
                    scaled_final_token_residual_stream * target_unembed_W,
                    'batch d_model -> batch',
                    'sum'
                    ) + target_unembed_b
                
                # PART 1: Direct Linear Attribution (DLA) Analysis
                # Get each head's contribution to the residual stream
                per_head_residual, head_labels = cache.stack_head_results(
                    layer=-1, pos_slice=-1, return_labels=True
                )
                
                # Apply cached layer normalization if needed
                if model_type == 'noLN':
                    scaled_head_residual = per_head_residual 
                else:    
                    scaled_head_residual = cache.apply_ln_to_stack(
                        per_head_residual, 
                        layer=-1, 
                        pos_slice=-1
                    )
                
                # Calculate each head's contribution using DLA
                DLA_heads = einops.einsum(
                    scaled_head_residual, target_unembed_W,
                    'head batch d_model, batch d_model -> head batch'
                )
                
                # PART 2: Direct Ablation Effect
                # Subtract each head's contribution from final layer residual stream
                resid_heads_ablated = final_token_residual_stream - per_head_residual 
                
                # Apply layer norm based on model type
                if model_type == 'noLN':
                    scaled_resid_heads_ablated = resid_heads_ablated
                else:    
                    scaled_resid_heads_ablated = self.apply_layer_norm(resid_heads_ablated)
                
                # Calculate logits without head contributions
                logits_without_heads = einops.einsum(
                    scaled_resid_heads_ablated, target_unembed_W,
                    'head batch d_model, batch d_model -> head batch'
                ) + target_unembed_b
                
                # Calculate direct ablation effect
                direct_ablation = logits - logits_without_heads
                
                # Calculate metrics
                abs_delta = (DLA_heads - direct_ablation).abs()
                delta_per_ex = abs_delta.mean(dim=0).cpu().detach().tolist()
                
                overestimation_ratio = (1. - DLA_heads / (direct_ablation + 1e-10)) * 100.
                overestimation_ratio_median = overestimation_ratio.median(dim=0)
                overestimation_ratio_per_ex = overestimation_ratio_median.values.cpu().detach().tolist()
                
                # Store results
                self.model_results[model_type]['absolute'].extend(delta_per_ex)
                self.model_results[model_type]['relative'].extend(overestimation_ratio_per_ex)
    
    def run_analysis(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Run analysis for all model types.
        
        Returns:
            Dict containing results for all models
        """
        for model_type in self.model_types:
            self.run_single_model_analysis(model_type)
        
        return self.model_results
    
    def plot_results(self, save_dir: str = ".") -> None:
        """
        Plot the results of the analysis with asymmetric error bounds based on quartiles.
        
        Args:
            save_dir: Directory to save the plots
        """
        for metric in ['absolute', 'relative']:
            # The order of models to display
            model_order = self.model_types
            
            # Get default matplotlib colors
            colors = sns.color_palette("colorblind")
            
            # Set up explicit colors for each model using default matplotlib colors
            model_colors = {
                model: colors[i % len(colors)] for i, model in enumerate(model_order)
            }
            
            # Determine the range for all histograms
            all_values = np.concatenate([self.model_results[model][metric] for model in model_order])
            min_val, max_val = np.min(all_values), np.max(all_values)
            bins = np.linspace(min_val, max_val, 30)  # 30 bins across the range
            
            plt.figure(figsize=(10, 6))
            
            # Plot histograms for each model
            for model in model_order:
                model_metric = np.array(self.model_results[model][metric])
                
                # Calculate median and quartiles
                q25, median, q75 = np.percentile(model_metric, [25, 50, 75])
                
                # Calculate bounds based on quartiles
                lower_bound = median - q25
                upper_bound = q75 - median
                
                # Create label with asymmetric bounds
                if metric == 'relative':
                    # For relative metric, keep the negative sign
                    label_str = f"{model}: {median:.2f}% [{-lower_bound:.2f}, +{upper_bound:.2f}]%"
                else:
                    # For absolute metric, remove the negative sign
                    label_str = f"{model}: {median:.3f} [{-lower_bound:.3f}, +{upper_bound:.3f}]"

                plt.hist(
                    model_metric, 
                    bins=bins,
                    histtype='step',                # Step style for clear edges
                    linewidth=2,                    # Thicker lines for better visibility
                    color=model_colors[model],      # Use the model-specific color
                    label=label_str
                )
                
                # Show median
                plt.axvline(
                    median,
                    color=model_colors[model],
                    linestyle='--',
                    linewidth=1.5
                )
            
            # Add a title and labels
            if metric == 'relative':
                title = 'Relative difference between DLA and Direct Effect'
                x_label = 'Relative difference (%)'
            else:
                title = 'Absolute difference between DLA and Direct Effect'
                x_label = 'Absolute difference'
            
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel('Counts')
            
            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.7)
            # Add a legend
            plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{metric}_DLA_vs_DE.png", dpi=300)
            plt.close()
            
    def get_summary_statistics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get summary statistics for all models and metrics using quartiles (25-75 percentiles).
        
        Returns:
            Dict containing median, lower bound, and upper bound for each model and metric
        """
        stats = {}
        for model in self.model_types:
            stats[model] = {}
            for metric in ['absolute', 'relative']:
                values = np.array(self.model_results[model][metric])
                
                # Get median and quartiles
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                
                # Calculate bounds based on quartiles
                lower_bound = median - q25
                upper_bound = q75 - median
                
                stats[model][metric] = {
                    'median': float(median),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = AttentionAttributionAnalysis(
        model_types=['baseline', 'finetuned', 'noLN'],
        model_dir="../models",
        model_size="small"
    )
    
    # Run the analysis
    results = analyzer.run_analysis()
    
    # Plot the results
    analyzer.plot_results(save_dir='figures')
    
    # Get summary statistics
    stats = analyzer.get_summary_statistics()
    print("Summary Statistics:")
    for model, model_stats in stats.items():
        print(f"\n{model}:")
        for metric, values in model_stats.items():
            if metric == 'relative':
                # For relative metric, keep the negative sign
                print(f"  {metric}: {values['median']:.2f}% [{-values['lower_bound']:.2f}, +{values['upper_bound']:.2f}]%")
            else:
                # For absolute metric, remove the negative sign
                print(f"  {metric}: {values['median']:.3f} [{values['lower_bound']:.3f}, +{values['upper_bound']:.3f}]")
