from typing import List, Optional, Dict

import os
import sys
import numpy as np
import torch as t
import einops
from tqdm import tqdm
from transformer_lens import utils
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


        
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
        self.num_heads = 12
        self.num_layers = 12
        
        self.device = utils.get_device()
        self.model_results = {model: {} for model in model_types}
        
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
            fold_ln=self.fold_ln
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
        
        result_absolute = []
        result_relative = []
        with t.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Running DLA vs direct ablation for {model_type}"):
                batch = batch.to(self.device)
                _, cache = model.run_with_cache(batch)
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
                overestimation_ratio = (1. - DLA_heads / (direct_ablation + 1e-10)) * 100.
                result_absolute.append(abs_delta)
                result_relative.append(overestimation_ratio)
            
            
            result_absolute = t.cat(result_absolute, dim=1).cpu().detach().numpy() # [heads, samples]
            result_relative = t.cat(result_relative, dim=1).cpu().detach().numpy() # [heads, samples]
                
            self.model_results[model_type]['absolute'] = result_absolute
            self.model_results[model_type]['relative'] = result_relative
    
    def run_analysis(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Run analysis for all model types.
        
        Returns:
            Dict containing results for all models
        """
        for model_type in self.model_types:
            self.run_single_model_analysis(model_type)
        
        return self.model_results

    def plot_histograms(self, save_dir: Optional[str] = None, outlier_range=(5, 95)) -> None:
        """
        Plot histograms of the flattened results with interval bounds based on quartiles.
        Handles outlier removal and uses a log y-scale.
        
        Args:
            save_dir: Directory to save the plots
            outlier_range: Tuple of (low, high) percentiles to filter outliers per model
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
            
            plt.figure(figsize=(10, 6))
            
            # Calculate statistics and plot each model
            for model in model_order:
                # Get flattened data for this model
                flattened = self.model_results[model][metric].flatten()
                
                # Calculate statistics on full data
                q25, median, q75 = np.percentile(flattened, [25, 50, 75])
                
                # Apply percentile filtering
                low_cut, high_cut = np.percentile(flattened, outlier_range)
                
                # For relative metric, make symmetric around 0 if applicable
                if metric == 'relative' and low_cut < 0 and high_cut > 0:
                    abs_max = max(abs(low_cut), abs(high_cut))
                    low_cut, high_cut = -abs_max, abs_max
                
                # Filter data
                filtered_data = flattened[(flattened >= low_cut) & (flattened <= high_cut)]
                
                # Create bins for this model
                model_bins = np.linspace(low_cut, high_cut, 50)
                
                # Create label with interval bounds
                if metric == 'relative':
                    label_str = f"{model}: {median:.2f}% [{q25:.2f}, {q75:.2f}]%"
                else:
                    label_str = f"{model}: {median:.3f} [{q25:.3f}, {q75:.3f}]"
                
                # Plot histogram
                plt.hist(
                    filtered_data, 
                    bins=model_bins,
                    histtype='step',
                    linewidth=2,
                    color=model_colors[model],
                    label=label_str
                )
            
            # Add a title and labels
            if metric == 'relative':
                x_label = 'Relative difference (%)'
            else:
                x_label = 'Absolute difference'
            
            plt.xlabel(x_label, fontsize=14)
            plt.ylabel('Counts (log scale)', fontsize=14)
            
            # Set log scale for y-axis
            plt.yscale('log')
            
            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.7)
            # Add a legend
            plt.legend()
           
            if save_dir: 
                plt.savefig(f"{save_dir}/{metric}_histogram.png", dpi=300)
            plt.close()    
    
    def plot_heatmaps(self, save_dir: Optional[str] = None) -> None:
        """
        Create simple Plotly heatmaps showing the median difference between DLA and Direct Effect
        for each attention head across models, with a shared global colormap.
        
        The heatmap will display attention heads in a grid where:
        - Each row represents a layer
        - Each column represents a head within that layer
        
        Args:
            save_dir: Directory to save the plots
        """
        # Import plotly for heatmaps
        for metric in ['absolute', 'relative']:
            # The order of models to display
            model_order = self.model_types
            
            # Compute per-head median values for each model
            median_per_head = {}
            for model in model_order:
                # Calculate median across samples for each head
                median_per_head[model] = np.median(self.model_results[model][metric], axis=1)
                # Ensure we have the expected number of heads
                if len(median_per_head[model]) == self.num_layers * self.num_heads:
                    median_per_head[model] = median_per_head[model].reshape(self.num_layers, self.num_heads)
                else:
                    # If there's a mismatch, just use what we have
                    print(f"Warning: Expected {n_layers * n_heads} heads for {model}, but got {len(median_per_head[model])}")
            
            # Determine global colorscale range
            all_values = np.concatenate([median_per_head[model].flatten() for model in model_order])
            vmin, vmax = np.min(all_values), np.max(all_values)
            
            # For relative metric, center the colormap at 0
            if metric == 'relative':
                abs_max = max(abs(vmin), abs(vmax))
                vmin, vmax = -abs_max, abs_max
            
            # Create a subplot with one column per model
            fig = make_subplots(
                rows=1, 
                cols=len(model_order),
                subplot_titles=model_order,
                horizontal_spacing=0.03
            )
            
            # Add heatmaps for each model
            for i, model in enumerate(model_order):
                fig.add_trace(
                    go.Heatmap(
                        z=median_per_head[model],
                        colorscale="RdBu_r" if metric == 'relative' else "Viridis",
                        zmid=0 if metric == 'relative' else None,
                        zmin=vmin,
                        zmax=vmax,
                        showscale=i == len(model_order) - 1,  # Only show colorbar for last model
                    ),
                    row=1, col=i+1
                )
                
                # Set axis titles and ticks for each subplot
                fig.update_xaxes(
                    title_text="Attention Head" if i == 1 else "",  # Only label the middle plot for x-axis
                    row=1, col=i+1
                )
                
                if i == 0:  # Only label the first plot for y-axis
                    fig.update_yaxes(
                        title_text="Layer",
                        row=1, col=i+1
                    )
                else:
                    fig.update_yaxes(showticklabels=False, row=1, col=i+1)
            
            # Update layout
            title = f"Per-head {metric.capitalize()} difference between DLA and Direct Effect"
            
            fig.update_layout(
                title=title,
                height=400,
                width=200 * len(model_order) + 100,  # Width based on number of models
                coloraxis=dict(colorscale="RdBu_r" if metric == 'relative' else "Viridis"),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Also save as static image
            if save_dir:
                fig.write_image(f"{save_dir}/{metric}_per_head_heatmap.png", scale=2)
            
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
                # Get the data (now shaped [heads, samples])
                values = self.model_results[model][metric]
                
                # Get global statistics by flattening
                flattened_values = values.flatten()
                q25, median, q75 = np.percentile(flattened_values, [25, 50, 75])
                
                # Get per-head statistics
                stats[model][metric] = {
                    # Global statistics
                    'median': float(median),
                    'lower_interval': float(q25),
                    'upper_interval': float(q75),
                }
        return stats

if __name__ == '__main__':
    # Example usage
    analyzer = AttentionAttributionAnalysis(
        model_types=['baseline', 'finetuned', 'noLN'],
        model_dir="../models",
        model_size="small"
    )
    
    # Run the analysis
    results = analyzer.run_analysis()
    
    # Plot the results
    analyzer.plot_histograms(save_dir='figures')
    analyzer.plot_heatmaps(save_dir='figures')
    
    # Get summary statistics
    stats = analyzer.get_summary_statistics()
    print("Summary Statistics:")
    for model, model_stats in stats.items():
        print(f"\n{model}:")
        for metric, values in model_stats.items():
            if metric == 'relative':
                # For relative metric, keep the negative sign
                print(f"  {metric}: {values['median']:.2f}% [{values['lower_interval']:.2f}, {values['upper_interval']:.2f}]%")
            else:
                # For absolute metric, remove the negative sign
                print(f"  {metric}: {values['median']:.3f} [{values['lower_interval']:.3f}, {values['upper_interval']:.3f}]")