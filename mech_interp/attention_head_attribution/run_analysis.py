# %%
from typing import List, Optional, Dict
import os
import sys
import numpy as np
import torch as t
import einops
from tqdm import tqdm
from transformer_lens import utils
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
        if self.model_size == 'small':
            self.num_heads = 12
            self.num_layers = 12
        elif self.model_size == 'medium':
            self.num_heads = 16
            self.num_layers = 24
        
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
        
        all_DLA_heads = []
        all_direct_ablation = []
        
        with t.no_grad():
            for batch in tqdm(self.dataloader, desc=f"Running analysis for {model_type}"):
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
                
                all_DLA_heads.append(DLA_heads)
                all_direct_ablation.append(direct_ablation)
            
            # Concatenate all batches
            all_DLA_heads = t.cat(all_DLA_heads, dim=1)  # [heads, all_samples]
            all_direct_ablation = t.cat(all_direct_ablation, dim=1)  # [heads, all_samples]
            
            # Calculate absolute difference
            abs_diff = (all_DLA_heads - all_direct_ablation).abs()
            
            # Store raw results for bootstrap calculations later
            self.model_results[model_type]['DLA_heads'] = all_DLA_heads.cpu().numpy()
            self.model_results[model_type]['direct_ablation'] = all_direct_ablation.cpu().numpy()
            
            # Calculate per-head NMAE (vectorized)
            mean_abs_diff_per_head = abs_diff.mean(dim=1)  # [heads]
            mean_direct_abs_per_head = all_direct_ablation.abs().mean(dim=1)  # [heads]
            
            # Apply small epsilon to avoid division by zero
            epsilon = 1e-10
            per_head_nmae = (mean_abs_diff_per_head / (mean_direct_abs_per_head + epsilon)) * 100
            
            # Convert to numpy for storage
            per_head_nmae_np = per_head_nmae.cpu().numpy()
            
            # Store absolute difference and per-head NMAE for plotting
            self.model_results[model_type]['absolute'] = abs_diff.cpu().numpy()
            self.model_results[model_type]['per_head_nmae'] = per_head_nmae_np
            
            # Calculate global NMAE (across all heads and samples)
            global_mean_abs_error = abs_diff.mean().item()
            global_mean_direct_ablation = all_direct_ablation.abs().mean().item()
            global_nmae = (global_mean_abs_error / (global_mean_direct_ablation + epsilon)) * 100
            
            # Store global NMAE
            self.model_results[model_type]['global_nmae'] = global_nmae
    
    def run_analysis(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run analysis for all model types.
        
        Returns:
            Dict containing results for all models
        """
        for model_type in self.model_types:
            self.run_single_model_analysis(model_type)
        
        return self.model_results
    
    def plot_heatmaps(self, save_dir: Optional[str] = None) -> None:
        """
        Create Plotly heatmaps showing:
        1. Absolute difference between DLA and Direct Effect
        2. Per-head NMAE (Normalized Mean Absolute Error)
        
        Args:
            save_dir: Directory to save the plots
        """
        # Create heatmaps for both absolute difference and NMAE
        for metric_name, metric_title in [
            ('absolute', 'Absolute Difference'),
            ('per_head_nmae', 'Per-head NMAE (%)')
        ]:
            # The order of models to display
            model_order = self.model_types
            
            # Compute per-head median values for each model
            median_per_head = {}
            for model in model_order:
                if metric_name == 'absolute':
                    # For absolute difference, calculate median across samples for each head
                    data = self.model_results[model][metric_name]  # [heads, samples]
                    median_per_head[model] = np.median(data, axis=1)
                else:
                    # For NMAE, it's already per-head
                    median_per_head[model] = self.model_results[model][metric_name]
                
                # Reshape to grid format
                if len(median_per_head[model]) == self.num_layers * self.num_heads:
                    median_per_head[model] = median_per_head[model].reshape(self.num_layers, self.num_heads)
                else:
                    print(f"Warning: Expected {self.num_layers * self.num_heads} heads for {model}, but got {len(median_per_head[model])}")
            
            # Determine global colorscale range
            all_values = np.concatenate([median_per_head[model].flatten() for model in model_order])
            
            # Remove potential infinity or very large values
            filtered_values = all_values[np.isfinite(all_values)]
            if len(filtered_values) > 0:
                vmin, vmax = np.min(filtered_values), np.max(filtered_values)
            else:
                vmin, vmax = 0, 1  # Fallback if all values are infinite
            
            # For NMAE, ensure a minimum range for visibility
            if metric_name == 'per_head_nmae' and vmax - vmin < 5:
                vmax = vmin + 5
            
            # Create a subplot with one column per model
            fig = make_subplots(
                rows=1, 
                cols=len(model_order),
                subplot_titles=model_order,
                horizontal_spacing=0.03
            )
            
            # Add heatmaps for each model
            for i, model in enumerate(model_order):
                data = median_per_head[model]
                
                # Handle potential infinity values for visualization
                if np.any(np.isinf(data)):
                    data = np.nan_to_num(data, nan=0, posinf=vmax*2, neginf=vmin*2)
                
                fig.add_trace(
                    go.Heatmap(
                        z=data,
                        colorscale='Reds',
                        zmin=0,
                        zmax=vmax,
                        showscale=i == len(model_order) - 1,
                        colorbar=dict(
                            title=dict(
                                text=metric_title,
                                side="right",
                                font=dict(size=14)
                            )
                        )
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
            title = f"Per-head {metric_title} between DLA and Direct Effect"
            
            fig.update_layout(
                title=title,
                height=400,
                width=200 * len(model_order) + 100,  # Width based on number of models
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Save as static image
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.write_image(f"{save_dir}/{metric_name}_per_head_heatmap.png", scale=2)
    
    def get_global_nmae_with_bootstrap_ci(self, model_type: str, bootstrap_samples: int = 1000, confidence_level: float = 0.95):
        """
        Calculate global NMAE with bootstrap confidence intervals for a model.
        
        Args:
            model_type: The model type to analyze
            bootstrap_samples: Number of bootstrap resamples
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (nmae, lower_bound, upper_bound)
        """
        # Get stored data
        DLA_heads = t.tensor(self.model_results[model_type]['DLA_heads'])
        direct_ablation = t.tensor(self.model_results[model_type]['direct_ablation'])
        
        # Original global NMAE
        abs_diff = (DLA_heads - direct_ablation).abs()
        mean_abs_error = abs_diff.mean().item()
        mean_direct_ablation = direct_ablation.abs().mean().item()
        epsilon = 1e-10  # Small value to prevent division by zero
        original_nmae = (mean_abs_error / (mean_direct_ablation + epsilon)) * 100
        
        # Bootstrap for confidence intervals
        n_heads, n_samples = DLA_heads.shape
        nmae_bootstraps = []
        
        for _ in range(bootstrap_samples):
            # Sample heads and samples with replacement
            head_indices = np.random.choice(n_heads, n_heads, replace=True)
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Get bootstrap sample
            bootstrap_DLA = DLA_heads[head_indices][:, sample_indices]
            bootstrap_direct = direct_ablation[head_indices][:, sample_indices]
            
            # Calculate bootstrap NMAE
            bootstrap_abs_diff = (bootstrap_DLA - bootstrap_direct).abs()
            bootstrap_mean_abs_error = bootstrap_abs_diff.mean().item()
            bootstrap_mean_direct_ablation = bootstrap_direct.abs().mean().item()
            bootstrap_nmae = (bootstrap_mean_abs_error / (bootstrap_mean_direct_ablation + epsilon)) * 100
            
            nmae_bootstraps.append(bootstrap_nmae)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower_bound, upper_bound = np.percentile(nmae_bootstraps, [lower_percentile, upper_percentile])
        
        return original_nmae, lower_bound, upper_bound
        
    def print_summary_statistics(self):
        """
        Print simplified summary statistics for all models.
        """
        print("\nNormalized Mean Absolute Error (NMAE) Summary:")
        print("-" * 50)
        print(f"{'Model':<20} {'NMAE':<8} {'95% CI':<20}")
        print("-" * 50)
        
        for model_type in self.model_types:
            nmae, lower, upper = self.get_global_nmae_with_bootstrap_ci(model_type)
            print(f"{model_type:<20} {nmae:<8.2f}% [{lower:.2f}%, {upper:.2f}%]")

if __name__ == '__main__':
    # Example usage
    model_str = 'small'
    analyzer = AttentionAttributionAnalysis(
        model_types=['baseline', 'finetuned', 'noLN'],
        model_dir="../models",
        model_size=model_str,
    )
    
    # Run the analysis
    results = analyzer.run_analysis()
    
    # Plot the heatmaps
    analyzer.plot_heatmaps(save_dir=f'figures/{model_str}')
    
    # Print summary statistics
    analyzer.print_summary_statistics()