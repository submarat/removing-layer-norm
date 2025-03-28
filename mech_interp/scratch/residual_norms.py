import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from typing import Dict, List, Optional, Union, Literal, Tuple
from transformer_lens import HookedTransformer

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_models import ModelFactory


class ResidualNorms:
    def __init__(self,
                 data_path: str,
                 model_dir: str = "../models",
                 num_samples: Optional[int] = None,
                 random_seed: int = 42,
                 last_token_only: bool = False,
                 agg: bool = False,
                 device: Optional[str] = None):

        """
        Initialize the analyzer with data and prepare models.
        
        Args:
            data_path: Path to the parquet file containing text data
            model_dir: Directory containing the model files
        """
        # Load data from parquet file
        self.df = pd.read_parquet(data_path)
        self.last_token_only = last_token_only
        if agg:
            self.get_aggregate_metrics()

        if num_samples:
            # Subsample dataframe for analysis
            np.random.seed(random_seed)
            sample_indices = np.random.choice(len(self.df), size=num_samples, replace=False)
            self.df = self.df.iloc[sample_indices].reset_index(drop=True)
            print(f"Sampled {len(self.df)} examples for analysis.")
        
        # Initialize model factory
        self.model_names = ['baseline', 'finetuned', 'noLN']
        self.model_factory = ModelFactory(self.model_names, model_dir=model_dir)

        # Results will be stored here after running analysis
        self.results = None
        
        # Set up color schemes for consistent visualization
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.model_colors = {
            'baseline': self.colors[0],
            'finetuned': self.colors[1],
            'noLN': self.colors[2]
        }  

    def get_aggregate_metrics(self):
        """
        Aggregate metrics accross all subsequences to get overall average results
        
        Returns:
        --------
        Aggregated df
        """
        # Define the metrics columns to aggregate
        metric_columns = [
            'ce_baseline', 'ce_finetuned', 'ce_noLN',
            'ce_diff_baseline_vs_finetuned', 'jsd_baseline_vs_finetuned', 'topk_jsd_baseline_vs_finetuned',
            'ce_diff_baseline_vs_noLN', 'jsd_baseline_vs_noLN', 'topk_jsd_baseline_vs_noLN',
            'ce_diff_finetuned_vs_noLN', 'jsd_finetuned_vs_noLN', 'topk_jsd_finetuned_vs_noLN'
        ]
        
        # Create a temporary function to get the row with the longest sequence for each group
        def get_longest_sequence_info(group):
            # Get the index of the row with the maximum sequence_length
            idx_max_length = group['sequence_length'].idxmax()
            # Return a Series with the relevant information from that row
            return pd.Series({
                'full_sequence': group.loc[idx_max_length, 'full_sequence'],
                'last_token': group.loc[idx_max_length, 'last_token'],
                'next_token': group.loc[idx_max_length, 'next_token'],
                'sequence_length': group.loc[idx_max_length, 'sequence_length']
            })
        # Group by original_idx and calculate mean for all metrics
        # For full_sequence, we'll take the longest one
        aggregated_metrics = self.df.groupby('original_idx')[metric_columns].mean()
        
        # Find the longest full_sequence and corresponding tokens for each original_idx
        longest_sequences_info = self.df.groupby('original_idx').apply(get_longest_sequence_info)
        
        # Combine the aggregated metrics with the longest sequences and their tokens
        aggregated_df = aggregated_metrics.join(longest_sequences_info).reset_index()
        
        print(f"Original shape: {self.df.shape}")
        print(f"After aggregation: {aggregated_df.shape}")
        
        self.df = aggregated_df

    
    def get_hook_names(self, model):
        """Get appropriate hook names for a given model."""
        hook_names = ['hook_embed']  # Raw embeddings
        
        # Add block-specific hooks
        n_layers = len(model.blocks)
        for i in range(n_layers):
            hook_names.append(f'blocks.{i}.hook_resid_mid')
            hook_names.append(f'blocks.{i}.hook_resid_post')
        
        return hook_names
   

    def run_analysis(self, batch_size: int = 8):
        """
        Run the residual stream analysis on all models.
        
        Args:
            batch_size: Number of examples to process at once
        """
        results = {}
        
        # Process data in batches
        for i in range(0, len(self.df), batch_size):
            batch_df = self.df.iloc[i:i+batch_size]
            
            # Get token sequences from the dataframe
            sequences = batch_df['full_sequence'].tolist()
            
            # Convert sequences to tensors
            inputs = torch.tensor(sequences).to(next(self.model_factory.models[self.model_names[0]].parameters()).device)
            
            # Process each model
            for model_name in self.model_names:
                if model_name not in results:
                    results[model_name] = {}
                
                model = self.model_factory.models[model_name]
                hook_names = self.get_hook_names(model)
                
                # Initialize results containers if needed
                for hook_name in hook_names:
                    if hook_name not in results[model_name]:
                        results[model_name][hook_name] = []
                
                # Define the hook function to capture norms for the entire batch
                def capture_norms(act, hook):
                    if self.last_token_only:
                        act = act[:, -1:, :] # -1: keeps the dims
                    l2_norm = torch.norm(act, p=2, dim=-1)
                    hook_name = hook.name
                    results[model_name][hook_name].append(l2_norm.detach().cpu())
                    return act
                
                # Run the model with hooks for the entire batch
                with torch.no_grad():
                    model.run_with_hooks(
                        inputs,
                        fwd_hooks=[(name, capture_norms) for name in hook_names]
                    )
        # Store results
        self.results = results

    def plot_norms(self, save_path=None):
        """
        Plot layer norms across layers for each model, separating first token position and other positions.
        
        Args:
            save_path: Path to save the plot, or None to display
        """
        if self.results is None:
            print("No results available. Please run analysis first.")
            return
        
        # Create a 1x2 grid: columns for first token and other tokens
        if self.last_token_only:
            fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Define simplified hook labels for x-axis
        hook_mapping = {
            'hook_embed': 'Embed',
            'hook_pos_embed': 'Embed + Pos'
        }
        
        # Add transformer block mappings
        for i in range(12):  # Assuming 12 layers for GPT-2 small
            hook_mapping[f'blocks.{i}.hook_resid_mid'] = f'Resid_mid_{i}'
            hook_mapping[f'blocks.{i}.hook_resid_post'] = f'Resid_post_{i}'
        
        # Extract all common hook names
        first_model_name = self.model_names[0]
        common_hooks = list(self.results[first_model_name].keys())
        
        # Sort hooks to ensure logical progression
        common_hooks.sort(key=lambda x: (
            0 if 'embed' in x else 
            1 if 'blocks' in x else 
            2,
            int(x.split('.')[1]) if 'blocks' in x and '.' in x else 0,
            0 if 'resid_mid' in x else 1 if 'resid_post' in x else 0
        ))
        
        # Get simplified labels for plotting
        x_labels = []
        for hook in common_hooks:
            for key, value in hook_mapping.items():
                if key in hook:
                    x_labels.append(value)
                    break

        if self.last_token_only:
            positions = [('first', 'Last Token Norm')]
        else:
            positions = [('first', 'First Token Norm'), ('others', 'Other Token Norms')]

        # First, calculate global min and max values for consistent y-axis scaling
        global_min = float('inf')
        global_max = float('-inf')
        
        for pos_idx, (pos_key, pos_title) in enumerate(positions):
            if self.last_token_only:
                ax = axes
            else:
                ax = axes[pos_idx]
            
            for model_name in self.model_names:
                model_data = self.results[model_name]
                
                # Collect mean and std for each hook point
                means = []
                stds = []
                
                for hook in common_hooks:
                    if hook in model_data:
                        # Concatenate all batches
                        all_values = torch.cat(model_data[hook], dim=0)
                        
                        # Split by position
                        if pos_key == 'first':
                            # First token position only
                            position_values = all_values[:, 0]
                        else:
                            # All other positions
                            position_values = all_values[:, 1:].reshape(-1)
                        
                        # Compute mean and std
                        mean = position_values.mean().item()
                        std = position_values.std().item()
                        
                        means.append(mean)
                        stds.append(std)

                        # Update global min/max
                        global_min = min(global_min, mean - std)
                        global_max = max(global_max, mean + std)
                
                # Plot with shaded uncertainty region
                x = np.arange(len(means))
                ax.plot(x, means, 'o-', label=model_name, color=self.model_colors[model_name])
                ax.fill_between(x, 
                               [m - s for m, s in zip(means, stds)],
                               [m + s for m, s in zip(means, stds)],
                               alpha=0.2, color=self.model_colors[model_name])
            
            # Set labels and title
            ax.set_xlabel('Layer')
            ax.set_ylabel('L2 Norn')
            ax.set_title(f'L2 Norm - {pos_title}')
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            # Set consistent y-axis limits for both subplots
            ax.set_ylim(global_min * 0.9, global_max * 1.1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        else:
            plt.show()
        
        return fig

    def create_layer_norm_video(self, output_path, fps=1):
        """
        Create a video of layer norms across token positions for each model.
        Each frame shows a different layer/hook point.
        
        Args:
            output_path: Path to save the video
            fps: Frames per second for the video
        """
        if self.last_token_only:
            return f"Cannot analyse positional residual stream variations, last_token_only=True"

        # Get all hook names in order
        model = self.model_factory.models[self.model_names[0]]
        hook_names = self.get_hook_names(model)
        
        # Set up the figure with a single plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Initialize lines and fill collections for each model
        lines = []
        fills = []
        max_val = 0  # For scaling
        min_val = float('inf')  # For scaling
        
        # Prepare data and scale for each model
        for model_name in self.model_names:
            model_data = self.results[model_name]
            
            for hook in hook_names:
                if hook in model_data:
                    # Get sequence length from first example
                    seq_len = model_data[hook][0].shape[1]
                    
                    # Compute mean and std across all sequences
                    all_values = torch.cat(model_data[hook], dim=0)
                    mean_values = all_values.mean(dim=0).numpy()
                    std_values = all_values.std(dim=0).numpy()
                    
                    # Update min/max for consistent scaling
                    max_val = max(max_val, (mean_values + std_values).max())
                    min_val = min(min_val, (mean_values - std_values).min())
            
            # Create initial line with label
            line, = ax.plot([], [], '-', linewidth=2, color=self.model_colors[model_name], label=model_name)
            lines.append(line)
            
            # Create initial fill for standard deviation
            fill = ax.fill_between([], [], [], alpha=0.2, color=self.model_colors[model_name])
            fills.append(fill)
        
        # Set up axes with padding for std fills
        ax.set_xlim(0, 10)
        ax.set_ylim(min_val * 0.7, max_val * 1.3)
        ax.set_yscale('log')
        ax.set_xlabel("Token Position")
        ax.set_ylabel("L2 Norm")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Make a simplified hook name mapping for titles
        hook_titles = {}
        for hook in hook_names:
            if 'hook_embed' in hook:
                hook_titles[hook] = 'Embeddings'
            elif 'blocks' in hook:
                parts = hook.split('.')
                layer_num = parts[1]
                if 'resid_mid' in hook:
                    hook_titles[hook] = f'Layer {layer_num} (Mid)'
                else:
                    hook_titles[hook] = f'Layer {layer_num} (Post)'
            elif 'ln_final' in hook:
                hook_titles[hook] = 'Final Layer Norm'
            else:
                hook_titles[hook] = hook
        
        # Animation function
        def init():
            for line in lines:
                line.set_data([], [])
            for fill in fills:
                fill.set_visible(False)
            return lines + fills
        
        def animate(frame):
            hook = hook_names[frame]
            ax.set_title(f"Residual L2 Norms Across Token Positions - {hook_titles[hook]}")
            
            for i, model_name in enumerate(self.model_names):
                if hook in self.results[model_name]:
                    all_values = torch.cat(self.results[model_name][hook], dim=0)
                    mean_values = all_values.mean(dim=0).numpy()
                    std_values = all_values.std(dim=0).numpy()
                    x = np.arange(len(mean_values))
                    
                    # Update line
                    lines[i].set_data(x, mean_values)
                    
                    # Update fill
                    fills[i].remove()
                    fills[i] = ax.fill_between(
                        x, 
                        mean_values - std_values, 
                        mean_values + std_values, 
                        alpha=0.2, 
                        color=self.model_colors[model_name]
                    )
            
            return lines + [fill for fill in fills if fill is not None]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, animate, frames=len(hook_names),
            init_func=init, blit=False, interval=1000/fps
        )
        
        # Save animation
        ani.save(output_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        
        print(f"Video saved to {output_path}")


if __name__ == '__main__':
    data_path = '/workspace/removing-layer-norm/mech_interp/inference_logs/dataset_apollo-pile_samples_5000_seqlen_512_prepend_False/inference_results.parquet'
    #data_path = 'divergences/convergent.parquet'
    # Initialize the analyzer with your dataset
    analyzer = ResidualNorms(
        data_path=data_path,  # Path to your parquet file
        model_dir="../models",                   # Directory with your models
        random_seed=42,                          # For reproducible sampling
        last_token_only=False,
        agg=True,
        #num_samples=20000
    )
    
    output_dir = 'norms'
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    
    # Run the analysis to collect norms at all hook points
    analyzer.run_analysis(batch_size=1)  # Process 8 examples at a time
    
    # Generate the layer norm plots
    fig = analyzer.plot_norms(os.path.join(output_dir, 'norms.png'))
    analyzer.create_layer_norm_video(os.path.join(output_dir, 'video.gif'))
