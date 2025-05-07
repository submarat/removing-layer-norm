import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from typing import Dict, List, Optional, Union, Literal, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_models import ModelFactory


class ResidualNorms:
    def __init__(self,
                 data_paths: Union[str, Dict[str, str]],
                 model_dir: str = "../models",
                 num_samples: Optional[int] = None,
                 random_seed: int = 42,
                 last_token_only: bool = False,
                 device: Optional[str] = None):

        """
        Initialize the analyzer with data and prepare models.
        
        Args:
            data_paths: Either a single path to a parquet file or a dictionary mapping dataset names to paths
            model_dir: Directory containing the model files
            num_samples: Number of samples to use (optional)
            random_seed: Random seed for reproducibility
            last_token_only: Whether to analyze only the last token
            device: Device to use for computation (optional)
        """
        self.df = pd.read_parquet(data_paths)
        self.df = self.df[self.df.sequence_length == self.df.sequence_length.max()].reset_index(drop=True)
        
        if num_samples:
            # Subsample dataframe for analysis
            np.random.seed(random_seed)
            sample_indices = np.random.choice(len(self.df), size=num_samples, replace=False)
            self.df = self.df.iloc[sample_indices].reset_index(drop=True)
            print(f"Sampled {len(self.df)} examples for analysis.")
        
        self.last_token_only = last_token_only
        
        # Initialize model factory
        self.model_names = ['baseline', 'finetuned', 'noLN']
        self.model_factory = ModelFactory(self.model_names, model_dir=model_dir)

        # Set up color schemes for consistent visualization
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.model_colors = {
            'baseline': self.colors[0],
            'finetuned': self.colors[1],
            'noLN': self.colors[2]
        }

        self.l2_norms = {model_name: {} for model_name in self.model_names}
        self.norm_growth_results = {model_name: {} for model_name in self.model_names}
        self.cosine_sim_results = {model_name: {} for model_name in self.model_names}


    def get_hook_names(self, model):
        """
        Get appropriate hook names for a given model.
        
        Args:
            model: The model to get hook names for
            
        Returns:
            list: List of hook names
        """
        hook_names = ['hook_embed', 'hook_pos_embed']  # Raw embeddings
        
        # Add block-specific hooks
        n_layers = len(model.blocks)
        for i in range(n_layers):
            hook_names.append(f'blocks.{i}.hook_resid_mid')
            hook_names.append(f'blocks.{i}.hook_resid_post')
        
        # Add final layer norm hook
        hook_names.append('ln_final.hook_normalized')
        
        return hook_names
    
    def _get_hook_pairs_and_labels(self):
        """
        Create logical pairs of hooks for comparison between consecutive layers,
        and generate readable labels for these pairs.
        
        Returns:
            tuple: (pairs, pair_labels, hooks, hook_labels) where:
                - pairs is a list of (input_hook, output_hook) tuples
                - pair_labels is a list of formatted strings for the pairs
                - hooks is a sorted list of all hook names
                - hook_labels is a dictionary mapping hook names to their display labels
        """
        # Extract all common hook names
        first_model_name = self.model_names[0]
        model = self.model_factory.models[first_model_name]
        hook_names = self.get_hook_names(model)
        
        # Sort hooks to ensure logical progression
        hook_names.sort(key=lambda x: (
            0 if 'hook_embed' in x and 'pos' not in x else 
            1 if 'hook_pos_embed' in x else
            2 if 'blocks' in x else 
            3 if 'ln_final' in x else
            4,
            int(x.split('.')[1]) if 'blocks' in x and '.' in x else 0,
            0 if 'resid_mid' in x else 1 if 'resid_post' in x else 2 if 'ln_final' in x else 0
        ))
        
        # Create hook label mapping
        hook_labels = {}
        for hook in hook_names:
            if 'hook_embed' in hook and 'pos' not in hook:
                hook_labels[hook] = 'Embed'
            elif 'hook_pos_embed' in hook:
                hook_labels[hook] = 'Pos_Embed'
            elif 'blocks' in hook:
                parts = hook.split('.')
                layer_num = parts[1]
                if 'resid_mid' in hook:
                    hook_labels[hook] = f'Resid_mid_{layer_num}'
                elif 'resid_post' in hook:
                    hook_labels[hook] = f'Resid_post_{layer_num}'
            elif 'ln_final' in hook:
                # Always use LN_final label, even for models where it's an identity function
                hook_labels[hook] = 'LN_final'                    
            else:
                hook_labels[hook] = hook.replace('hook_', '')
        
        # Create pairs for comparison (input -> output)
        pairs = []
        
        # Explicitly add the Embed -> Pos_Embed pair first
        if 'hook_embed' in hook_names and 'hook_pos_embed' in hook_names:
            pairs.append(('hook_embed', 'hook_pos_embed'))
        
        # Add other logical pairs
        for i in range(len(hook_names) - 1):
            current_hook = hook_names[i]
            next_hook = hook_names[i+1]
            
            # Skip if we've already added embed->pos_embed
            if current_hook == 'hook_embed' and next_hook == 'hook_pos_embed':
                continue
            
            # Add pos_embed -> first layer's resid_mid
            if 'hook_pos_embed' in current_hook and 'blocks.0.hook_resid_mid' in next_hook:
                pairs.append((current_hook, next_hook))
            
            # Add standard block transitions
            elif 'blocks' in current_hook and 'blocks' in next_hook:
                current_parts = current_hook.split('.')
                next_parts = next_hook.split('.')
                
                # Same layer: resid_mid -> resid_post
                if current_parts[1] == next_parts[1] and 'resid_mid' in current_hook and 'resid_post' in next_hook:
                    pairs.append((current_hook, next_hook))
                
                # Adjacent layers: resid_post -> next resid_mid
                elif int(next_parts[1]) == int(current_parts[1]) + 1 and 'resid_post' in current_hook and 'resid_mid' in next_hook:
                    pairs.append((current_hook, next_hook))
                    
            # Add transition from final transformer block -> ln_final
            elif 'blocks' in current_hook and 'resid_post' in current_hook and 'ln_final' in next_hook:
                pairs.append((current_hook, next_hook))
        
        # Generate pair labels using hook labels
        pair_labels = []
        for input_hook, output_hook in pairs:
            pair_labels.append(f"{hook_labels[output_hook]}")
        
        return pairs, pair_labels, hook_names, hook_labels

    def run_analysis(self, batch_size: int = 1):
        """
        Run the residual stream analysis on all models for all datasets.
        
        Args:
            batch_size: Number of examples to process at once
        """
        # Get pairs for norm growth and cosine similarity calculations
        pairs, _, _, _ = self._get_hook_pairs_and_labels()
        
        print(f"Created {len(pairs)} hook pairs for comparison")
        for i, (input_hook, output_hook) in enumerate(pairs):
            print(f"  Pair {i}: {input_hook} -> {output_hook}")
        
        self._run_dataset_analysis(self.df, pairs, batch_size)
    
    def _run_dataset_analysis(self, df, pairs, batch_size):
        """
        Run analysis for a single dataset
        
        Args:
            df: DataFrame containing the dataset
            pairs: Hook pairs to analyze
            batch_size: Batch size for processing
        """
        for model_name in self.model_names:
            # Initialize result storage for each pair
            for i, (input_hook, output_hook) in enumerate(pairs):
                self.norm_growth_results[model_name][i] = []
                self.cosine_sim_results[model_name][i] = []
        
        # Process data in batches
        for i in tqdm(range(0, len(df), batch_size), desc=f"Running analysis"):
            batch_df = df.iloc[i:i+batch_size]
            
            # Get token sequences from the dataframe
            sequences = batch_df['full_sequence'].tolist()
            
            # Convert sequences to tensors
            inputs = torch.tensor(sequences).to(next(self.model_factory.models[self.model_names[0]].parameters()).device)
            
            # Process each model
            for model_name in self.model_names:
                # Dictionary to store all activations for this batch
                batch_activations = {}
                
                model = self.model_factory.models[model_name]
                hook_names = self.get_hook_names(model)
                
                # Initialize results containers if needed
                results_dict = self.l2_norms[model_name]
                
                for hook_name in hook_names:
                    if hook_name not in results_dict:
                        results_dict[hook_name] = []
                
                # Define the hook function to capture activations only
                def capture_activations(act, hook):
                    hook_name = hook.name
                    
                    if self.last_token_only:
                        act = act[:, -1:, :]  # -1: keeps the dims
                    
                    # Store activation for later metric computation
                    batch_activations[hook_name] = act.detach().cpu()
                    
                    return act
                
                # Create hooks list for all hooks except ln_final for models that don't have it
                hooks_list = []
                for hook_name in hook_names:
                    hooks_list.append((hook_name, capture_activations))
                
                # Run the model with hooks for the entire batch
                with torch.no_grad():
                    output = model.run_with_hooks(
                        inputs,
                        fwd_hooks=hooks_list
                    )                
                
                # For models without ln_final, manually create the identity activation
                if 'noLN' in model_name:
                    # Get the last layer's output
                    last_layer_idx = len(model.blocks) - 1
                    last_resid_hook = f'blocks.{last_layer_idx}.hook_resid_post'
                    batch_activations['ln_final.hook_normalized'] = batch_activations[last_resid_hook].clone()
                
                # First calculate and store L2 norms for each hook
                for hook_name, activation in batch_activations.items():
                    l2_norm = torch.norm(activation, p=2, dim=-1)
                    results_dict[hook_name].append(l2_norm)
                
                # Then compute paired metrics
                for pair_idx, (input_hook, output_hook) in enumerate(pairs):
                    # Check if both input and output activations are available
                    if input_hook in batch_activations and output_hook in batch_activations:
                        input_act = batch_activations[input_hook]
                        output_act = batch_activations[output_hook]
                        
                        # Compute L2 norm growth
                        input_norm = torch.norm(input_act, p=2, dim=-1)
                        output_norm = torch.norm(output_act, p=2, dim=-1)
                        norm_ratio = output_norm / input_norm
                        self.norm_growth_results[model_name][pair_idx].append(norm_ratio)
                        
                        # Compute cosine similarity
                        # Need to reshape for batch and position-wise calculation
                        batch_size, seq_len, hidden_dim = input_act.shape
                        
                        # Reshape to (batch_size * seq_len, hidden_dim)
                        input_flat = input_act.reshape(-1, hidden_dim)
                        output_flat = output_act.reshape(-1, hidden_dim)
                        
                        # Compute cosine similarity for all tokens at once
                        cos_sim = torch.nn.functional.cosine_similarity(input_flat, output_flat, dim=1)
                        
                        # Reshape back to (batch_size, seq_len)
                        cos_sim = cos_sim.reshape(batch_size, seq_len)
                        self.cosine_sim_results[model_name][pair_idx].append(cos_sim)
    
    def plot_norms(self, save_path=None):
        """
        Plot layer norms across layers for each model and dataset.
        
        Args:
            save_path: Path to save the plot, or None to display
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Extract hook labels
        _, _, hook_names, hook_labels = self._get_hook_pairs_and_labels()
        x_labels = [hook_labels[hook] for hook in hook_names]
        
        # Define positions for analysis
        if self.last_token_only:
            positions = [('first', 'Last Token Norm')]
            fig, axes = plt.subplots(1, 1, figsize=(8, 8))
            axes = np.array([[axes]])  # Make it 2D for consistent indexing
        else:
            positions = [('first', 'First Token Norm'), ('others', 'Other Token Norms')]
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            axes = axes.reshape(1, -1)  # Make it 2D for consistent indexing

        # Calculate global min and max values for consistent y-axis scaling
        global_min = float('inf')
        global_max = float('-inf')
            
        for pos_idx, (pos_key, pos_title) in enumerate(positions):
            ax = axes[0, pos_idx]
            
            for model_name in self.model_names:
                model_data = self.l2_norms[model_name]
                
                # Collect mean and std for each hook point
                means = []
                stds = []
                
                for hook in hook_names:
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
            ax.set_ylabel('L2 Norm')
            ax.set_title(f'{pos_title}')
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
    
    def plot_cosine_similarity(self, save_path=None):
        """
        Plot the cosine similarity between consecutive layers.
    
        Args:
            save_path: Path to save the plot, or None to display
    
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Get pairs and their labels directly
        pairs, pair_labels, _, _ = self._get_hook_pairs_and_labels()
        x_labels = pair_labels
    
        if self.last_token_only:
            positions = [('first', 'Last Token Cosine Similarity')]
            fig, axes = plt.subplots(1, 1, figsize=(14, 8))
            axes = np.array([[axes]])
        else:
            positions = [('first', 'First Token Cosine Similarity'), ('others', 'Other Token Cosine Similarity')]
            fig, axes = plt.subplots(1, 2, figsize=(28, 8))
            axes = axes.reshape(1, -1)
            
        # Compute cosine similarity for each model and pair
        for pos_idx, (pos_key, pos_title) in enumerate(positions):
            ax = axes[0, pos_idx]
            
            for model_name in self.model_names:
                # Collect cosine similarities for each pair
                cos_sims_mean = []
                cos_sims_std = []
                
                for pair_idx in range(len(pairs)):
                    if pair_idx in self.cosine_sim_results[model_name] and self.cosine_sim_results[model_name][pair_idx]:
                        # Concatenate all batches
                        all_sims = torch.cat(self.cosine_sim_results[model_name][pair_idx], dim=0)
                        
                        # Split by position
                        if pos_key == 'first':
                            # First token position only
                            position_sims = all_sims[:, 0]
                        else:
                            # All other positions
                            position_sims = all_sims[:, 1:].reshape(-1)
                        
                        # Compute mean and std
                        mean = position_sims.mean().item()
                        std = position_sims.std().item()
                    else:
                        # No data for this pair, use NaN to skip in plot
                        print(f"Warning: No data for {model_name} at pair_idx {pair_idx} ({pairs[pair_idx]})")
                        mean = float('nan')
                        std = float('nan')
                    
                    cos_sims_mean.append(mean)
                    cos_sims_std.append(std)
                
                # Filter out NaN values for plotting
                x = []
                means = []
                stds = []
                for i, (mean, std) in enumerate(zip(cos_sims_mean, cos_sims_std)):
                    if not np.isnan(mean) and not np.isnan(std):
                        x.append(i)
                        means.append(mean)
                        stds.append(std)
                
                # Plot with shaded uncertainty region
                ax.plot(x, means, 'o-', label=model_name, color=self.model_colors[model_name])
                ax.fill_between(x,
                               [max(0, m - s) for m, s in zip(means, stds)],  # Ensure lower bound >= 0
                               [min(1, m + s) for m, s in zip(means, stds)],  # Ensure upper bound <= 1
                               alpha=0.2, color=self.model_colors[model_name])
            
            # Set labels and title
            ax.set_xlabel('Layer Output')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title(f'{pos_title}')
            
            # Set x-ticks
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Set y-axis limits between 0 and 1
            ax.set_ylim(0, 1)
    
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        else:
            plt.show()
    
        return fig

    def plot_norm_growth(self, save_path=None):
        """
        Plot the L2 norm growth ratio between consecutive layers.
        
        Args:
            save_path: Path to save the plot, or None to display
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Get hook pairs and labels
        pairs, pair_labels, _, _ = self._get_hook_pairs_and_labels()
        x_labels = pair_labels
        
        if self.last_token_only:
            positions = [('first', 'Last Token Norm Growth')]
            fig, axes = plt.subplots(1, 1, figsize=(12, 8))
            axes = np.array([[axes]])
        else:
            positions = [('first', 'First Token Norm Growth'), ('others', 'Other Token Norm Growth')]
            fig, axes = plt.subplots(1, 2, figsize=(24, 8))
            axes = axes.reshape(1, -1)
            
        # First, calculate global min and max values for consistent y-axis scaling
        global_min = float('inf')
        global_max = float('-inf')
            
        for pos_idx, (pos_key, pos_title) in enumerate(positions):
            ax = axes[0, pos_idx]
            
            for model_name in self.model_names:
                # Collect norm growth ratios for each pair
                growth_ratios_mean = []
                growth_ratios_std = []
                
                for pair_idx in range(len(pairs)):
                    # Check if we have data for this pair
                    if pair_idx in self.norm_growth_results[model_name] and self.norm_growth_results[model_name][pair_idx]:
                        # Concatenate all batches
                        all_ratios = torch.cat(self.norm_growth_results[model_name][pair_idx], dim=0)
                        
                        # Split by position
                        if pos_key == 'first':
                            # First token position only
                            position_ratios = all_ratios[:, 0]
                        else:
                            # All other positions
                            position_ratios = all_ratios[:, 1:].reshape(-1)
                        
                        # Compute mean and std
                        mean = position_ratios.mean().item()
                        std = position_ratios.std().item()
                    else:
                        # No data for this pair, use NaN to skip in plot
                        print(f"Warning: No data for {model_name} at pair_idx {pair_idx} ({pairs[pair_idx]})")
                        mean = float('nan')
                        std = float('nan')
                    
                    growth_ratios_mean.append(mean)
                    growth_ratios_std.append(std)
                    
                    # Update global min/max if we have valid data
                    if not np.isnan(mean) and not np.isnan(std):
                        global_min = min(global_min, mean - std)
                        global_max = max(global_max, mean + std)
                
                # Filter out NaN values for plotting
                x = []
                means = []
                stds = []
                for i, (mean, std) in enumerate(zip(growth_ratios_mean, growth_ratios_std)):
                    if not np.isnan(mean) and not np.isnan(std):
                        x.append(i)
                        means.append(mean)
                        stds.append(std)
                
                # Plot with shaded uncertainty region
                ax.plot(x, means, 'o-', label=model_name, color=self.model_colors[model_name])
                ax.fill_between(x, 
                               [m - s for m, s in zip(means, stds)],
                               [m + s for m, s in zip(means, stds)],
                               alpha=0.2, color=self.model_colors[model_name])
            
            # Add a horizontal line at y=1 for reference (no growth)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # Set labels and title
            ax.set_xlabel('Layer Transition')
            ax.set_ylabel('L2 Norm Growth Ratio (Output/Input)')
            ax.set_title(f'{pos_title}')
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

if __name__ == '__main__':
    data_paths = '/workspace/removing-layer-norm/mech_interp/inference_logs/dataset_luca-pile_samples_1000_seqlen_512_prepend_False/inference_results.parquet'
    output_dir = 'figures/last_token'
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = ResidualNorms(
        data_paths=data_paths,
        model_dir="../models",
        last_token_only=True,
    )
    
    # Run analysis and generate plots
    analyzer.run_analysis()
    analyzer.plot_norms(os.path.join(output_dir, "l2_norms.png"))
    analyzer.plot_norm_growth(os.path.join(output_dir, "l2_norm_growth.png"))
    analyzer.plot_cosine_similarity(os.path.join(output_dir, "cosine_similarity.png"))
