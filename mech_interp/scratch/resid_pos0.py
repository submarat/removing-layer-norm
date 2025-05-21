import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Union, Literal, Tuple
from transformer_lens import HookedTransformer
from tqdm import tqdm

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_models import ModelFactory


class VocabularyNormsAnalyzer:
    def __init__(self,
                 model_dir: str = "../models",
                 batch_size: int = 32,
                 device: Optional[str] = None):
        """
        Initialize the analyzer to examine norm behavior for every token in the vocabulary.
        
        Args:
            model_dir: Directory containing the model files
            batch_size: Number of tokens to process at once
            device: Device to use for computation
        """
        # Initialize model factory
        self.model_names = ['baseline', 'finetuned', 'noLN']
        self.model_factory = ModelFactory(self.model_names, model_dir=model_dir)
        
        # Set batch size for processing
        self.batch_size = batch_size
        
        # Results will be stored here after running analysis
        self.results = None
        
        # Set up color schemes for consistent visualization
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.model_colors = {
            'baseline': self.colors[0],
            'finetuned': self.colors[1],
            'noLN': self.colors[2]
        }
    
    def get_hook_names(self, model):
        """Get appropriate hook names for a given model."""
        hook_names = ['hook_embed']  # Raw embeddings
        
        # Add block-specific hooks
        n_layers = len(model.blocks)
        for i in range(n_layers):
            hook_names.append(f'blocks.{i}.hook_resid_mid')
            hook_names.append(f'blocks.{i}.hook_resid_post')
        
        return hook_names
    
    def analyze_vocabulary(self):
        """
        Run analysis for every token in the vocabulary as a single-token sequence.
        """
        results = {}
        
        # Get vocabulary size from the first model
        model = self.model_factory.models[self.model_names[0]]
        vocab_size = model.cfg.d_vocab  # Using d_vocab instead of vocab_size
        print(f"Analyzing {vocab_size} tokens in the vocabulary")
        
        # Process tokens in batches
        for token_start in tqdm(range(0, vocab_size, self.batch_size)):
            token_end = min(token_start + self.batch_size, vocab_size)
            token_indices = list(range(token_start, token_end))
            
            # Create single-token sequences for each token index
            inputs = torch.tensor([[token_idx] for token_idx in token_indices], dtype=torch.long)
            inputs = inputs.to(next(model.parameters()).device)
            
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
                
                # Define the hook function to capture norms
                def capture_norms(act, hook):
                    # For single token, we're only interested in position 0
                    l2_norm = torch.norm(act, p=2, dim=-1)
                    hook_name = hook.name
                    results[model_name][hook_name].append(l2_norm.detach().cpu())
                    return act
                
                # Run the model with hooks
                with torch.no_grad():
                    model.run_with_hooks(
                        inputs,
                        fwd_hooks=[(name, capture_norms) for name in hook_names]
                    )
        
        # Store results
        self.results = results
        return results
    
    def generate_random_vectors(self, model, n_vectors=1000):
        """
        Generate random vectors with the same dimensions as model embeddings.
        
        Args:
            model: The transformer model to match dimensions with
            n_vectors: Number of random vectors to generate
            
        Returns:
            Dictionary with hook names and corresponding random vector norms
        """
        results = {}
        hook_names = self.get_hook_names(model)
        d_model = model.cfg.d_model
        
        # Generate random vectors matching embedding distribution
        # For a more accurate comparison, we match the distribution of the embedding layer
        with torch.no_grad():
            # Get embedding layer statistics
            embed_weight = model.W_E
            mean = embed_weight.mean().item()
            std = embed_weight.std().item()
            
            # Create random vectors with matching distribution
            random_vecs = torch.randn(n_vectors, 1, d_model) * std + mean
            random_vecs = random_vecs.to(embed_weight.device)
            # Process random vectors through the model
            for hook_name in hook_names:
                # Define hook function to capture norms
                def capture_random_norms(act, hook):
                    if hook.name == 'hook_embed':
                        # For embed hook, replace activations with our random vectors
                        l2_norm = torch.norm(random_vecs, p=2, dim=-1)
                        results[hook_name] = l2_norm.detach().cpu()
                        return random_vecs
                    else:
                        # For other hooks, calculate norm and let activations pass through
                        l2_norm = torch.norm(act, p=2, dim=-1)
                        results[hook_name] = l2_norm.detach().cpu()
                        return act
                
                # Create dummy input for the forward pass
                dummy_input = torch.zeros(1, 1, dtype=torch.long).to(embed_weight.device)
                
                # Run model with hooks
                model.run_with_hooks(
                    dummy_input,
                    fwd_hooks=[(hook_name, capture_random_norms)]
                )
        
        return results
    
    def plot_vocab_norms(self, save_path=None, random_vectors=True, n_random=1000):
        """
        Plot layer norms across layers for each model:
        1. Statistical distribution for vocabulary tokens
        2. Statistical distribution for random vectors matching embedding dimensionality
        
        Args:
            save_path: Path to save the plot, or None to display
            random_vectors: Whether to include random vector comparison
            n_random: Number of random vectors to generate
        """
        if self.results is None:
            print("No results available. Please run analyze_vocabulary first.")
            return
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Define simplified hook labels for x-axis
        hook_mapping = {
            'hook_embed': 'Embed',
        }
        
        # Add transformer block mappings
        model = self.model_factory.models[self.model_names[0]]
        n_layers = len(model.blocks)
        for i in range(n_layers):
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
        
        # First plot: Statistical summary across all vocabulary tokens
        ax = axes[0]
        
        # Define percentiles to show
        percentiles = [5, 25, 50, 75, 95]
        
        for model_name in self.model_names:
            model_data = self.results[model_name]
            
            # Collect statistics for each hook point
            medians = []
            p25_75 = []  # 25th and 75th percentiles
            p5_95 = []   # 5th and 95th percentiles
            
            for hook in common_hooks:
                if hook in model_data:
                    # Concatenate all batches
                    all_values = torch.cat(model_data[hook], dim=0)
                    # Extract position 0 norms (single token)
                    position_values = all_values[:, 0].numpy()
                    
                    # Compute percentiles
                    p = np.percentile(position_values, percentiles)
                    
                    medians.append(p[2])  # 50th percentile
                    p25_75.append((p[1], p[3]))  # 25th and 75th
                    p5_95.append((p[0], p[4]))   # 5th and 95th
            
            # Plot with shaded regions for percentile ranges
            x = np.arange(len(medians))
            ax.plot(x, medians, 'o-', label=model_name, color=self.model_colors[model_name])
            
            # Fill between 25th and 75th percentiles
            ax.fill_between(x, 
                           [p[0] for p in p25_75],
                           [p[1] for p in p25_75],
                           alpha=0.3, color=self.model_colors[model_name])
            
            # Fill between 5th and 95th percentiles
            ax.fill_between(x, 
                           [p[0] for p in p5_95],
                           [p[1] for p in p5_95],
                           alpha=0.1, color=self.model_colors[model_name])
        
        # Set labels and title
        ax.set_xlabel('Layer')
        ax.set_ylabel('L2 Norm')
        ax.set_title('L2 Norm - Statistical Distribution Across All Vocabulary Tokens')
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Second plot: Random vectors with matching dimensions
        ax = axes[1]
        
        # Only proceed with random vectors if enabled
        if random_vectors:
            # Generate random vector results for each model
            random_results = {}
            
            for model_name in self.model_names:
                model = self.model_factory.models[model_name]
                random_results[model_name] = self.generate_random_vectors(model, n_vectors=n_random)
            
            # Plot statistics for random vectors
            for model_name in self.model_names:
                # Collect statistics for each hook point
                medians = []
                p25_75 = []  # 25th and 75th percentiles
                p5_95 = []   # 5th and 95th percentiles
                
                for hook in common_hooks:
                    if hook in random_results[model_name]:
                        # Get values for this hook
                        values = random_results[model_name][hook].numpy()
                        
                        # Compute percentiles
                        p = np.percentile(values, percentiles)
                        
                        medians.append(p[2])  # 50th percentile
                        p25_75.append((p[1], p[3]))  # 25th and 75th
                        p5_95.append((p[0], p[4]))   # 5th and 95th
                
                # Plot with shaded regions for percentile ranges
                x = np.arange(len(medians))
                ax.plot(x, medians, 'o-', label=model_name, color=self.model_colors[model_name])
                
                # Fill between 25th and 75th percentiles
                ax.fill_between(x, 
                               [p[0] for p in p25_75],
                               [p[1] for p in p25_75],
                               alpha=0.3, color=self.model_colors[model_name])
                
                # Fill between 5th and 95th percentiles
                ax.fill_between(x, 
                               [p[0] for p in p5_95],
                               [p[1] for p in p5_95],
                               alpha=0.1, color=self.model_colors[model_name])
            
            # Set labels and title
            ax.set_xlabel('Layer')
            ax.set_ylabel('L2 Norm')
            ax.set_title(f'L2 Norm - Statistical Distribution Across {n_random} Random Vectors')
        else:
            ax.text(0.5, 0.5, 'Random vector analysis disabled', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
        
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        else:
            plt.show()
        
        return fig
        
        for i, model_name in enumerate(self.model_names):
            ax = axes[i] if len(self.model_names) > 1 else axes
            
            # Get initial and final norms
            initial_norms = torch.cat(self.results[model_name][first_hook], dim=0)[:, 0].numpy()
            final_norms = torch.cat(self.results[model_name][last_hook], dim=0)[:, 0].numpy()
            
            # Calculate norm growth
            norm_growth = final_norms / initial_norms
            
            # Create x-axis values (frequency or token index)
            if token_frequencies is not None:
                x_values = token_frequencies['frequency'].values[:len(norm_growth)]
                x_label = 'Token Frequency (log scale)'
                log_scale = True
            else:
                x_values = np.arange(len(norm_growth))
                x_label = 'Token Index'
                log_scale = False
            
            # Plot scatter with transparency
            ax.scatter(x_values, norm_growth, alpha=0.3, color=self.model_colors[model_name])
            
            # Add trend line
            z = np.polyfit(x_values, norm_growth, 1)
            p = np.poly1d(z)
            ax.plot(x_values, p(x_values), "r--", alpha=0.8)
            
            # Set labels and title
            ax.set_xlabel(x_label)
            ax.set_ylabel('Norm Growth (Final/Initial)')
            ax.set_title(f'{model_name} Token Norm Growth')
            
            if log_scale:
                ax.set_xscale('log')
                
            ax.grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200)
        else:
            plt.show()
        
        return fig


if __name__ == '__main__':
    # Initialize the vocabulary norms analyzer
    analyzer = VocabularyNormsAnalyzer(
        model_dir="../models",
        batch_size=64  # Process 64 tokens at a time
    )
    
    output_dir = 'vocab_norms'
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the vocabulary analysis
    analyzer.analyze_vocabulary()
    
    # Generate the norm plots with both token statistics and random vector comparison
    analyzer.plot_vocab_norms(
        save_path=os.path.join(output_dir, 'vocab_vs_random_norms_nofolding.png'),
        random_vectors=True,
        n_random=50000  # Number of random vectors to generate
    )
