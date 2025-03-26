import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Optional, Union, Literal, Tuple
from transformer_lens import HookedTransformer

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_models import ModelFactory


class ModelWeights:
    """
    A class for analyzing and visualizing model weights across different model variants.
    Provides methods for comparing token embeddings, positional embeddings, various attention weights,
    and ffn weights.
    """
    
    def __init__(self, model_dir: str, device: Optional[Union[str, torch.device]] = None):
        """
        Initialize the ModelWeights analysis class.
        
        Parameters:
        -----------
        model_dir : str
            Directory containing model files or where models will be downloaded
        device : str or torch.device, optional
            Device to load models on (default: auto-detect)
        """
        # Set up the colorblind-friendly style
        plt.style.use('seaborn-v0_8-colorblind')
        
        # Store the model names for consistent reference
        self.models_to_load = ['baseline', 'finetuned', 'noLN']
        
        # Set up color schemes for consistent visualization
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.model_colors = {
            'baseline': self.colors[0],
            'finetuned': self.colors[1],
            'noLN': self.colors[2]
        }
        
        # Store the model directory for later reference
        self.model_dir = model_dir
        
        # Load the models
        self.model_factory = ModelFactory(
            model_names=self.models_to_load,
            model_dir=model_dir,
            device=device,
            fold_ln=True,
            center_unembed=False,
            eval_mode=True
        )
        
        # Access the loaded models
        self.models = self.model_factory.models
        
        # Print model loading confirmation
        print(f"Loaded models: {', '.join(self.models.keys())}")


    def get_embedding_stats(self, model: HookedTransformer) -> Dict[str, torch.Tensor]:
        """
        Calculate statistics for the token embedding matrix.
        
        Parameters:
        -----------
        model : HookedTransformer
            The model to analyze
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary containing embedding statistics
        """
        # Get the token embedding matrix (W_E)
        embed_matrix = model.W_E.detach().cpu()  # Shape: [vocab_size, d_model]
        
        # Calculate L2 norms for each embedding vector
        l2_norms = torch.norm(embed_matrix, dim=1)
        
        # Calculate L1 norms for each embedding vector
        l1_norms = torch.norm(embed_matrix, p=1, dim=1)
        
        # Calculate L1/L2 ratio
        l1_l2_ratio = l1_norms / l2_norms
        
        return {
            'l2_norms': l2_norms,
            'l1_norms': l1_norms,
            'l1_l2_ratio': l1_l2_ratio,
        }
   

    def get_pos_embedding_stats(self, model: HookedTransformer) -> Dict[str, torch.Tensor]:
        """
        Calculate statistics for the positional embedding matrix.
        
        Parameters:
        -----------
        model : HookedTransformer
            The model to analyze
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary containing positional embedding statistics
        """
        # Get the positional embedding matrix (W_pos)
        pos_embed_matrix = model.W_pos.detach().cpu()  # Shape: [max_seq_len, d_model]
        
        # Calculate L2 norms for each positional embedding vector
        l2_norms = torch.norm(pos_embed_matrix, dim=1)
        
        # Calculate L1 norms for each positional embedding vector
        l1_norms = torch.norm(pos_embed_matrix, p=1, dim=1)
        
        # Calculate L1/L2 ratio
        l1_l2_ratio = l1_norms / l2_norms
        
        return {
            'l2_norms': l2_norms,
            'l1_norms': l1_norms,
            'l1_l2_ratio': l1_l2_ratio,
        }


    def plot_embedding_stats(self, bins: int = 100, figsize: Tuple[int, int] = (16, 6),
                             log_scale: bool = True, save_path: Optional[str] = None):
        """
        Plot histograms of L2 norm and L1/L2 ratio for token embeddings across models.

        Parameters:
        -----------
        bins : int, optional
            Number of bins for histograms
        figsize : tuple, optional
            Figure size (width, height)
        log_scale : bool, optional
            Whether to use logarithmic scale for y-axis
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Get embedding stats for all models
        embedding_stats = {model_name: self.get_embedding_stats(model)
                           for model_name, model in self.models.items()}

        # Plot L2 norm histograms
        for model_name, stats in embedding_stats.items():
            mean_l2 = stats['l2_norms'].mean().item()
            axes[0].hist(
                stats['l2_norms'].numpy(),
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=0.7,
                label=f'{model_name} mean: {mean_l2:.4f}',
                color=self.model_colors[model_name]
            )

            # Add mean lines
            axes[0].axvline(
                mean_l2,
                color=self.model_colors[model_name],
                linestyle='--',
                linewidth=1.5,
                label=f'{model_name} mean: {mean_l2:.4f}'
            )

        # Plot L1/L2 ratio histograms
        for model_name, stats in embedding_stats.items():
            mean_ratio = stats['l1_l2_ratio'].mean().item()
            axes[1].hist(
                stats['l1_l2_ratio'].numpy(),
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=0.7,
                label=f'{model_name} mean: {mean_ratio:.4f}',
                color=self.model_colors[model_name]
            )

            # Add mean lines
            axes[1].axvline(
                mean_ratio,
                color=self.model_colors[model_name],
                linestyle='--',
                linewidth=1.5,
            )

        # Set titles and labels
        axes[0].set_title('Token Embedding L2 Norms', fontsize=16)
        axes[0].set_xlabel('L2 Norm', fontsize=14)
        axes[0].set_ylabel('Count', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].grid(alpha=0.3)

        axes[1].set_title('Token Embedding L1/L2 Ratio', fontsize=16)
        axes[1].set_xlabel('L1/L2 Ratio', fontsize=14)
        axes[1].set_ylabel('Count', fontsize=14)
        axes[1].legend(fontsize=12)
        axes[1].grid(alpha=0.3)

        # Set log scale if requested
        if log_scale:
            axes[0].set_yscale('log')
            axes[0].set_ylabel('Count (log scale)', fontsize=14)
            axes[1].set_yscale('log')
            axes[1].set_ylabel('Count (log scale)', fontsize=14)

        plt.suptitle('Token Embedding Statistics Across Models', fontsize=18)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


    def plot_pos_embedding_values(self, figsize: Tuple[int, int] = (16, 6), 
                                  save_path: Optional[str] = None):
        """
        Plot positional embedding norms as scatter plots showing how they change with position index.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Get positional embedding stats for all models
        pos_embedding_stats = {model_name: self.get_pos_embedding_stats(model) 
                              for model_name, model in self.models.items()}
        
        # Set up position indices
        for model_name, stats in pos_embedding_stats.items():
            # Get the position indices
            positions = np.arange(len(stats['l2_norms']))
            
            # Plot L2 norms by position
            axes[0].scatter(
                positions, 
                stats['l2_norms'].numpy(),
                label=model_name,
                color=self.model_colors[model_name],
                alpha=0.7,
                s=15
            )
            
            # Plot L1/L2 ratios by position
            axes[1].scatter(
                positions, 
                stats['l1_l2_ratio'].numpy(),
                label=model_name,
                color=self.model_colors[model_name],
                alpha=0.7,
                s=15
            )
        
        # Add trend lines with lowess or moving average for clearer visualization
        for model_name, stats in pos_embedding_stats.items():
            positions = np.arange(len(stats['l2_norms']))
            
            # Simple moving average for trend lines
            window = 20
            l2_norms = stats['l2_norms'].numpy()
            l1_l2_ratios = stats['l1_l2_ratio'].numpy()
            
            # Calculate moving averages
            l2_smoothed = np.convolve(l2_norms, np.ones(window)/window, mode='valid')
            l1_l2_smoothed = np.convolve(l1_l2_ratios, np.ones(window)/window, mode='valid')
            
            # Adjust positions for the reduced length of smoothed data
            smooth_positions = positions[window-1:][:len(l2_smoothed)]
            
            # Plot trend lines
            axes[0].plot(
                smooth_positions, 
                l2_smoothed,
                color=self.model_colors[model_name],
                linewidth=2,
                alpha=0.8
            )
            
            axes[1].plot(
                smooth_positions, 
                l1_l2_smoothed,
                color=self.model_colors[model_name],
                linewidth=2,
                alpha=0.8
            )
        
        # Set titles and labels
        axes[0].set_title('Positional Embedding L2 Norms by Position', fontsize=16)
        axes[0].set_xlabel('Position Index', fontsize=14)
        axes[0].set_ylabel('L2 Norm', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].grid(alpha=0.3)
        
        axes[1].set_title('Positional Embedding L1/L2 Ratio by Position', fontsize=16)
        axes[1].set_xlabel('Position Index', fontsize=14)
        axes[1].set_ylabel('L1/L2 Ratio', fontsize=14)
        axes[1].legend(fontsize=12)
        axes[1].grid(alpha=0.3)
        
        plt.suptitle('Positional Embedding Metrics by Position', fontsize=18)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


    def get_attention_weight_stats(self, model: HookedTransformer) -> Dict:
        """
        Calculate statistics for attention weight matrices (Wq, Wk, Wv, Wo) across all layers.
        
        Parameters:
        -----------
        model : HookedTransformer
            The model to analyze
            
        Returns:
        --------
        Dict[str, Dict[str, List[torch.Tensor]]]
            Dictionary containing statistics for each attention weight matrix type across layers
        """
        # Number of layers in the model
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        d_model = model.cfg.d_model
        d_head = model.cfg.d_head
        
        # Initialize result dictionaries
        stats = {
            'Wq': {'l2_norms': []},
            'Wk': {'l2_norms': []},
            'Wv': {'l2_norms': []},
            'Wo': {'l2_norms': []}
        }
        
        # Loop through each layer
        for layer_idx in range(n_layers):
            # Extract the weight matrices
            W_q = model.blocks[layer_idx].attn.W_Q.detach().cpu()  # [d_model, n_heads * d_head]
            W_k = model.blocks[layer_idx].attn.W_K.detach().cpu()
            W_v = model.blocks[layer_idx].attn.W_V.detach().cpu()
            W_o = model.blocks[layer_idx].attn.W_O.detach().cpu()  # [n_heads * d_head, d_model]
            
            # Reshape to separate heads for Q, K, V
            W_q_heads = W_q.reshape(d_model, n_heads, d_head).permute(1, 0, 2)  # [n_heads, d_model, d_head]
            W_k_heads = W_k.reshape(d_model, n_heads, d_head).permute(1, 0, 2)
            W_v_heads = W_v.reshape(d_model, n_heads, d_head).permute(1, 0, 2)
            
            # Reshape for O
            W_o_heads = W_o.reshape(n_heads, d_head, d_model)  # [n_heads, d_head, d_model]
            
            # Calculate per-head Frobenius norms
            W_q_head_norms = torch.tensor([torch.norm(W_q_heads[i]) for i in range(n_heads)])
            W_k_head_norms = torch.tensor([torch.norm(W_k_heads[i]) for i in range(n_heads)])
            W_v_head_norms = torch.tensor([torch.norm(W_v_heads[i]) for i in range(n_heads)])
            W_o_head_norms = torch.tensor([torch.norm(W_o_heads[i]) for i in range(n_heads)])
            
            # Calculate mean and std of head norms as tuples
            stats['Wq']['l2_norms'].append((W_q_head_norms.mean(), W_q_head_norms.std()))
            stats['Wk']['l2_norms'].append((W_k_head_norms.mean(), W_k_head_norms.std()))
            stats['Wv']['l2_norms'].append((W_v_head_norms.mean(), W_v_head_norms.std()))
            stats['Wo']['l2_norms'].append((W_o_head_norms.mean(), W_o_head_norms.std()))
        
        return stats


    def plot_attention_weights(self, figsize: Tuple[int, int] = (10, 16), 
                              save_path: Optional[str] = None):
        """
        Plot mean L2 norms with standard deviation for attention weight matrices 
        (Wq, Wk, Wv, Wo) across all layers for all models.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Get attention weight stats for all models
        attention_stats = {model_name: self.get_attention_weight_stats(model) 
                          for model_name, model in self.models.items()}
        
        # Weight types and their corresponding row index
        weight_types = ['Wq', 'Wk', 'Wv', 'Wo']
        
        # Layer indices as x-axis
        layers = np.arange(len(attention_stats[list(attention_stats.keys())[0]]['Wq']['l2_norms']))
        
        # Plot for each weight type in its own subplot
        for row_idx, weight_type in enumerate(weight_types):
            for model_name, stats in attention_stats.items():
                # Extract means and standard deviations
                means = np.array([norm[0].numpy() for norm in stats[weight_type]['l2_norms']])
                stds = np.array([norm[1].numpy() for norm in stats[weight_type]['l2_norms']])
                
                # Plot mean line
                line = axes[row_idx].plot(
                    layers,
                    means,
                    'o-',
                    label=model_name,
                    color=self.model_colors[model_name],
                    alpha=0.8,
                    markersize=6
                )
                
                # Fill between for standard deviation
                axes[row_idx].fill_between(
                    layers,
                    means - stds,
                    means + stds,
                    color=self.model_colors[model_name],
                    alpha=0.2
                )
        
        # Add labels, legends, and grid for each subplot
        for row_idx, weight_type in enumerate(weight_types):
            axes[row_idx].set_ylabel(f'{weight_type} Mean L2 Norm', fontsize=12)
            
            # Only add x-axis label to the bottom subplot
            if row_idx == 3:
                axes[row_idx].set_xlabel('Layer Index', fontsize=12)
            
            # Add legend only to the first subplot
            if row_idx == 0:
                axes[row_idx].legend(fontsize=10)
            
            # Add grid to all subplots
            axes[row_idx].grid(alpha=0.3)
        
        plt.suptitle('Mean L2 Norm of Attention Weight Matrices Across Layers', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


    def get_ffn_weight_stats(self, model: HookedTransformer) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        """
        Calculate statistics for FFN weight matrices (W_in, W_out) across all layers.
        W_in is the first matrix multiplication (d_model -> 4*d_model)
        W_out is the second matrix multiplication (4*d_model -> d_model)
        
        Parameters:
        -----------
        model : HookedTransformer
            The model to analyze
            
        Returns:
        --------
        Dict[str, Dict[str, List[torch.Tensor]]]
            Dictionary containing statistics for each FFN weight matrix type across layers
        """
        # Number of layers in the model
        n_layers = model.cfg.n_layers
        
        # Initialize result dictionaries
        stats = {
            'W_in': {'l2_norms': []},
            'W_out': {'l2_norms': []}
        }
        
        # Loop through each layer
        for layer_idx in range(n_layers):
            # Extract the weight matrices
            W_in = model.blocks[layer_idx].mlp.W_in.detach().cpu()  # [d_model, 4*d_model]
            W_out = model.blocks[layer_idx].mlp.W_out.detach().cpu()  # [4*d_model, d_model]
            
            # Calculate L2 norms - we compute the Frobenius norm of the entire matrix
            stats['W_in']['l2_norms'].append(torch.norm(W_in))
            stats['W_out']['l2_norms'].append(torch.norm(W_out))
            
        # Convert lists to tensors
        for weight_type in stats:
            for stat_type in stats[weight_type]:
                stats[weight_type][stat_type] = torch.stack(stats[weight_type][stat_type])
        
        return stats

    def plot_ffn_weights(self, figsize: Tuple[int, int] = (16, 6), 
                         save_path: Optional[str] = None):
        """
        Plot L2 norms for FFN weight matrices (W_in, W_out)
        across all layers for all models.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Get FFN weight stats for all models
        ffn_stats = {model_name: self.get_ffn_weight_stats(model) 
                    for model_name, model in self.models.items()}
        
        # Weight types and their corresponding column index
        weight_types = {
            'W_in': 0,  # d_model -> 4*d_model
            'W_out': 1  # 4*d_model -> d_model
        }
        
        # Plot statistics for each weight type and model
        for model_name, stats in ffn_stats.items():
            # Layer indices as x-axis
            layers = np.arange(len(stats['W_in']['l2_norms']))
            
            # Plot each weight type in its own column
            for weight_type, col_idx in weight_types.items():
                # L2 norms
                axes[col_idx].plot(
                    layers,
                    stats[weight_type]['l2_norms'].numpy(),
                    'o-',
                    label=model_name,
                    color=self.model_colors[model_name],
                    alpha=0.8,
                    markersize=6
                )
        
        # Add labels, legends, and grid for each column
        weight_type_names = {
            'W_in': 'FFN W_in',
            'W_out': 'FFN W_out'
        }
        
        for col_idx, weight_type in enumerate(['W_in', 'W_out']):
            # Set titles and labels
            axes[col_idx].set_title(f'{weight_type_names[weight_type]} L2 Norm', fontsize=16)
            axes[col_idx].set_ylabel('L2 Norm', fontsize=14)
            axes[col_idx].set_xlabel('Layer Index', fontsize=14)
            
            # Add legend
            axes[col_idx].legend(fontsize=12)
            
            # Add grid
            axes[col_idx].grid(alpha=0.3)
        
        plt.suptitle('Feed-Forward Network Weight L2 Norms Across Layers', fontsize=20)
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


    def plot_all(self, output_dir: str = 'weight_summary'):
        """
        Generate and save all plots to the specified output directory.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save plots (will be created if it doesn't exist)
        """
        import os
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plots to directory: {output_dir}")
        
        # Generate and save all plots
        print("Generating embedding stats plot...")
        self.plot_embedding_stats(save_path=os.path.join(output_dir, 'embedding.png'))
        
        print("Generating positional embedding plot...")
        self.plot_pos_embedding_values(save_path=os.path.join(output_dir, 'pos_embedding.png'))
        
        print("Generating attention weights plot...")
        self.plot_attention_weights(save_path=os.path.join(output_dir, 'attn_weights.png'))
        
        print("Generating FFN weights plot...")
        self.plot_ffn_weights(save_path=os.path.join(output_dir, 'ffn_weights.png'))
        
        print(f"All plots saved successfully to {output_dir}")


if __name__ == '__main__':
    model_dir = '/workspace/removing-layer-norm/mech_interp/models/'
    weights = ModelWeights(model_dir)
    weights.plot_all()
