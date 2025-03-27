import os
from typing import Dict, List, Optional, Union, Literal, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


class MetricsSummary:
    """
    A class for visualizing model comparison metrics across baseline, finetuned, and noLN models.
    Provides methods for filtering data and creating various visualization types.
    """
    
    def __init__(self, data_path: str, min_seq_length: Optional[int] = None, agg : bool = False):
        """
        Initialize the ModelComparison class.
        
        Parameters:
        -----------
        data_path : str
            Path to the parquet file containing model comparison data
        min_seq_length : int, optional
            Minimum sequence length to include in the analysis
        """
        # Set the colorblind-friendly style
        plt.style.use('seaborn-v0_8-colorblind')
        
        # Load the data
        self.data_path = data_path
        self.df = pd.read_parquet(data_path)
        
        # Store the model names for consistent reference
        self.models = ['baseline', 'finetuned', 'noLN']
        
        # Set up color schemes for consistent visualization
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.model_colors = {
            'baseline': self.colors[0],
            'finetuned': self.colors[1],
            'noLN': self.colors[2]
        }
        
        # Apply sequence length filter if provided
        if min_seq_length is not None:
            self.filter_by_sequence_length(min_seq_length)
        # Aggregate metrics if requested
        self.agg = agg
        if self.agg:
            self.get_aggregate_metrics()


    def filter_by_sequence_length(self, min_length: int):
        """
        Filter the dataframe to include only examples with sequence length >= min_length.
        
        Parameters:
        -----------
        min_length : int
            Minimum sequence length to include
        """
        self.df = self.df[self.df['sequence_length'] >= min_length].reset_index(drop=True)
        print(f"Filtered data to {len(self.df)} examples with sequence length >= {min_length}")


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


    def plot_ce_histogram(self, bins : int = 50, alpha : float = 0.7,
                          figsize: tuple = (12, 6), save_path: Optional[str] = None,
                          logscale: bool = True):
        """
        Plot step histograms comparing CE loss across all three models.
        
        Parameters:
        -----------
        bins : int, optional
            Number of bins for the histogram
        alpha : float, optional
            Transparency level for the histograms
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
        logscale : bool, optional
            Whether to use log for y-axis scale
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model in self.models:
            ax.hist(
                self.df[f'ce_{model}'],
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=alpha,
                label=model,
                color=self.model_colors[model]
            )
        
        # Add mean lines
        for model in self.models:
            mean_val = self.df[f'ce_{model}'].mean()
            ax.axvline(
                mean_val,
                color=self.model_colors[model],
                linestyle='--',
                linewidth=1.5,
                label=f'{model} mean: {mean_val:.4f}'
            )
        
        ax.set_title('Cross-Entropy Loss Distribution Across Models', fontsize=16)
        ax.set_xlabel('Cross-Entropy Loss', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        if logscale:
            ax.set_yscale('log')
            ax.set_ylabel('Count (log scale)', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


    def plot_jsd_histograms(self, bins: int = 50, alpha: float = 0.7,
                            figsize: tuple = (16, 6), save_path: Optional[str] = None,
                            logscale: bool = True):
        """
        Plot a (1,2) subplot of histograms showing JSD and topk_JSD distributions
        for all model comparisons.
        
        Parameters:
        -----------
        bins : int, optional
            Number of bins for the histograms
        alpha : float, optional
            Transparency level for the histograms
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
        logscale : bool, optional
            Whether to use log for y-axis scale
           
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        model_pairs = [('baseline', 'finetuned'), ('baseline', 'noLN'), ('finetuned', 'noLN')]
        pair_colors = [self.colors[i] for i in range(len(model_pairs))]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot JSD histograms
        for i, (m1, m2) in enumerate(model_pairs):
            pair = f'{m1}_vs_{m2}'
            label = f'{m1} vs {m2}'
            
            axes[0].hist(
                self.df[f'jsd_{pair}'],
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=alpha,
                label=label,
                color=pair_colors[i]
            )
            
            axes[1].hist(
                self.df[f'topk_jsd_{pair}'],
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=alpha,
                label=label,
                color=pair_colors[i]
            )
        
        # Set titles and labels
        axes[0].set_title('Jensen-Shannon Divergence', fontsize=16)
        axes[0].set_xlabel('JSD', fontsize=14)
        axes[0].set_ylabel('Count', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].grid(alpha=0.3)
        
        axes[1].set_title('Top-K Jensen-Shannon Divergence', fontsize=16)
        axes[1].set_xlabel('Top-K JSD', fontsize=14)
        axes[1].set_ylabel('Count', fontsize=14)
        axes[1].legend(fontsize=12)
        axes[1].grid(alpha=0.3)
        if logscale:
            axes[0].set_yscale('log')
            axes[0].set_ylabel('Count (log scale)', fontsize=14)
            axes[1].set_yscale('log')
            axes[1].set_ylabel('Count (log scale)', fontsize=14)

        
        plt.suptitle('Distribution of Divergence Metrics Across Model Comparisons', fontsize=18)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


    def plot_jsd_scatterplots(self, figsize: tuple = (16, 8), alpha: float = 0.6, save_path: Optional[str] = None):
        """
        Plot a (1,2) subplot of scatterplots comparing JSD and topk_JSD metrics
        between baseline_vs_finetuned and baseline_vs_noLN.
        Points are colored by CE loss of noLN model.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height)
        alpha : float, optional
            Transparency level for the scatter points
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Create a shared color normalization based on noLN CE loss
        norm = Normalize(
            vmin=self.df['ce_noLN'].min(),
            vmax=np.percentile(self.df['ce_noLN'], 95)  # Cap at 95th percentile to avoid outliers
            #vmax=self.df['ce_noLN'].max()
        )
        
        # JSD scatterplot
        sc1 = axes[0].scatter(
            self.df['jsd_baseline_vs_finetuned'],
            self.df['jsd_baseline_vs_noLN'],
            c=self.df['ce_noLN'],
            cmap='viridis',
            alpha=alpha,
            norm=norm,
            s=25
        )
        
        # Add diagonal line for reference
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # topk_JSD scatterplot
        sc2 = axes[1].scatter(
            self.df['topk_jsd_baseline_vs_finetuned'],
            self.df['topk_jsd_baseline_vs_noLN'],
            c=self.df['ce_noLN'],
            cmap='viridis',
            alpha=alpha,
            norm=norm,
            s=25
        )
        
        # Add diagonal line for reference
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Set titles and labels
        axes[0].set_title('Jensen-Shannon Divergence Comparison', fontsize=16)
        axes[0].set_xlabel('JSD: baseline vs finetuned', fontsize=14)
        axes[0].set_ylabel('JSD: baseline vs noLN', fontsize=14)
        axes[0].grid(alpha=0.3)
        
        axes[1].set_title('Top-K Jensen-Shannon Divergence Comparison', fontsize=16)
        axes[1].set_xlabel('Top-K JSD: baseline vs finetuned', fontsize=14)
        axes[1].set_ylabel('Top-K JSD: baseline vs noLN', fontsize=14)
        axes[1].grid(alpha=0.3)
        
        # Add a shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sc1, cax=cbar_ax)
        cbar.set_label('CE Loss (noLN model)', fontsize=14)
        
        plt.suptitle('Comparison of Divergence Metrics Between Model Pairs', fontsize=18)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to make room for colorbar
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


    def plot_all(self, output_dir: Optional[str] = None):
        """
        Generate and save all visualization types.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the figures
        prefix : str, optional
            Prefix for the filenames
            
        Returns:
        --------
        list
            List of paths to the generated figures
        """
        metrics_str = 'agg' if self.agg else ''

        # If no output directory specified, use the directory of the parquet file
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(self.data_path))

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # CE histogram
        ce_path = os.path.join(output_dir, f'ce_histogram_{metrics_str}.png')
        self.plot_ce_histogram(save_path=ce_path)
        
        # JSD histograms
        jsd_path = os.path.join(output_dir, f'jsd_histograms_{metrics_str}.png')
        self.plot_jsd_histograms(save_path=jsd_path)
        
        # JSD scatterplots
        scatter_path = os.path.join(output_dir, f'jsd_scatterplots_{metrics_str}.png')
        self.plot_jsd_scatterplots(save_path=scatter_path)
        
        print(f"All figures saved to {output_dir}")


# Example usage:
if __name__ == "__main__":
    # Initialize with data path
    data_path = '/workspace/removing-layer-norm/mech_interp/inference_logs/dataset_apollo-pile_samples_5000_seqlen_512_prepend_True/inference_results.parquet'
    metrics_comparison = MetricsSummary(data_path, min_seq_length=50, agg=True)
    # Generate and save all plots
    metrics_comparison.plot_all('metrics')
