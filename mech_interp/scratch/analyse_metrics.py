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


    def get_aggregate_metrics(self, agg_metric='ce_baseline'):
        """
        Aggregate all subsequences metrics of original sequence to one, based on the highest agg_metric

        Parameters:
        --------
        agg_metric : column name in dataframe to use as reference for aggregation
        
        Returns:
        --------
        Aggregated df
        """
        # Ensure the agg_metric exists in the dataframe
        if agg_metric not in self.df.columns:
            raise ValueError(f"Column '{agg_metric}' not found in dataframe")

        # Function to select the row with maximum value in the specified column
        def select_max_row(group):
            return group.loc[group[agg_metric].idxmin()]

        # For each original_idx, select the row with the maximum value in the specified column
        aggregated_df = self.df.groupby('original_idx').apply(select_max_row).reset_index(drop=True)

        print(f"Original shape: {self.df.shape}")
        print(f"After aggregation: {aggregated_df.shape}")

        self.df = aggregated_df
       

    def plot_ce_histogram(self, bins : int = 50, alpha : float = 0.7,
                          figsize: tuple = (8, 6), save_path: Optional[str] = None,
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
            mean_val = self.df[f'ce_{model}'].mean()
            ax.hist(
                self.df[f'ce_{model}'],
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=alpha,
                label=f'{model}, mean: {mean_val:.3f}',
                color=self.model_colors[model]
            )
            ax.axvline(
                mean_val,
                color=self.model_colors[model],
                linestyle='--',
                linewidth=1.5,
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
                            figsize: tuple = (8, 6), save_path: Optional[str] = None,
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
        
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        
        # Plot JSD histograms
        for i, (m1, m2) in enumerate(model_pairs):
            pair = f'{m1}_vs_{m2}'
            mean_val = self.df[f'jsd_{pair}'].mean()
            axes.hist(
                self.df[f'jsd_{pair}'],
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=alpha,
                label=f'{pair}, mean: {mean_val:.3f}',
                color=pair_colors[i]
                )
            axes.axvline(
                mean_val,
                color=pair_colors[i],
                linestyle='--',
                linewidth=1.5,
                )
        # Set titles and labels
        axes.set_title('Jensen-Shannon Divergence', fontsize=16)
        axes.set_xlabel('JSD', fontsize=14)
        axes.set_ylabel('Count', fontsize=14)
        axes.legend(fontsize=12)
        axes.grid(alpha=0.3)
        
        if logscale:
            axes.set_yscale('log')
            axes.set_ylabel('Count (log scale)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


    def plot_jsd_scatterplots(self, figsize: tuple = (8, 6),
                              alpha: float = 0.6, save_path: Optional[str] = None):
            """
            Plot a scatterplot comparing JSD
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
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            
            # Create a shared color normalization based on noLN CE loss
            if self.agg:
                vmax = self.df['ce_finetuned'].max()
            else:
                vmax=np.percentile(self.df['ce_finetuned'], 95)  # Cap at 95th percentile to avoid outliers
            norm = Normalize(vmin=self.df['ce_finetuned'].min(), vmax=vmax)
            
            # JSD scatterplot
            sc1 = axes.scatter(
                self.df['jsd_baseline_vs_finetuned'],
                self.df['jsd_baseline_vs_noLN'],
                c=self.df['ce_noLN'],
                cmap='viridis',
                alpha=alpha,
                norm=norm,
                s=25
            )
            
            # Add diagonal line for reference
            axes.plot([0, self.df['jsd_baseline_vs_finetuned'].max()],
                      [0, self.df['jsd_baseline_vs_noLN'].max()], 'k--', alpha=0.5)

           
            # Set titles and labels
            axes.set_title('Jensen-Shannon Divergence Comparison', fontsize=16)
            axes.set_xlabel('JSD: baseline vs finetuned', fontsize=14)
            axes.set_ylabel('JSD: baseline vs noLN', fontsize=14)
            axes.grid(alpha=0.3)
            
            # Add a shared colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(sc1, cax=cbar_ax)
            cbar.set_label('CE Loss (noLN model)', fontsize=14)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig    



    def plot_ce_scatterplots(self, figsize: tuple = (8, 6),
                                  alpha: float = 0.6, save_path: Optional[str] = None):
            """
            Plot a scatterplot comparing CE loss
            between baseline and noLN.
            Points are colored by CE loss of finetuned model.
            
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
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            
            # Create a shared color normalization based on finetuned CE loss
            if self.agg:
                vmax = self.df['ce_finetuned'].max()
            else:
                vmax=np.percentile(self.df['ce_finetuned'], 95)  # Cap at 95th percentile to avoid outliers
            norm = Normalize(vmin=self.df['ce_finetuned'].min(), vmax=vmax)
            
            # CE Loss scatterplot
            sc1 = axes.scatter(
                self.df['ce_baseline'],
                self.df['ce_noLN'],
                c=self.df['ce_finetuned'],
                cmap='viridis',
                alpha=alpha,
                norm=norm,
                s=25
            )
            
            # Add diagonal line for reference
            axes.plot([0, self.df['ce_baseline'].max()],
                      [0, self.df['ce_noLN'].max()], 'k--', alpha=0.5)
           
            # Set titles and labels
            axes.set_title('CE Loss comparison Comparison', fontsize=16)
            axes.set_xlabel('CE Loss (baseline)', fontsize=14)
            axes.set_ylabel('CE Loss (noLN)', fontsize=14)
            axes.grid(alpha=0.3)
            
            # Add a shared colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(sc1, cax=cbar_ax)
            cbar.set_label('CE Loss (finetuned model)', fontsize=14)
            
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
        metrics_str = '_agg' if self.agg else ''

        # If no output directory specified, use the directory of the parquet file
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(self.data_path))

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # CE histogram
        ce_path = os.path.join(output_dir, f'ce_histogram{metrics_str}.png')
        self.plot_ce_histogram(save_path=ce_path)
        
        # JSD histograms
        jsd_path = os.path.join(output_dir, f'jsd_histograms{metrics_str}.png')
        self.plot_jsd_histograms(save_path=jsd_path)
        
        # JSD scatterplots
        jsd_scatter_path = os.path.join(output_dir, f'jsd_scatterplots{metrics_str}.png')
        self.plot_jsd_scatterplots(save_path=jsd_scatter_path)
        
        # CE scatterplots
        ce_scatter_path = os.path.join(output_dir, f'ce_scatterplots{metrics_str}.png')
        self.plot_ce_scatterplots(save_path=ce_scatter_path)
       
        
        print(f"All figures saved to {output_dir}")


# Example usage:
if __name__ == "__main__":
    # Initialize with data path
    data_path = '/workspace/removing-layer-norm/mech_interp/inference_logs/dataset_luca-pile_samples_5000_seqlen_512_prepend_False/inference_results.parquet'
    metrics_comparison = MetricsSummary(data_path, agg=False)
    # Generate and save all plots
    metrics_comparison.plot_all('metrics')
