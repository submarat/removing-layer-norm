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
    
    
    def plot_entropy_histogram(self, bins : int = 50, alpha : float = 0.7,
                               figsize: tuple = (8, 6), save_path: Optional[str] = None,
                               logscale: bool = True):
        """
        Plot step histograms comparing entropy across all three models.
        
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
            mean_val = self.df[f'entropy_{model}'].mean()
            ax.hist(
                self.df[f'entropy_{model}'],
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
        
        ax.set_title('Entropy Distribution Across Models', fontsize=16)
        ax.set_xlabel('Entropy', fontsize=14)
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


    def plot_jsd_histogram(self, bins: int = 50, alpha: float = 0.7,
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


    def plot_ce_hexplot(self,
                        models: List = ['baseline', 'noLN'],
                        figsize: tuple = (8, 6),
                        sample_size: Optional[int] = None, 
                        save_path: Optional[str] = None):

        """
        Plot a hexbin plot comparing CE loss between baseline and noLN.
        
        Parameters:
        -----------
        models : list,
            Models to analyse
        figsize : tuple, optional
            Figure size (width, height)
        sample_size : int, optional
            Number of points to sample for faster rendering
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
           # Sample data if requested
        if sample_size and len(self.df) > sample_size:
            print(f"Sampling {sample_size} points from {len(self.df)} total points")
            sampled_df = self.df.sample(n=sample_size, random_state=42)
        else:
            sampled_df = self.df
        
        fig, ax = plt.subplots(figsize=figsize)

        # Create the hexbin plot directly with matplotlib (for more control)
        hb = ax.hexbin(
            sampled_df[f'ce_{models[0]}'],
            sampled_df[f'ce_{models[1]}'],
            gridsize=50,
            cmap='viridis',
            bins='log',  # Use logarithmic binning
            mincnt=1     # Show bins with at least 1 point
        )
    
        # Add colorbar
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('Count (log scale)')
    
        # Get min/max values with some padding but avoid extreme outliers
        x_max = np.percentile(sampled_df[f'ce_{models[0]}'], 99.5) * 1.05
        y_max = np.percentile(sampled_df[f'ce_{models[1]}'], 99.5) * 1.05
        max_lim = max(x_max, y_max)
    
        # Add diagonal line for reference
        ax.plot([0, max_lim], [0, max_lim], 'r--', alpha=0.7, 
                linewidth=1.5, zorder=10)
    
        # Set titles and labels
        ax.set_xlabel(f'CE Loss ({models[0]})', fontsize=14)
        ax.set_ylabel(f'CE Loss ({models[1]})', fontsize=14)
    
        # Set axis limits to focus on where most data is (avoiding outliers)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
    
        # Improve aesthetics
        plt.tight_layout()
    
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig    

    
    def plot_entropy_hexplot(self,
                             models: List = ['baseline', 'noLN'],
                             figsize: tuple = (8, 6),
                             sample_size: Optional[int] = None, 
                             save_path: Optional[str] = None):

        """
        Plot a hexbin plot comparing entropies between baseline and noLN.
        
        Parameters:
        -----------
         models : list,
            Models to analyse
        figsize : tuple, optional
            Figure size (width, height)
        sample_size : int, optional
            Number of points to sample for faster rendering
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
           # Sample data if requested
        if sample_size and len(self.df) > sample_size:
            print(f"Sampling {sample_size} points from {len(self.df)} total points")
            sampled_df = self.df.sample(n=sample_size, random_state=42)
        else:
            sampled_df = self.df
        
        fig, ax = plt.subplots(figsize=figsize)

        # Create the hexbin plot directly with matplotlib (for more control)
        hb = ax.hexbin(
            sampled_df[f'entropy_{models[0]}'],
            sampled_df[f'entropy_{models[1]}'],
            gridsize=50,
            cmap='viridis',
            bins='log',  # Use logarithmic binning
            mincnt=1     # Show bins with at least 1 point
        )
    
        # Add colorbar
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('Count (log scale)')
    
        # Get min/max values with some padding but avoid extreme outliers
        x_max = np.percentile(sampled_df[f'entropy_{models[0]}'], 99.5) * 1.05
        y_max = np.percentile(sampled_df[f'entropy_{models[1]}'], 99.5) * 1.05
        max_lim = max(x_max, y_max)
    
        # Add diagonal line for reference
        ax.plot([0, max_lim], [0, max_lim], 'r--', alpha=0.7, 
                linewidth=1.5, zorder=10)
    
        # Set titles and labels
        ax.set_xlabel(f'Entropy ({models[0]})', fontsize=14)
        ax.set_ylabel(f'Entropy ({models[1]})', fontsize=14)
    
        # Set axis limits to focus on where most data is (avoiding outliers)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
    
        # Improve aesthetics
        plt.tight_layout()
    
        # Save if path is provided
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
        
        # Entropy histogram
        entropy_path = os.path.join(output_dir, f'entropy_histogram{metrics_str}.png')
        self.plot_entropy_histogram(save_path=entropy_path, logscale=False)
        
        # JSD histograms
        jsd_path = os.path.join(output_dir, f'jsd_histograms{metrics_str}.png')
        self.plot_jsd_histogram(save_path=jsd_path)
        
        # CE hexplot (baseline vs noLN)
        ce_hex_path = os.path.join(output_dir, f'ce_hexplot{metrics_str}.png')
        self.plot_ce_hexplot(save_path=ce_hex_path)

         # CE hexplot (finetuned vs noLN)
        ce_hex_path = os.path.join(output_dir, f'ce_finetuned_hexplot{metrics_str}.png')
        self.plot_ce_hexplot(models=['finetuned', 'noLN'], save_path=ce_hex_path)
        
        # Entropy hexplot (baseline vs noLN)
        entropy_hex_path = os.path.join(output_dir, f'entropy_hexplot{metrics_str}.png')
        self.plot_entropy_hexplot(save_path=entropy_hex_path)

        # Entropy hexplot (finetuned vs noLN)
        entropy_hex_path = os.path.join(output_dir, f'entropy_finetuned_hexplot{metrics_str}.png')
        self.plot_entropy_hexplot(models=['finetuned', 'noLN'], save_path=entropy_hex_path)
       
        print(f"All figures saved to {output_dir}")


# Example usage:
if __name__ == "__main__":
    # Initialize with data path
    data_path = '/workspace/removing-layer-norm/mech_interp/inference_logs/dataset_luca-pile_samples_1000_seqlen_512_prepend_False/inference_results.parquet'
    metrics_comparison = MetricsSummary(data_path, agg=False)
    # Generate and save all plots
    metrics_comparison.plot_all('figures')
