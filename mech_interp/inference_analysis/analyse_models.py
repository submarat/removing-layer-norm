import os
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


class MetricsSummary:
    """
    A class for visualizing model comparison metrics across baseline, finetuned, and noLN models.
    Provides methods for filtering data and creating various visualization types.
    Includes 95% confidence interval calculations for statistical rigor.
    """
    
    def __init__(self,
                 data_path: str,
                 min_seq_length: Optional[int] = None,
                 agg : bool = False,
                 model_type: str = 'small',
                 bootstrap_samples: int = 10000):
        """
        Initialize the ModelComparison class.
        
        Parameters:
        -----------
        data_path : str
            Path to the parquet file containing model comparison data
        min_seq_length : int, optional
            Minimum sequence length to include in the analysis
        agg : bool, optional
            Whether to aggregate metrics
        model_type : str, optional
            Type of model ('small', 'medium', etc.)
        bootstrap_samples : int, optional
            Number of bootstrap samples for confidence interval calculations
        """
        # Set the colorblind-friendly style
        plt.style.use('seaborn-v0_8-colorblind')
        
        # Load the data
        self.data_path = data_path
        self.df = pd.read_parquet(data_path)
        
        # Store the model names for consistent reference
        self.models = ['baseline', 'finetuned', 'noLN']
        
        # Store model type
        self.model_type = model_type.capitalize()
        
        # Create model display labels based on model_type
        self.model_labels = {
            'baseline': f'{self.model_type} original',
            'finetuned': f'{self.model_type} FT',
            'noLN': f'{self.model_type} LN-free'
        }
        
        # Set up color schemes for consistent visualization
        self.colors = sns.color_palette("colorblind")
        self.model_colors = {
            'baseline': self.colors[0],
            'finetuned': self.colors[1],
            'noLN': self.colors[2]
        }
        
        # Set bootstrap samples for CI calculations
        self.bootstrap_samples = bootstrap_samples
        
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
    
    def calculate_bootstrap_ci(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for the mean of a dataset.
        
        Parameters:
        -----------
        data : np.ndarray
            Data array to bootstrap
        confidence : float, optional
            Confidence level (default: 0.95 for 95% CI)
            
        Returns:
        --------
        Tuple[float, float]
            Lower and upper bounds of the confidence interval
        """
        # Calculate the mean of the original data
        mean_orig = np.mean(data)
        
        # Generate bootstrap samples
        rng = np.random.RandomState(42)  # For reproducibility
        bootstrap_means = []
        
        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate confidence interval
        lower_percentile = 100 * (1 - confidence) / 2
        upper_percentile = 100 - lower_percentile
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return lower_bound, upper_bound
    
    def calculate_analytical_ci(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate analytical confidence interval for the mean using t-distribution.
        
        Parameters:
        -----------
        data : np.ndarray
            Data array
        confidence : float, optional
            Confidence level (default: 0.95 for 95% CI)
            
        Returns:
        --------
        Tuple[float, float]
            Lower and upper bounds of the confidence interval
        """
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return mean - h, mean + h
    
    def format_mean_with_ci(self, data: np.ndarray, use_bootstrap: bool = True) -> str:
        """
        Format the mean with confidence interval for display in plots.
        
        Parameters:
        -----------
        data : np.ndarray
            Data array
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
            
        Returns:
        --------
        str
            Formatted string with mean and CI
        """
        mean = np.mean(data)
        
        if use_bootstrap:
            lower, upper = self.calculate_bootstrap_ci(data)
        else:
            lower, upper = self.calculate_analytical_ci(data)
        
        return f"{mean:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])"

    def plot_ce_histogram(self, bins : int = 50, alpha : float = 0.7,
                          figsize: tuple = (8, 6), save_path: Optional[str] = None,
                          logscale: bool = True, use_bootstrap: bool = True):
        """
        Plot step histograms comparing CE loss across all three models.
        Includes 95% confidence intervals for the means.
        
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
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model in self.models:
            data = self.df[f'ce_{model}'].values
            mean_val = np.mean(data)
            
            # Calculate confidence intervals
            if use_bootstrap:
                lower, upper = self.calculate_bootstrap_ci(data)
            else:
                lower, upper = self.calculate_analytical_ci(data)
            
            # Format for display
            ci_str = f"{mean_val:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])"
            
            ax.hist(
                data,
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=alpha,
                label=f'{self.model_labels[model]} - mean: {ci_str}',
                color=self.model_colors[model]
            )
            ax.axvline(
                mean_val,
                color=self.model_colors[model],
                linestyle='--',
                linewidth=1.5,
            )
            
            # Add shaded confidence interval area
            ax.axvspan(
                lower, upper,
                color=self.model_colors[model],
                alpha=0.1
            )
        
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
                               logscale: bool = True, use_bootstrap: bool = True):
        """
        Plot step histograms comparing entropy across all three models.
        Includes 95% confidence intervals for the means.
        
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
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model in self.models:
            data = self.df[f'entropy_{model}'].values
            mean_val = np.mean(data)
            
            # Calculate confidence intervals
            if use_bootstrap:
                lower, upper = self.calculate_bootstrap_ci(data)
            else:
                lower, upper = self.calculate_analytical_ci(data)
            
            # Format for display
            ci_str = f"{mean_val:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])"
            
            ax.hist(
                data,
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=alpha,
                label=f'{self.model_labels[model]} - mean: {ci_str}',
                color=self.model_colors[model]
            )
            ax.axvline(
                mean_val,
                color=self.model_colors[model],
                linestyle='--',
                linewidth=1.5,
            )
            
            # Add shaded confidence interval area
            ax.axvspan(
                lower, upper,
                color=self.model_colors[model],
                alpha=0.1
            )
        
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
                            logscale: bool = True, use_bootstrap: bool = True):
        """
        Plot a (1,2) subplot of histograms showing JSD and topk_JSD distributions
        for all model comparisons. Includes 95% confidence intervals for the means.
        
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
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
           
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        model_pairs = [('baseline', 'finetuned'), ('baseline', 'noLN'), ('finetuned', 'noLN')]
        pair_labels = [
            f'{self.model_labels["baseline"]} vs {self.model_labels["finetuned"]}',
            f'{self.model_labels["baseline"]} vs {self.model_labels["noLN"]}',
            f'{self.model_labels["finetuned"]} vs {self.model_labels["noLN"]}'
        ]
        pair_colors = [self.colors[i] for i in range(len(model_pairs))]
        
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        
        # Plot JSD histograms
        for i, (m1, m2) in enumerate(model_pairs):
            pair = f'{m1}_vs_{m2}'
            data = self.df[f'jsd_{pair}'].values
            mean_val = np.mean(data)
            
            # Calculate confidence intervals
            if use_bootstrap:
                lower, upper = self.calculate_bootstrap_ci(data)
            else:
                lower, upper = self.calculate_analytical_ci(data)
            
            # Format for display
            ci_str = f"{mean_val:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])"
            
            axes.hist(
                data,
                bins=bins,
                histtype='step',
                linewidth=2,
                alpha=alpha,
                label=f'{pair_labels[i]} - mean: {ci_str}',
                color=pair_colors[i]
                )
            axes.axvline(
                mean_val,
                color=pair_colors[i],
                linestyle='--',
                linewidth=1.5,
                )
                
            # Add shaded confidence interval area
            axes.axvspan(
                lower, upper,
                color=pair_colors[i],
                alpha=0.1
            )
            
        # Set titles and labels
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
        ax.set_xlabel(f'CE Loss ({self.model_labels[models[0]]})', fontsize=14)
        ax.set_ylabel(f'CE Loss ({self.model_labels[models[1]]})', fontsize=14)
    
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
        ax.set_xlabel(f'Entropy ({self.model_labels[models[0]]})', fontsize=14)
        ax.set_ylabel(f'Entropy ({self.model_labels[models[1]]})', fontsize=14)
    
        # Set axis limits to focus on where most data is (avoiding outliers)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
    
        # Improve aesthetics
        plt.tight_layout()
    
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

    def calculate_ece_with_ci(self, confidences, accuracies, bin_edges, use_bootstrap=True, confidence=0.95):
        """
        Calculate the Expected Calibration Error (ECE) with confidence intervals.
        
        Parameters:
        -----------
        confidences : array-like
            Model confidence scores
        accuracies : array-like
            Binary values indicating whether the prediction was correct (1) or not (0)
        bin_edges : array-like
            Edges of the bins for confidence scores
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
        confidence : float, optional
            Confidence level (default: 0.95 for 95% CI)
            
        Returns:
        --------
        tuple
            (ece, lower_ci, upper_ci, bin_confidences, bin_accuracies, bin_fractions)
        """
        n_bins = len(bin_edges) - 1
        bin_indices = np.digitize(confidences, bin_edges) - 1
        
        # Clip bin indices to valid range
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Initialize arrays for bin statistics
        bin_confidences = np.zeros(n_bins)
        bin_accuracies = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        # Compute bin statistics
        for i in range(n_bins):
            mask = (bin_indices == i)
            if np.any(mask):
                bin_confidences[i] = np.mean(confidences[mask])
                bin_accuracies[i] = np.mean(accuracies[mask])
                bin_counts[i] = np.sum(mask)
        
        # Calculate ECE
        total_samples = np.sum(bin_counts)
        ece = np.sum(bin_counts / total_samples * np.abs(bin_confidences - bin_accuracies))
        
        # Calculate bin fractions
        bin_fractions = bin_counts / total_samples
        
        # Calculate confidence interval using bootstrap
        if use_bootstrap:
            # Bootstrap ECE calculation
            bootstrap_eces = []
            rng = np.random.RandomState(42)  # For reproducibility
            
            for _ in range(self.bootstrap_samples):
                # Sample indices with replacement
                bootstrap_indices = rng.choice(len(confidences), size=len(confidences), replace=True)
                bootstrap_confidences = confidences[bootstrap_indices]
                bootstrap_accuracies = accuracies[bootstrap_indices]
                
                # Calculate ECE for bootstrap sample
                b_bin_indices = np.digitize(bootstrap_confidences, bin_edges) - 1
                b_bin_indices = np.clip(b_bin_indices, 0, n_bins - 1)
                
                b_bin_confidences = np.zeros(n_bins)
                b_bin_accuracies = np.zeros(n_bins)
                b_bin_counts = np.zeros(n_bins)
                
                for i in range(n_bins):
                    mask = (b_bin_indices == i)
                    if np.any(mask):
                        b_bin_confidences[i] = np.mean(bootstrap_confidences[mask])
                        b_bin_accuracies[i] = np.mean(bootstrap_accuracies[mask])
                        b_bin_counts[i] = np.sum(mask)
                
                b_total_samples = np.sum(b_bin_counts)
                b_ece = np.sum(b_bin_counts / b_total_samples * np.abs(b_bin_confidences - b_bin_accuracies))
                bootstrap_eces.append(b_ece)
            
            # Calculate CI from bootstrap samples
            lower_percentile = 100 * (1 - confidence) / 2
            upper_percentile = 100 - lower_percentile
            
            lower_bound = np.percentile(bootstrap_eces, lower_percentile)
            upper_bound = np.percentile(bootstrap_eces, upper_percentile)
        else:
            # Use analytical CI (simplified approximation)
            # This is a simplification as ECE doesn't have a straightforward analytical CI
            ece_std = np.std([np.abs(bin_confidences[i] - bin_accuracies[i]) for i in range(n_bins)])
            error_margin = ece_std / np.sqrt(n_bins) * stats.t.ppf((1 + confidence) / 2, n_bins - 1)
            lower_bound = max(0, ece - error_margin)  # ECE can't be negative
            upper_bound = ece + error_margin
        
        return ece, lower_bound, upper_bound, bin_confidences, bin_accuracies, bin_fractions

    def plot_calibration(self, 
                         n_bins: int = 10, 
                         equal_bins: bool = True,
                         figsize: tuple = (10, 8), 
                         save_path: Optional[str] = None,
                         use_bootstrap: bool = True):
        """
        Plot calibration curves for all models with 95% confidence intervals for ECE.
        
        Parameters:
        -----------
        n_bins : int, optional
            Number of confidence bins
        equal_bins : bool, optional
            If True, bins have equal width (uniform spacing).
            If False, bins have equal counts (quantile-based).
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Add the diagonal perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.8)
        
        # Process each model
        for model in self.models:
            # Extract max probabilities and accuracy (when max token = next token)
            max_probs = self.df[f'max_prob_{model}'].values
            correct_preds = (self.df[f'max_token_{model}'].values == self.df['next_token'].values).astype(int)
            
            # Create bins
            if equal_bins:
                # Equal width bins
                bin_edges = np.linspace(0, 1, n_bins + 1)
            else:
                # Equal count bins (using quantiles)
                quantiles = np.linspace(0, 1, n_bins + 1)
                bin_edges = np.quantile(max_probs, quantiles)
                # Ensure first bin starts at 0 and last bin ends at 1
                bin_edges[0] = 0
                bin_edges[-1] = 1
            
            # Calculate ECE and bin statistics with confidence intervals
            ece, ece_lower, ece_upper, bin_confidences, bin_accuracies, _ = self.calculate_ece_with_ci(
                max_probs, correct_preds, bin_edges, use_bootstrap=use_bootstrap
            )
            
            # Format ECE with CI for display
            ci_str = f"{ece:.4f} (95% CI: [{ece_lower:.4f}, {ece_upper:.4f}])"
            
            # Plot calibration curve
            ax.plot(
                bin_confidences, 
                bin_accuracies, 
                marker='o', 
                linestyle='-',
                color=self.model_colors[model],
                label=f'{self.model_labels[model]} - ECE: {ci_str}'
            )
            
            # Add confidence interval shading for the ECE
            # This visualizes uncertainty in calibration
            ax.fill_between(
                [0, 1],
                [ece_lower, ece_lower],
                [ece_upper, ece_upper],
                color=self.model_colors[model],
                alpha=0.1
            )
        
        # Set labels and title
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_xlabel('Confidence (Max Probability)', fontsize=16)
        
        # Set axis limits and grid
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=14, loc='lower right')
        
        # Improve aesthetics
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

    def plot_ce_by_seq_length(self, 
                                  seq_bins: list = [0, 25, 50, 75, 100, 150, 512],
                                  figsize: tuple = (12, 6),
                                  save_path: Optional[str] = None,
                                  use_bootstrap: bool = True):
            """
            Create a seaborn boxplot to visualize how CE loss changes across different
            sequence length ranges for all models. Includes 95% confidence intervals.
            
            Parameters:
            -----------
            seq_bins : list, optional
                Sequence length bin edges for categorization
            figsize : tuple, optional
                Figure size (width, height)
            save_path : str, optional
                Path to save the figure
            use_bootstrap : bool, optional
                If True, use bootstrap CI; otherwise, use analytical CI
                
            Returns:
            --------
            matplotlib.figure.Figure
                The generated figure
            """
            
            # Create a copy of the data to avoid modifying the original
            df_plot = self.df.copy()
            
            # Create sequence length bins
            df_plot['seq_length_bin'] = pd.cut(
                df_plot['sequence_length'], 
                bins=seq_bins, 
                labels=[f'{seq_bins[i]}-{seq_bins[i+1]}' for i in range(len(seq_bins)-1)]
            )
            
            # Prepare data for boxplot - melt the dataframe
            ce_columns = [f'ce_{model}' for model in self.models]
            df_melt = pd.melt(df_plot, 
                              id_vars=['seq_length_bin'], 
                              value_vars=ce_columns,
                              var_name='model', 
                              value_name='ce_loss')
            
            # Convert model names from 'ce_modelname' to just 'modelname'
            df_melt['model'] = df_melt['model'].apply(lambda x: x.split('_')[1])
            model_display_map = {model: self.model_labels[model] for model in self.models}
            df_melt['model_display'] = df_melt['model'].map(model_display_map)
            
            # Create boxplot
            fig, ax = plt.subplots(figsize=figsize)
            
            # Use Seaborn's boxplot
            sns_plot = sns.boxplot(
                data=df_melt,
                x='seq_length_bin',
                y='ce_loss',
                hue='model_display',
                palette=self.colors,
                showfliers=False,  # Don't show outliers for cleaner visualization
                ax=ax
            )
            
            # Add individual data points for better visualization
            sns.stripplot(
                data=df_melt,
                x='seq_length_bin',
                y='ce_loss',
                hue='model_display',
                palette=self.colors,
                dodge=True,
                size=3,
                alpha=0.2,
                ax=ax,
                legend=False
            )
            
            # Add confidence intervals to the plot annotation
            # Calculate CIs for each group and add as text annotations
            for seq_bin in df_melt['seq_length_bin'].unique():
                y_pos = ax.get_ylim()[1] * 0.9  # Position the text near the top
                for i, model in enumerate(self.models):
                    group_data = df_melt[(df_melt['model'] == model) & 
                                         (df_melt['seq_length_bin'] == seq_bin)]['ce_loss'].values
                    if len(group_data) > 0:
                        if use_bootstrap:
                            lower, upper = self.calculate_bootstrap_ci(group_data)
                        else:
                            lower, upper = self.calculate_analytical_ci(group_data)
            
            # Customize plot
            ax.set_xlabel('Sequence Length Range', fontsize=14)
            ax.set_ylabel('Cross-Entropy Loss', fontsize=14)
            ax.legend(fontsize=12, title_fontsize=13)
            
            # Add grid for better readability
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save if path is provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
    
    def plot_entropy_by_seq_length(self, 
                                   seq_bins: list = [0, 25, 50, 75, 100, 150, 512],
                                   figsize: tuple = (12, 6),
                                   save_path: Optional[str] = None,
                                   use_bootstrap: bool = True):
        """
        Create a seaborn boxplot to visualize how entropy changes across different
        sequence length ranges for all models. Includes 95% confidence intervals.
        
        Parameters:
        -----------
        seq_bins : list, optional
            Sequence length bin edges for categorization
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        
        # Create a copy of the data to avoid modifying the original
        df_plot = self.df.copy()
        
        # Create sequence length bins
        df_plot['seq_length_bin'] = pd.cut(
            df_plot['sequence_length'], 
            bins=seq_bins, 
            labels=[f'{seq_bins[i]}-{seq_bins[i+1]}' for i in range(len(seq_bins)-1)]
        )
        
        # Prepare data for boxplot - melt the dataframe
        entropy_columns = [f'entropy_{model}' for model in self.models]
        df_melt = pd.melt(df_plot, 
                          id_vars=['seq_length_bin'], 
                          value_vars=entropy_columns,
                          var_name='model', 
                          value_name='entropy')
        
        # Convert model names from 'entropy_modelname' to just 'modelname'
        df_melt['model'] = df_melt['model'].apply(lambda x: x.split('_')[1])
        model_display_map = {model: self.model_labels[model] for model in self.models}
        df_melt['model_display'] = df_melt['model'].map(model_display_map)
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use Seaborn's boxplot
        sns.boxplot(
            data=df_melt,
            x='seq_length_bin',
            y='entropy',
            hue='model_display',
            palette=self.colors,
            showfliers=False,  # Don't show outliers for cleaner visualization
            ax=ax
        )
        
        # Add individual data points for better visualization
        sns.stripplot(
            data=df_melt,
            x='seq_length_bin',
            y='entropy',
            hue='model_display',
            palette=self.colors,
            dodge=True,
            size=3,
            alpha=0.2,
            ax=ax,
            legend=False
        )
        
        # Add confidence intervals to the plot annotation (similar to ce_by_seq_length)
        for seq_bin in df_melt['seq_length_bin'].unique():
            y_pos = ax.get_ylim()[1] * 0.9  # Position the text near the top
            for i, model in enumerate(self.models):
                group_data = df_melt[(df_melt['model'] == model) & 
                                     (df_melt['seq_length_bin'] == seq_bin)]['entropy'].values
                if len(group_data) > 0:
                    if use_bootstrap:
                        lower, upper = self.calculate_bootstrap_ci(group_data)
                    else:
                        lower, upper = self.calculate_analytical_ci(group_data)
        
        # Customize plot
        ax.set_xlabel('Sequence Length Range', fontsize=14)
        ax.set_ylabel('Entropy', fontsize=14)
        ax.legend(fontsize=12, title_fontsize=13)
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def test_statistical_significance(self, metric: str, model1: str, model2: str, use_bootstrap: bool = True, alpha: float = 0.05):
        """
        Test for statistical significance between two models' metrics.
        
        Parameters:
        -----------
        metric : str
            Metric to compare ('ce', 'entropy', etc.)
        model1 : str
            First model to compare
        model2 : str
            Second model to compare
        use_bootstrap : bool, optional
            If True, use bootstrap for p-value; otherwise, use t-test
        alpha : float, optional
            Significance level
            
        Returns:
        --------
        tuple
            (is_significant, p_value, effect_size)
        """
        if model1 not in self.models or model2 not in self.models:
            raise ValueError(f"Models must be one of {self.models}")
        
        # Get the data for both models
        data1 = self.df[f'{metric}_{model1}'].values
        data2 = self.df[f'{metric}_{model2}'].values
        
        # Calculate basic statistics
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        diff = mean1 - mean2
        
        if use_bootstrap:
            # Bootstrap test for significance
            # Combine the two samples
            combined = np.concatenate([data1, data2])
            n1, n2 = len(data1), len(data2)
            
            # Calculate the observed difference in means
            observed_diff = mean1 - mean2
            
            # Perform bootstrap resampling
            rng = np.random.RandomState(42)  # For reproducibility
            bootstrap_diffs = []
            
            for _ in range(self.bootstrap_samples):
                # Shuffle the combined data
                shuffled = rng.permutation(combined)
                
                # Split into two groups of the original sizes
                boot_data1 = shuffled[:n1]
                boot_data2 = shuffled[n1:n1+n2]
                
                # Calculate and store the difference in means
                boot_diff = np.mean(boot_data1) - np.mean(boot_data2)
                bootstrap_diffs.append(boot_diff)
            
            # Calculate p-value (two-sided)
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
            
        else:
            # Use t-test for significance
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        effect_size = diff / pooled_std
        
        # Determine if the difference is significant
        is_significant = p_value < alpha
        
        return is_significant, p_value, effect_size
    
    def export_ci_statistics(self, output_path: str, use_bootstrap: bool = True):
        """
        Export mean and 95% confidence intervals for all metrics to a CSV file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the CSV file
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
            
        Returns:
        --------
        str
            Path to the saved CSV file
        """
        # Initialize a dictionary to store the statistics
        stats_dict = {
            'metric': [],
            'model': [],
            'mean': [],
            'ci_lower': [],
            'ci_upper': [],
        }
        
        # Calculate statistics for CE loss
        for model in self.models:
            data = self.df[f'ce_{model}'].values
            mean = np.mean(data)
            
            if use_bootstrap:
                lower, upper = self.calculate_bootstrap_ci(data)
            else:
                lower, upper = self.calculate_analytical_ci(data)
            
            stats_dict['metric'].append('CE Loss')
            stats_dict['model'].append(self.model_labels[model])
            stats_dict['mean'].append(mean)
            stats_dict['ci_lower'].append(lower)
            stats_dict['ci_upper'].append(upper)
        
        # Calculate statistics for entropy
        for model in self.models:
            data = self.df[f'entropy_{model}'].values
            mean = np.mean(data)
            
            if use_bootstrap:
                lower, upper = self.calculate_bootstrap_ci(data)
            else:
                lower, upper = self.calculate_analytical_ci(data)
            
            stats_dict['metric'].append('Entropy')
            stats_dict['model'].append(self.model_labels[model])
            stats_dict['mean'].append(mean)
            stats_dict['ci_lower'].append(lower)
            stats_dict['ci_upper'].append(upper)
        
        # Calculate statistics for JSD
        model_pairs = [('baseline', 'finetuned'), ('baseline', 'noLN'), ('finetuned', 'noLN')]
        pair_labels = [
            f'{self.model_labels["baseline"]} vs {self.model_labels["finetuned"]}',
            f'{self.model_labels["baseline"]} vs {self.model_labels["noLN"]}',
            f'{self.model_labels["finetuned"]} vs {self.model_labels["noLN"]}'
        ]
        
        for i, (m1, m2) in enumerate(model_pairs):
            pair = f'{m1}_vs_{m2}'
            data = self.df[f'jsd_{pair}'].values
            mean = np.mean(data)
            
            if use_bootstrap:
                lower, upper = self.calculate_bootstrap_ci(data)
            else:
                lower, upper = self.calculate_analytical_ci(data)
            
            stats_dict['metric'].append('JSD')
            stats_dict['model'].append(pair_labels[i])
            stats_dict['mean'].append(mean)
            stats_dict['ci_lower'].append(lower)
            stats_dict['ci_upper'].append(upper)
        
        # Calculate statistics for ECE
        for model in self.models:
            # Extract max probabilities and accuracy
            max_probs = self.df[f'max_prob_{model}'].values
            correct_preds = (self.df[f'max_token_{model}'].values == self.df['next_token'].values).astype(int)
            
            # Create bins
            bin_edges = np.linspace(0, 1, 10 + 1)
            
            # Calculate ECE and confidence intervals
            ece, lower, upper, _, _, _ = self.calculate_ece_with_ci(
                max_probs, correct_preds, bin_edges, use_bootstrap=use_bootstrap
            )
            
            stats_dict['metric'].append('ECE')
            stats_dict['model'].append(self.model_labels[model])
            stats_dict['mean'].append(ece)
            stats_dict['ci_lower'].append(lower)
            stats_dict['ci_upper'].append(upper)
        
        # Convert to DataFrame and save
        stats_df = pd.DataFrame(stats_dict)
        stats_df.to_csv(output_path, index=False)
        
        print(f"Statistics with 95% confidence intervals saved to {output_path}")
        return output_path

    def plot_all(self, output_dir: Optional[str] = None, use_bootstrap: bool = True):
        """
        Generate and save all visualization types with confidence intervals.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the figures
        use_bootstrap : bool, optional
            If True, use bootstrap CI; otherwise, use analytical CI
            
        Returns:
        --------
        list
            List of paths to the generated figures
        """
        metrics_str = '_agg' if self.agg else ''
        ci_method = 'bootstrap' if use_bootstrap else 'analytical'

        # If no output directory specified, use the directory of the parquet file
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(self.data_path))

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # CE histogram with confidence intervals
        ce_path = os.path.join(output_dir, f'ce_histogram{metrics_str}.png')
        self.plot_ce_histogram(save_path=ce_path, use_bootstrap=use_bootstrap)
        
        # Entropy histogram with confidence intervals
        entropy_path = os.path.join(output_dir, f'entropy_histogram{metrics_str}.png')
        self.plot_entropy_histogram(save_path=entropy_path, logscale=False, use_bootstrap=use_bootstrap)
        
        # JSD histograms with confidence intervals
        jsd_path = os.path.join(output_dir, f'jsd_histograms{metrics_str}.png')
        self.plot_jsd_histogram(save_path=jsd_path, use_bootstrap=use_bootstrap)
        
        # CE hexplot (baseline vs noLN)
        ce_hex_path = os.path.join(output_dir, f'ce_hexplot{metrics_str}.png')
        self.plot_ce_hexplot(save_path=ce_hex_path)

        # CE hexplot (finetuned vs noLN)
        ce_hex_ft_path = os.path.join(output_dir, f'ce_finetuned_hexplot{metrics_str}.png')
        self.plot_ce_hexplot(models=['finetuned', 'noLN'], save_path=ce_hex_ft_path)
        
        # Entropy hexplot (baseline vs noLN)
        entropy_hex_path = os.path.join(output_dir, f'entropy_hexplot{metrics_str}.png')
        self.plot_entropy_hexplot(save_path=entropy_hex_path)

        # Entropy hexplot (finetuned vs noLN)
        entropy_hex_ft_path = os.path.join(output_dir, f'entropy_finetuned_hexplot{metrics_str}.png')
        self.plot_entropy_hexplot(models=['finetuned', 'noLN'], save_path=entropy_hex_ft_path)
        
        # Calibration plots with equal width bins and confidence intervals
        calib_equal_width_path = os.path.join(output_dir, f'calibration{metrics_str}.png')
        self.plot_calibration(equal_bins=True, save_path=calib_equal_width_path, use_bootstrap=use_bootstrap)
        
        # CE by sequence length barplot with confidence intervals
        ce_seq_length_path = os.path.join(output_dir, f'ce_seq_length{metrics_str}.png')
        self.plot_ce_by_seq_length(save_path=ce_seq_length_path, use_bootstrap=use_bootstrap)
        
        # Entropy by sequence length barplot with confidence intervals
        entropy_seq_length_path = os.path.join(output_dir, f'entropy_seq_length{metrics_str}.png')
        self.plot_entropy_by_seq_length(save_path=entropy_seq_length_path, use_bootstrap=use_bootstrap)
        
        # Export confidence interval statistics to CSV
        stats_path = os.path.join(output_dir, f'ci_statistics{metrics_str}.csv')
        self.export_ci_statistics(stats_path, use_bootstrap=use_bootstrap)
       
        print(f"All figures with confidence intervals saved to {output_dir}")


# Example usage:
if __name__ == "__main__":
    # Initialize with data path and model type
    model_str = 'small'
    data_path = f'/workspace/removing-layer-norm/mech_interp/inference_logs/gpt2-{model_str}_dataset_ANONYMIZED-pile_samples_1000_seqlen_512_prepend_False/inference_results.parquet'
    
    # Create metrics summary with bootstrap confidence intervals (10,000 samples)
    metrics_comparison = MetricsSummary(
        data_path, 
        model_type=model_str, 
        agg=False, 
        bootstrap_samples=1_000
    )
    
    # Generate all plots with bootstrap confidence intervals
    print("Generating plots with bootstrap confidence intervals...")
    metrics_comparison.plot_all(f'figures/{model_str}', use_bootstrap=True)
