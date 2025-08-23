import os
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


class DivergenceSummary:
    """
    A class for visualizing model comparison metrics across baseline, finetuned, and noLN models.
    Provides methods for filtering data and creating various visualization types.
    """
    
    def __init__(self,
                 data_path: str,
                 min_seq_length: Optional[int] = None,
                 agg: bool = False,
                 model_type: str = 'Small'):
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


    def analyze_ce_loss_regions(self, 
                               models: List = ['baseline', 'noLN'],
                               epsilon: float = 0.5,
                               sample_size: Optional[int] = None):
        """
        Analyze CE loss by splitting data into three regions and calculating
        the absolute contribution to total delta from each region.
        
        Parameters:
        -----------
        models : list
            Models to analyze [baseline_model, comparison_model]
        epsilon : float
            Tolerance for the diagonal band width
        sample_size : int, optional
            Number of points to sample for analysis
        
        Returns:
        --------
        dict
            Dictionary containing analysis results for each region
        """
        # Sample data if requested
        if sample_size and len(self.df) > sample_size:
            print(f"Sampling {sample_size} points from {len(self.df)} total points")
            data = self.df.sample(n=sample_size, random_state=42)
        else:
            data = self.df
            
        total_points = len(data)

        # Extract losses for both models
        base_loss = data[f'ce_{models[0]}'].values
        comp_loss = data[f'ce_{models[1]}'].values
        
        # Calculate point-wise differences
        point_diffs = comp_loss - base_loss
        
        # Define regions
        diagonal_mask = np.abs(point_diffs) <= epsilon
        upper_mask = point_diffs > epsilon
        lower_mask = point_diffs < -epsilon
        
        # Calculate total delta between models (should equal sum of all point differences)
        total_delta  = np.sum(point_diffs) / total_points
        
        # Define regions dictionary
        regions = {
            "diagonal_band": diagonal_mask,
            "upper_triangular": upper_mask,
            "lower_triangular": lower_mask
        }
        
        results = {}
        
        # Calculate absolute contribution and percentage for each region
        for region_name, mask in regions.items():
            # Sum of differences in this region
            region_delta = np.sum(point_diffs[mask]) / total_points
            
            # Percentage of total delta
            pct_of_total_delta = 100 * region_delta / total_delta if total_delta != 0 else 0
            
            # Number of points in this region
            region_count = np.sum(mask)
            
            results[region_name] = {
                "count": int(region_count),
                "percentage_of_points": 100 * region_count / total_points,
                "absolute_delta_contribution": float(region_delta),
                "percentage_of_total_delta": float(pct_of_total_delta),
                "avg_point_diff": float(np.mean(point_diffs[mask])) if region_count > 0 else 0
            }
        
        # Add summary data
        results["summary"] = {
            "total_points": len(base_loss),
            "total_delta": float(total_delta),
            "avg_delta_per_point": float(np.mean(point_diffs)),
            "epsilon": epsilon
        }
        
        return results

    def print_region_analysis(self, results):
        """
        Print a formatted summary of the region analysis results
        focused on absolute delta contributions.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from analyze_ce_loss_regions
        """
        summary = results["summary"]
        
        print(f"\n{'='*80}")
        print(f"CE Loss Delta Analysis")
        print(f"{'='*80}")
        
        print(f"\nTotal points: {summary['total_points']:,}")
        print(f"Total delta: {summary['total_delta']:.4f}")
        print(f"Average delta per point: {summary['avg_delta_per_point']:.4f}")
        print(f"Epsilon (diagonal band width): {summary['epsilon']}")
        
        print(f"\n{'-'*80}")
        print(f"Delta Contribution by Region:")
        print(f"{'-'*80}")
        
        # Print header
        headers = ["Region", "Absolute Delta", "% of Total Delta", "% of Points", "Avg Diff/Point"]
        col_widths = [max(len(h), 20) for h in headers]
        header_fmt = "  ".join([f"{{:{w}}}" for w in col_widths])
        
        print(header_fmt.format(*headers))
        print("  ".join(["-" * w for w in col_widths]))
        
        # Print region data
        for region_name in ["diagonal_band", "upper_triangular", "lower_triangular"]:
            region = results[region_name]
            
            row = [
                region_name.replace("_", " ").title(),
                f"{region['absolute_delta_contribution']:.4f}",
                f"{region['percentage_of_total_delta']:.2f}%",
                f"{region['percentage_of_points']:.2f}%",
                f"{region['avg_point_diff']:.4f}"
            ]
            
            print(header_fmt.format(*row))
        
        # Verification
        delta_sum = sum(results[r]["absolute_delta_contribution"] for r in ["diagonal_band", "upper_triangular", "lower_triangular"])
        print(f"\nSum of regional deltas: {delta_sum:.4f} (should equal total delta)")
        
        print(f"\n{'-'*80}")
        print("Interpretation:")
        print(f"• Positive delta means comparison model has higher loss than baseline")
        print(f"• 'Absolute Delta' shows each region's contribution to the total delta")
        print(f"• The sum of all regional absolute deltas equals the total delta between models")
        print(f"• '% of Total Delta' shows what portion of the overall difference comes from each region")
        print(f"{'-'*80}\n")

    def visualize_loss_regions(self, 
                              models: List = ['baseline', 'noLN'],
                              epsilon: float = 0.5,
                              figsize: tuple = (10, 8),
                              sample_size: Optional[int] = None,
                              save_path: Optional[str] = None):
        """
        Visualize the three regions (diagonal, upper triangular, lower triangular) 
        and display their absolute contributions to the total delta.
        
        Parameters:
        -----------
        models : list
            Models to analyze [baseline_model, comparison_model]
        epsilon : float
            Tolerance for the diagonal band width
        figsize : tuple
            Figure size (width, height)
        sample_size : int, optional
            Number of points to sample for analysis
        save_path : str, optional
            Path to save the figure
        
        Returns:
        --------
        tuple
            (matplotlib.figure.Figure, dict) containing the figure and results
        """
        # Sample data as before
        if sample_size and len(self.df) > sample_size:
            data = self.df.sample(n=sample_size, random_state=42)
        else:
            data = self.df
        
        # Extract losses
        x_loss = data[f'ce_{models[0]}'].values
        y_loss = data[f'ce_{models[1]}'].values
        
        # Define regions
        diff = y_loss - x_loss
        diagonal_mask = np.abs(diff) < epsilon
        upper_mask = diff >= epsilon
        lower_mask = diff <= -epsilon
        
        # Get analysis results with absolute delta contributions
        results = self.analyze_ce_loss_regions(models, epsilon, sample_size)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create enhanced labels with percentage information
        diagonal_region = results["diagonal_band"]
        upper_region = results["upper_triangular"]
        lower_region = results["lower_triangular"]
        
        diagonal_label = f'Diagonal Loss ({diagonal_region["percentage_of_points"]:.1f}% of data): {diagonal_region["absolute_delta_contribution"]:.3f} ({diagonal_region["percentage_of_total_delta"]:.1f}% of Δ)'
        upper_label = f'Upper Loss ({upper_region["percentage_of_points"]:.1f}% of data): {upper_region["absolute_delta_contribution"]:.3f} ({upper_region["percentage_of_total_delta"]:.1f}% of Δ)'
        lower_label = f'Lower Loss ({lower_region["percentage_of_points"]:.1f}% of data): {lower_region["absolute_delta_contribution"]:.3f} ({lower_region["percentage_of_total_delta"]:.1f}% of Δ)'
        
        # Plot points in different colors by region
        scatter_size = 5
        scatter_alpha = 0.5
        ax.scatter(x_loss[diagonal_mask], y_loss[diagonal_mask], s=scatter_size, alpha=scatter_alpha, 
                  c=self.colors[0], label=diagonal_label)
        ax.scatter(x_loss[upper_mask], y_loss[upper_mask], s=scatter_size, alpha=scatter_alpha, 
                  c=self.colors[1], label=upper_label)
        ax.scatter(x_loss[lower_mask], y_loss[lower_mask], s=scatter_size, alpha=scatter_alpha, 
                  c=self.colors[2], label=lower_label)
        
        # Get limits consistently with the hexbin plot
        x_max = np.percentile(x_loss, 99.5) * 1.05
        y_max = np.percentile(y_loss, 99.5) * 1.05
        max_lim = max(x_max, y_max)
        
        # Add diagonal reference line
        ax.plot([0, max_lim], [0, max_lim], 'k--', alpha=0.7)
        
        # Add diagonal band lines
        ax.plot([0, max_lim], [epsilon, max_lim + epsilon], 'k:', alpha=0.5)
        ax.plot([0, max_lim], [-epsilon, max_lim - epsilon], 'k:', alpha=0.5)
        
        # Labels and title
        ax.set_xlabel(f'CE Loss ({self.model_labels[models[0]]})', fontsize=12)
        ax.set_ylabel(f'CE Loss ({self.model_labels[models[1]]})', fontsize=12)
        #ax.set_title(f'CE Loss Comparison: {models[1]} vs {models[0]}', fontsize=14)
        
        # Set axis limits
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        
        # Add legend
        ax.legend(loc='lower right', fontsize=14)
        
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, results


# Example usage:
if __name__ == "__main__":
    # Initialize with data path
    model_str = 'small'
    fig_path = f'figures/{model_str}/divergence_contribution.png'
    data_path = f'/workspace/removing-layer-norm/mech_interp/inference_logs/gpt2-{model_str}_dataset_ANONYMIZED-pile_samples_1000_seqlen_512_prepend_False/inference_results.parquet'
    divergences = DivergenceSummary(data_path, agg=False, model_type=model_str)
    # Generate and save all plots
    fig, results = divergences.visualize_loss_regions(epsilon=3.5, save_path=fig_path)
    fig.show()
