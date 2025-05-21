import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any, Union
import re
from transformers import AutoTokenizer


class DivergenceAnalyzer:
    """
    A class to analyze and identify interesting examples in model divergence data.
    This specifically focuses on finding examples where a noLN model diverges
    from baseline and finetuned models, while baseline and finetuned remain similar.
    """
    
    def __init__(self, data_path: str, min_seq_length: Optional[int] = None, agg: bool = False):
        """
        Initialize the analyzer by loading data from a parquet file and setting up tokenizer.
        
        Args:
            data_path: Path to the parquet file containing divergence data
            min_seq_length: Minimum sequence length to include in analysis (optional)
            agg: Whether to aggregate metrics (default: False)
        """
        
        # Load data from parquet file
        self.df = pd.read_parquet(data_path)

        # Apply sequence length filter if provided
        if min_seq_length is not None:
            self.filter_by_sequence_length(min_seq_length)

        # Aggregate metrics if requested
        self.agg = agg
        if self.agg:
            self.get_aggregate_metrics()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Initialize containers for different example types
        self.high_divergence_examples = None
        self.low_divergence_examples = None
        
        # Map of special whitespace characters to their visible representations
        self.whitespace_map = {
            '\r': '\\r',
            '\t': '\\t',
            '\x0c': '\\x0c',
            '\n': '\\n',
            '\n\n': '\\n\\n',
            '\u200b': '\\u200b',
            '\xa0': '\\xa0',
            '\u2002': '\\u2002',
            '\u3000': '\\u3000',
            '\x0b': '\\x0b',
            '\u200e': '\\u200e',
        }


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
        

    def find_interesting_examples(self, 
                                 x_max_threshold: float = 0.1,
                                 y_min_threshold: float = 0.1) -> pd.DataFrame:
        """
        Find examples where noLN diverges but baseline and finetuned are similar.
        
        Args:
            x_max_threshold: Maximum JSD between baseline and finetuned
            y_min_threshold: Minimum JSD between baseline and noLN
            
        Returns:
            DataFrame containing filtered examples, sorted by highest divergence
        """
        self.high_divergence_examples = self.df[
            (self.df['jsd_baseline_vs_finetuned'] <= x_max_threshold) &
            (self.df['jsd_baseline_vs_noLN'] >= y_min_threshold)
        ]
        
        # Return sorted by divergence with noLN
        return self.high_divergence_examples.sort_values('jsd_baseline_vs_noLN', ascending=False)


    def find_low_divergence_examples(self, max_jsd_threshold: float = 0.01) -> pd.DataFrame:
        """
        Find examples where all three models have very similar behavior.
        
        Args:
            max_jsd_threshold: Maximum JSD between any pair of models
            
        Returns:
            DataFrame containing examples with consistently low divergence
        """
        self.low_divergence_examples = self.df[
            (self.df['jsd_baseline_vs_finetuned'] <= max_jsd_threshold) &
            (self.df['jsd_baseline_vs_noLN'] <= max_jsd_threshold) &
            (self.df['jsd_finetuned_vs_noLN'] <= max_jsd_threshold)
        ]
        
        # Sum all JSDs to rank examples
        self.low_divergence_examples['total_jsd'] = (
            self.low_divergence_examples['jsd_baseline_vs_finetuned'] + 
            self.low_divergence_examples['jsd_baseline_vs_noLN'] +
            self.low_divergence_examples['jsd_finetuned_vs_noLN']
        )
        
        return self.low_divergence_examples.sort_values('total_jsd')
    
    
    def get_normal_examples(self) -> pd.DataFrame:
        """
        Get examples that are neither high divergence nor low divergence.
        
        Returns:
            DataFrame containing normal examples
        """
        if self.high_divergence_examples is None or self.low_divergence_examples is None:
            raise ValueError("You must first call find_interesting_examples and find_low_divergence_examples")
        
        # Create masks for high and low divergence examples
        high_div_indices = set(self.high_divergence_examples.index.tolist())
        low_div_indices = set(self.low_divergence_examples.index.tolist())
        
        # Find examples that are neither high nor low divergence
        normal_mask = ~(self.df.index.isin(high_div_indices) | self.df.index.isin(low_div_indices))
        normal_examples = self.df[normal_mask].copy()
        
        return normal_examples
       

    def decode_token(self, token_id: Union[int, List[int]]) -> str:
        """
        Decode a token ID or list of token IDs using the tokenizer, if available.
        
        Args:
            token_id: The token ID(s) to decode - can be a single ID or a list of IDs
            
        Returns:
            The decoded token string or the original ID if tokenizer is unavailable
        """
        # Handle list of ints
        if isinstance(token_id, list):
            return self.tokenizer.decode(token_id)
        # Decode the token (wrapping in list for single tokens)
        return self.tokenizer.decode([token_id])
    
    def make_whitespace_visible(self, text: str) -> str:
        """
        Replace invisible whitespace characters with visible representations.
        
        Args:
            text: The text to process
            
        Returns:
            Text with whitespace characters made visible
        """
        result = text
        for char, replacement in self.whitespace_map.items():
            result = result.replace(char, replacement)
        return result
        
    def plot_examples(self, 
                     highlighted_examples: Optional[pd.DataFrame] = None,
                     low_divergence_examples: Optional[pd.DataFrame] = None,
                     title: str = "Jensen-Shannon Divergence Comparison",
                     figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot the divergence data with color-coded examples based on their category.
        
        Args:
            highlighted_examples: DataFrame of high divergence examples (noLN differs)
            low_divergence_examples: DataFrame of low divergence examples (all models similar)
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            The matplotlib figure containing the plot
        """
        # Use colorblind-friendly style
        plt.style.use('seaborn-v0_8-colorblind')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create masks for different categories
        highlighted_mask = np.zeros(len(self.df), dtype=bool)
        low_div_mask = np.zeros(len(self.df), dtype=bool)
        
        if highlighted_examples is not None and not highlighted_examples.empty:
            highlighted_indices = set(highlighted_examples.index.tolist())
            highlighted_mask = self.df.index.isin(highlighted_indices)
        
        if low_divergence_examples is not None and not low_divergence_examples.empty:
            low_div_indices = set(low_divergence_examples.index.tolist())
            low_div_mask = self.df.index.isin(low_div_indices)
        
        # Create a mask for regular (uncategorized) examples
        regular_mask = ~(highlighted_mask | low_div_mask)
        
        # Plot each category with different colors
        # Regular examples (gray)
        scatter_regular = ax.scatter(
            self.df.loc[regular_mask, 'jsd_baseline_vs_finetuned'],
            self.df.loc[regular_mask, 'jsd_baseline_vs_noLN'],
            c='gray', alpha=0.3, s=30, label='Regular Examples'
        )
        
        # Low divergence examples (blue)
        if low_divergence_examples is not None and not low_divergence_examples.empty:
            scatter_low = ax.scatter(
                self.df.loc[low_div_mask, 'jsd_baseline_vs_finetuned'],
                self.df.loc[low_div_mask, 'jsd_baseline_vs_noLN'],
                c='blue', alpha=0.7, s=50, label='Low Divergence'
            )
        
        # Highlighted examples (red)
        if highlighted_examples is not None and not highlighted_examples.empty:
            scatter_high = ax.scatter(
                self.df.loc[highlighted_mask, 'jsd_baseline_vs_finetuned'],
                self.df.loc[highlighted_mask, 'jsd_baseline_vs_noLN'],
                c='red', alpha=0.7, s=50, label='High Divergence'
            )
        
        # Add standard diagonal reference line (y=x)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Set equal aspect ratio to ensure the diagonal is actually at 45 degrees
        ax.set_aspect('equal')
        
        # Labels and legend
        ax.set_xlabel('JSD: baseline vs finetuned')
        ax.set_ylabel('JSD: baseline vs noLN')
        ax.set_title(title)
        ax.legend(loc='upper left')
        
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig
  

    def analyze_examples(self, 
                        examples: pd.DataFrame, 
                        n_examples: int = 5,
                        print_output: bool = True) -> None:
        """
        Analyze examples and display key information with highlighted tokens.
        
        Args:
            examples: DataFrame of selected examples
            n_examples: Number of examples to analyze
            print_output: Whether to print analysis to console
        """
        # Take the top n examples
        top_examples = examples.head(n_examples)
        
        if print_output:
            print(f"Found {len(examples)} examples. Showing top {n_examples}:")
            print("=" * 80)
        
        for idx, row in top_examples.iterrows():
            # Print basic example info and metrics
            print(f"Example #{row['original_idx']}")
            print(f"JSD (baseline vs finetuned): {row['jsd_baseline_vs_finetuned']:.6f}")
            print(f"JSD (baseline vs noLN): {row['jsd_baseline_vs_noLN']:.6f}")
            print(f"CE Loss: baseline={row['ce_baseline']:.4f}, finetuned={row['ce_finetuned']:.4f}, noLN={row['ce_noLN']:.4f}")
            
            # Process and display tokens
            full_text = row['full_sequence'].tolist()
            last_token = row['last_token']
            next_token = row['next_token']
            
            # Decode tokens
            full_text = self.decode_token(full_text)
            last_token_str = self.decode_token(last_token)
            next_token_str = self.decode_token(next_token)
            
            # Make whitespace visible
            full_text_vis = self.make_whitespace_visible(full_text)
            last_token_vis = self.make_whitespace_visible(last_token_str)
            next_token_vis = self.make_whitespace_visible(next_token_str)
            
            # Print the formatted output
            print("\nSequence with highlighted tokens:")
            print(f"{full_text_vis}")
            print(f"Last token: [{last_token_vis}]")
            print(f"Next token: -> [{next_token_vis}]")
            print("-" * 80)
   

    def analyze_both_extremes(self, n_examples: int = 5) -> Tuple[List[dict], List[dict]]:
        """
        Analyze both most divergent and least divergent examples.
        
        Args:
            n_examples: Number of examples from each extreme to analyze
            
        Returns:
            Tuple containing (high_divergence_results, low_divergence_results)
        """
        # Find most divergent examples
        high_divergence = self.find_interesting_examples()
        print(f"MOST DIVERGENT EXAMPLES (noLN differs significantly):")
        print("=" * 80)
        high_results = self.analyze_examples(high_divergence, n_examples)
        
        # Find least divergent examples (all models behave similarly)
        low_divergence = self.find_low_divergence_examples()
        print("\n\nLEAST DIVERGENT EXAMPLES (all models behave similarly):")
        print("=" * 80)
        low_results = self.analyze_examples(low_divergence, n_examples)
        
        return high_results, low_results
    
    
    def plot_sequence_length_boxplot(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a box plot comparing sequence lengths across different example types.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            The matplotlib figure containing the box plot
        """
        # Ensure we have the necessary example categorizations
        if self.high_divergence_examples is None or self.low_divergence_examples is None:
            raise ValueError("You must first call find_interesting_examples and find_low_divergence_examples")
        
        # Get the normal examples
        normal_examples = self.get_normal_examples()
        
        # Create a new DataFrame for visualization
        plot_data = []
        
        # Add high divergence examples
        for _, row in self.high_divergence_examples.iterrows():
            plot_data.append({
                'Example Type': 'High Divergence',
                'Sequence Length': row['sequence_length']
            })
        
        # Add low divergence examples
        for _, row in self.low_divergence_examples.iterrows():
            plot_data.append({
                'Example Type': 'Low Divergence',
                'Sequence Length': row['sequence_length']
            })
        
        # Add normal examples
        for _, row in normal_examples.iterrows():
            plot_data.append({
                'Example Type': 'Normal',
                'Sequence Length': row['sequence_length']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the figure
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        sns.boxplot(x='Example Type', y='Sequence Length', data=plot_df, ax=ax)
        
        # Add a strip plot to show individual points
        sns.stripplot(x='Example Type', y='Sequence Length', data=plot_df, 
                    size=4, color='black', alpha=0.3, ax=ax)
        
        # Customize the plot
        ax.set_title('Sequence Length Distribution by Example Type')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    
    def plot_loss_boxplots(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create box plots comparing CE loss across different models and example types.
        
        Args:
            figsize: Figure size as (width, height)
            
        Returns:
            The matplotlib figure containing the box plots
        """
        # Ensure we have the necessary example categorizations
        if self.high_divergence_examples is None or self.low_divergence_examples is None:
            raise ValueError("You must first call find_interesting_examples and find_low_divergence_examples")
        
        # Get the normal examples
        normal_examples = self.get_normal_examples()
        
        # Create a new DataFrame for visualization in long format
        plot_data = []
        
        # Process high divergence examples
        for _, row in self.high_divergence_examples.iterrows():
            for model in ['baseline', 'finetuned', 'noLN']:
                plot_data.append({
                    'Example Type': 'High Divergence',
                    'Model': model,
                    'Cross-Entropy Loss': row[f'ce_{model}']
                })
        
        # Process low divergence examples
        for _, row in self.low_divergence_examples.iterrows():
            for model in ['baseline', 'finetuned', 'noLN']:
                plot_data.append({
                    'Example Type': 'Low Divergence',
                    'Model': model,
                    'Cross-Entropy Loss': row[f'ce_{model}']
                })
        
        # Process normal examples
        for _, row in normal_examples.iterrows():
            for model in ['baseline', 'finetuned', 'noLN']:
                plot_data.append({
                    'Example Type': 'Normal',
                    'Model': model,
                    'Cross-Entropy Loss': row[f'ce_{model}']
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the figure
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        sns.boxplot(x='Example Type', y='Cross-Entropy Loss', hue='Model', data=plot_df, ax=ax)
        
        # Customize the plot
        ax.set_title('Cross-Entropy Loss Distribution by Example Type and Model')
        ax.grid(axis='y', alpha=0.3)
        
        # Adjust legend position
        plt.legend(title='Model', loc='upper right')
        
        plt.tight_layout()
        return fig
    
    
    def run_full_analysis(self, output_dir: str, n_examples: int = 5,
                         x_max_threshold: float = 0.1, 
                         y_min_threshold: float = 0.1,
                         max_jsd_threshold: float = 0.01):
        """
        Run a complete analysis pipeline and save results to the specified directory.
        
        Args:
            output_dir: Directory to save analysis results
            n_examples: Number of examples to analyze from each category
            x_max_threshold: Maximum JSD between baseline and finetuned for high divergence
            y_min_threshold: Minimum JSD between baseline and noLN for high divergence
            max_jsd_threshold: Maximum JSD for low divergence examples
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find interesting examples (high divergence)
        print("\nFinding examples where noLN diverges but baseline and finetuned are similar...")
        high_divergence = self.find_interesting_examples(
            x_max_threshold=x_max_threshold,
            y_min_threshold=y_min_threshold,
        )
        high_divergence.to_parquet(os.path.join(output_dir, 'divergent.parquet'))
        
        print(f"\nANALYZING HIGH DIVERGENCE EXAMPLES:")
        self.analyze_examples(high_divergence, n_examples=n_examples)
        
        # Find low divergence examples
        print("\nFinding examples where all models behave similarly...")
        low_divergence = self.find_low_divergence_examples(max_jsd_threshold=max_jsd_threshold)
        low_divergence.to_parquet(os.path.join(output_dir, 'convergent.parquet'))
        
        print(f"\nANALYZING LOW DIVERGENCE EXAMPLES:")
        self.analyze_examples(low_divergence, n_examples=n_examples)
        
        # Create visualization with both example types color-coded
        print("\nCreating visualization with color-coded examples...")
        combined_fig = self.plot_examples(
            highlighted_examples=high_divergence,
            low_divergence_examples=low_divergence,
            title="Model Divergence Comparison"
        )
        combined_fig.savefig(os.path.join(output_dir, "divergence_analysis.png"), dpi=300)
        
        # Create and save sequence length box plot
        print("\nCreating sequence length box plot...")
        seq_len_fig = self.plot_sequence_length_boxplot()
        seq_len_fig.savefig(os.path.join(output_dir, "sequence_length_boxplot.png"), dpi=300)
        
        # Create and save loss box plots
        print("\nCreating cross-entropy loss box plots...")
        loss_fig = self.plot_loss_boxplots()
        loss_fig.savefig(os.path.join(output_dir, "loss_boxplots.png"), dpi=300)
        
        print(f"\nAnalysis complete! Results saved to {output_dir}/")


# Example usage as a main script
if __name__ == "__main__":
    data_path = "/workspace/removing-layer-norm/mech_interp/inference_logs/dataset_luca-pile_samples_5000_seqlen_512_prepend_False/inference_results.parquet"
    output_dir = "divergences"
    
    # Create analyzer (loads data and initializes tokenizer)
    analyzer = DivergenceAnalyzer(data_path, agg=True)
    
    # Run the full analysis pipeline
    analyzer.run_full_analysis(
        output_dir=output_dir,
        n_examples=5,
        x_max_threshold=0.1,
        y_min_threshold=0.1,
        max_jsd_threshold=0.01
    )