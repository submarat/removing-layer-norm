import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any, Union
import re
from transformers import AutoTokenizer


class DivergenceAnalyzer:
    """
    A class to analyze and identify interesting examples in model divergence data.
    This specifically focuses on finding examples where a noLN model diverges
    from baseline and finetuned models, while baseline and finetuned remain similar.
    """
    
    def __init__(self, data_path: str, min_seq_length: Optional[int] = 50, agg: bool = False):
        """
        Initialize the analyzer by loading data from a parquet file and setting up tokenizer.
        
        Args:
            data_path: Path to the parquet file containing divergence data
            tokenizer_name: Name of the HuggingFace model for tokenizer (default: "gpt2")
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
        
        # Compute ratios for filtering
        self._compute_ratios()
        
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
            ' ': '\\s'
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
        self.df = aggregated_df 
        
        
    def _compute_ratios(self) -> None:
        """Compute ratio of divergences, handling division by zero."""
        epsilon = 1e-10  # Small value to avoid division by zero
        self.df['jsd_ratio'] = self.df['jsd_baseline_vs_noLN'] / (self.df['jsd_baseline_vs_finetuned'] + epsilon)


    def find_interesting_examples(self, 
                                 x_max_threshold: float,
                                 y_min_threshold: float,
                                 min_ratio_threshold: float) -> pd.DataFrame:
        """
        Find examples where noLN diverges but baseline and finetuned are similar.
        
        Args:
            x_max_threshold: Maximum JSD between baseline and finetuned
            y_min_threshold: Minimum JSD between baseline and noLN
            min_ratio_threshold: Minimum ratio of y/x divergences
            
        Returns:
            DataFrame containing filtered examples, sorted by divergence ratio
        """
        interesting_examples = self.df[
            (self.df['jsd_baseline_vs_finetuned'] <= x_max_threshold) &
            (self.df['jsd_baseline_vs_noLN'] >= y_min_threshold) &
            (self.df['jsd_ratio'] >= min_ratio_threshold)
        ]
        
        return interesting_examples.sort_values('jsd_ratio', ascending=False)


    def find_low_divergence_examples(self, max_jsd_threshold: float = 0.01) -> pd.DataFrame:
        """
        Find examples where all three models have very similar behavior.
        
        Args:
            max_jsd_threshold: Maximum JSD between any pair of models
            
        Returns:
            DataFrame containing examples with consistently low divergence
        """
        low_divergence = self.df[
            (self.df['jsd_baseline_vs_finetuned'] <= max_jsd_threshold) &
            (self.df['jsd_baseline_vs_noLN'] <= max_jsd_threshold) &
            (self.df['jsd_finetuned_vs_noLN'] <= max_jsd_threshold)
        ]
        
        # Sum all JSDs to rank examples
        low_divergence['total_jsd'] = (
            low_divergence['jsd_baseline_vs_finetuned'] + 
            low_divergence['jsd_baseline_vs_noLN'] +
            low_divergence['jsd_finetuned_vs_noLN']
        )
        
        return low_divergence.sort_values('total_jsd')
        
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
            print(f"Divergence Ratio: {row['jsd_ratio']:.2f}")
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
        # Find most divergent examples (high ratio of noLN vs baseline/finetuned)
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
   

# Example usage as a main script
if __name__ == "__main__":
    data_path = "/workspace/removing-layer-norm/mech_interp/inference_logs/dataset_apollo-pile_samples_5000_seqlen_512_prepend_False/inference_results.parquet"
    output_dir = "divergences"
    n_examples = 5
    agg = False
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer (loads data and initializes tokenizer)
    analyzer = DivergenceAnalyzer(data_path, min_seq_length=50, agg=agg)
    
    # Find interesting examples (high divergence)
    print("\nFinding examples where noLN diverges but baseline and finetuned are similar...")
    high_divergence = analyzer.find_interesting_examples(
        x_max_threshold=1.0,    # Max JSD for baseline vs finetuned
        y_min_threshold=0.2,    # Min JSD for baseline vs noLN
        min_ratio_threshold=3.0  # Min ratio difference
    )
    high_divergence.to_parquet((os.path.join(output_dir, 'divergent.parquet')))
    
    print(f"\nANALYZING HIGH DIVERGENCE EXAMPLES:")
    analyzer.analyze_examples(high_divergence, n_examples=n_examples)
    
    # Find low divergence examples
    print("\nFinding examples where all models behave similarly...")
    low_divergence = analyzer.find_low_divergence_examples(max_jsd_threshold=0.01)
    low_divergence.to_parquet((os.path.join(output_dir, 'convergent.parquet')))

    
    print(f"\nANALYZING LOW DIVERGENCE EXAMPLES:")
    analyzer.analyze_examples(low_divergence, n_examples=n_examples)
    
    # Create visualization with both example types color-coded
    print("\nCreating visualization with color-coded examples...")
    combined_fig = analyzer.plot_examples(
        highlighted_examples=high_divergence,
        low_divergence_examples=low_divergence,
        title="Model Divergence Comparison"
    )
    combined_fig.savefig(os.path.join(output_dir, "divergence_analysis.png"), dpi=300)
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
