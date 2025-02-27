#%%
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from visualize_token_samples import visualize_highlighted_tokens_in_markdown
from transformers import GPT2TokenizerFast
from circuitsvis.tokens import colored_tokens
from IPython.display import display
from IPython.display import HTML
plt.rcParams.update({'font.size': 18})  # Increase default font size


def combine_rendered_html(*html_objects, layout="vertical", background_color="white", labels=None, title=None):
    """
    Combine multiple RenderedHTML objects into a single HTML string.
    
    Args:
        *html_objects: Variable number of RenderedHTML objects
        layout: "vertical" or "horizontal" layout
        background_color: Background color for the container
        labels: List of labels for each visualization (defaults to "Sentence 1", "Sentence 2", etc.)
        title: Optional title to display at the top of the visualization
        
    Returns:
        HTML: IPython HTML object that can be displayed
    """
    if layout == "horizontal":
        container_style = "display: flex; flex-direction: row; flex-wrap: wrap;"
        item_style = "margin: 10px;"
    else:  # vertical
        container_style = "display: flex; flex-direction: column;"
        item_style = "margin: 10px 0;"
    
    # Add background color, padding, border, and rounded corners to the container
    container_style += f" background-color: {background_color}; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
    
    # Create default labels if none provided
    if labels is None:
        labels = [f"Sentence {i+1}" for i in range(len(html_objects))]
    
    # Ensure we have enough labels
    if len(labels) < len(html_objects):
        labels.extend([f"Sentence {i+1}" for i in range(len(labels), len(html_objects))])
    
    combined_html = f'<div style="{container_style}">\n'
    
    # Add title if provided
    if title:
        title_style = "font-weight: bold; font-size: 20px; margin-bottom: 15px; text-align: center;"
        combined_html += f'<div style="{title_style}">{title}</div>\n'
    
    for i, html_obj in enumerate(html_objects):
        # Wrap each visualization in a div
        combined_html += f'<div style="{item_style}">\n'
        
        # Add the label with some styling
        label_style = "font-weight: bold; margin-bottom: 5px; font-size: 16px;"
        combined_html += f'<div style="{label_style}">{labels[i]}</div>\n'
        
        # Add the visualization
        combined_html += html_obj._repr_html_() + "\n"
        combined_html += '</div>\n'
    
    combined_html += '</div>'
    
    # Return an HTML object that Jupyter can render
    return HTML(combined_html)

class H5DataAnalyzer:
    """Analyzes H5 files containing model inference results."""
    
    def __init__(self, h5_file_path, output_dir=None):
        """
        Initialize the analyzer with the path to an H5 file.
        
        Args:
            h5_file_path: Path to the H5 file containing inference results
            output_dir: Directory to save analysis results (defaults to same directory as h5_file)
        """
        self.h5_file_path = Path(h5_file_path)
        # Extract metadata from filename
        self.metadata = self._extract_metadata(self.h5_file_path)
        
        # Set default output directory if not provided
        if output_dir is None:
            self.output_dir = os.path.join(self.h5_file_path.parent, "analysis_results_" + self.metadata['dataset'])
        else:
            self.output_dir = Path(output_dir)
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        

        
        # Load data
        self.data = self._load_data()
        
    def _extract_metadata(self, file_path):
        """Extract metadata from the file path."""
        # Parse the directory name which contains the metadata
        dir_name = file_path.parent.name
        
        # Example directory name: dataset_apollo-pile_samples_10000_seqlen_50
        parts = dir_name.split('_')
        metadata = {}
        
        for i in range(0, len(parts), 2):
            if i+1 < len(parts):
                key = parts[i]
                value = parts[i+1]
                # Convert numeric values to integers
                if value.isdigit():
                    value = int(value)
                metadata[key] = value
                print(f"{key}: {value}")
                
        return metadata
    
    def _load_data(self):
        """Load data from the H5 file."""
        with h5py.File(self.h5_file_path, 'r') as f:
            # Get available models and model pairs
            models = list(f['CE_loss'].keys())
            model_pairs = list(f['JSD'].keys())
            
            # Load tokens
            tokens = f['tokens'][:]
            
            # Load CE losses
            ce_losses = {model: f['CE_loss'][model][:] for model in models}
            
            # Calculate CE differences for all model pairs
            ce_diffs = {}
            for model1 in models:
                for model2 in models:
                    if model1 < model2:  # Only calculate each pair once
                        pair_name = f"{model1}_vs_{model2}"
                        ce_diffs[pair_name] = ce_losses[model1] - ce_losses[model2]
            
            # Load JSD
            jsd = {pair: f['JSD'][pair][:] for pair in model_pairs}
            
        return {
            'tokens': tokens,
            'ce_losses': ce_losses,
            'ce_diffs': ce_diffs,
            'jsd': jsd,
            'models': models,
            'model_pairs': model_pairs
        }
    
    def _get_metric_data(self, metric_type):
        """Helper function to get metric data and metadata based on metric type."""
        if metric_type.lower() == 'ce':
            return {
                'values': self.data['ce_losses'],
                'models': self.data['models'],
                'label': 'CE Loss',
                'title_prefix': 'Cross-Entropy Loss',
                'bin_range': (0, 90, 0.5)  # (start, stop, step)
            }
        elif metric_type.lower() == 'ce_diff':
            return {
                'values': self.data['ce_diffs'],
                'models': list(self.data['ce_diffs'].keys()),
                'label': 'CE Loss Difference',
                'title_prefix': 'Cross-Entropy Loss Difference',
                'bin_range': (-90, 90, 0.5)  # (start, stop, step)
            }
        elif metric_type.lower() == 'jsd':
            return {
                'values': self.data['jsd'],
                'models': self.data['model_pairs'],
                'label': 'JS Divergence',
                'title_prefix': 'Jensen-Shannon Divergence',
                'bin_range': (0, 0.8, 0.01)  # (start, stop, step)
            }
        else:
            raise ValueError(f"Invalid metric_type '{metric_type}'. Must be 'ce', 'ce_diff', or 'jsd'")

    def _check_nans(self, values, model_name, metric_type, context=""):
        """Helper function to check for NaNs and print warnings."""
        nan_mask = np.isnan(values)
        nan_count = np.sum(nan_mask)
        if nan_count > 0:
            # Get indices of NaN values
            nan_indices = np.where(nan_mask)
            print(f"\nWARNING: Found {nan_count} NaN values in {metric_type} for {model_name}{context}")
            print(f"First 5 NaN positions (sample_idx, position_idx):")
            for i in range(min(5, len(nan_indices[0]))):
                if len(nan_indices) == 2:  # 2D array
                    print(f"  ({nan_indices[0][i]}, {nan_indices[1][i]})")
                else:  # 1D array
                    print(f"  ({nan_indices[0][i]})")
            print(f"NaN percentage: {(nan_count / values.size) * 100:.2f}%\n")
        return nan_count

    def plot_average_metric_values_by_position(self, metric_type, visualize=False):
        """Generic function to plot average metric values per position."""
        data = self._get_metric_data(metric_type)
        sequence_length = next(iter(data['values'].values())).shape[1]
        dataset_name = self.metadata.get('dataset', 'Unknown')
        
        plt.figure(figsize=(12, 8))
        
        for model in data['models']:
            values = data['values'][model]
            if metric_type.lower() == 'jsd':
                self._check_nans(values, model, metric_type)
                mean_value = np.nanmean(values, axis=0)
            else:
                mean_value = np.mean(values, axis=0)
            plt.plot(range(sequence_length), mean_value, label=f'{model}')
        
        plt.xlabel('Position in Sequence')
        plt.ylabel(f'Average {data["label"]}')
        plt.title(f'Average {data["title_prefix"]} by Position - Dataset: {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = self.output_dir / f"avg_{metric_type}_by_position.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()
        print(f"Saved average {data['label']} plot to {output_path}")

    def plot_metric_values_distribution(self, metric_type, log_scale=False, plot_type='histogram', bins=None, visualize=False):
        """Generic function to plot metric value distributions.
        
        Args:
            metric_type: Type of metric to analyze ('ce', 'ce_diff', or 'jsd')
            log_scale: Whether to use log scale for y-axis
            plot_type: Type of plot to create ('histogram' or 'boxplot')
        """
        data = self._get_metric_data(metric_type)
        dataset_name = self.metadata.get('dataset', 'Unknown')
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'histogram':
            colors = plt.cm.tab10.colors[:len(data['models'])]
            if bins is None:
                start, stop, step = data['bin_range']
                bins = np.arange(start, stop, step)
            
            for i, model in enumerate(data['models']):
                values = data['values'][model].flatten()
                if metric_type.lower() == 'jsd':
                    nan_count = self._check_nans(values, model, metric_type)
                    values = values[~np.isnan(values)]
                    if nan_count > 0:
                        print(f"Plotting histogram after removing NaN values...")
                
                if metric_type.lower() == 'jsd':
                    values = values[values < np.percentile(values, 99.5)]
                
                plt.hist(values, bins=bins, alpha=0.5, label=f'{model}', color=colors[i])
                
            plt.xlabel(data['label'])
            plt.ylabel('Frequency')
            plt.legend()
            
        elif plot_type == 'boxplot':
            # Prepare data for boxplot
            plot_data = []
            labels = []
            
            for model in data['models']:
                values = data['values'][model].flatten()
                if metric_type.lower() == 'jsd':
                    values = values[~np.isnan(values)]
                    if metric_type.lower() == 'jsd':
                        values = values[values < np.percentile(values, 99.5)]
                plot_data.append(values)
                labels.append(model)
            
            # Create boxplot 
            bplot = plt.boxplot(plot_data, labels=labels, patch_artist=True)
            
            plt.xlabel('Model')
            plt.ylabel(data['label'])
            plt.xticks(rotation=45)
            
        else:
            raise ValueError(f"Invalid plot_type '{plot_type}'. Must be 'histogram' or 'boxplot'")
        
        plt.title(f'Distribution of {data["title_prefix"]} - Dataset: {dataset_name}')
        plt.grid(True, alpha=0.3)
        
        if log_scale:
            plt.yscale('log')
        
        # Adjust layout for rotated labels in boxplot
        plt.tight_layout()
        
        log_suffix = "_log" if log_scale else ""
        output_path = self.output_dir / f"{metric_type}_{plot_type}{log_suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()
        print(f"Saved {data['label']} {plot_type} to {output_path}")

    def get_top_k_indices(self, metric_type, model_or_pair, top_k=10, highest=True):
        """Return the indices of the highest or lowest metric values."""
        data = self._get_metric_data(metric_type)
        
        if model_or_pair not in data['models']:
            raise ValueError(f"Model/pair '{model_or_pair}' not found. Available: {data['models']}")
        
        values = data['values'][model_or_pair]
        
        # Create a mask for NaN values (especially relevant for JSD)
        nan_mask = np.isnan(values)
        
        # Replace NaN values with -inf or inf to exclude them from the results
        values_no_nan = values.copy()
        if highest:
            values_no_nan[nan_mask] = -np.inf
            flat_indices = np.argsort(values_no_nan.flatten())[-top_k:][::-1]
        else:
            values_no_nan[nan_mask] = np.inf
            flat_indices = np.argsort(values_no_nan.flatten())[:top_k]
        
        # Convert flat indices back to 2D indices
        num_positions = values.shape[1]
        sample_indices = flat_indices // num_positions
        position_indices = flat_indices % num_positions
        
        indices = list(zip(sample_indices, position_indices))
        result_values = [values[i, j] for i, j in indices]
        
        return indices, result_values
    

    def get_top_k_indices_diff(self, metric_type, model_or_pair1, model_or_pair2, top_k=10, highest=True):
        data = self._get_metric_data(metric_type)
        if model_or_pair1 not in data['models'] or model_or_pair2 not in data['models']:
            raise ValueError(f"Model/pair '{model_or_pair1}' or '{model_or_pair2}' not found. Available: {data['models']}")
        
        values1 = data['values'][model_or_pair1]
        values2 = data['values'][model_or_pair2]
        values = values1 - values2

        # Create a mask for NaN values (especially relevant for JSD)
        nan_mask = np.isnan(values)
        # Replace NaN values with -inf or inf to exclude them from the results

        values_no_nan = values.copy()
        if highest:
            values_no_nan[nan_mask] = -np.inf
            flat_indices = np.argsort(values_no_nan.flatten())[-top_k:][::-1]
        else:
            values_no_nan[nan_mask] = np.inf
            flat_indices = np.argsort(values_no_nan.flatten())[:top_k]
        
        # Convert flat indices back to 2D indices
        num_positions = values.shape[1]
        sample_indices = flat_indices // num_positions
        position_indices = flat_indices % num_positions
        
        indices = list(zip(sample_indices, position_indices))
        result_values = [values[i, j] for i, j in indices]
        return indices, result_values
        

    def plot_metric_values_distribution_by_position(self, metric_type, positions, log_scale=False, plot_type='histogram', bins=None, visualize=False):
        """
        Plot distribution of metric values at specific sequence positions.
        Args:
            metric_type: Type of metric to analyze ('ce', 'ce_diff', or 'jsd')
            positions: Position or list of positions in the sequence to analyze
            log_scale: Whether to use log scale for y-axis
            plot_type: Type of plot to create ('histogram' or 'boxplot')
        """
        data = self._get_metric_data(metric_type)
        dataset_name = self.metadata.get('dataset', 'Unknown')
        
        # Convert single position to list for consistent handling
        if not isinstance(positions, list):
            positions = [positions]
            
        # Check if positions are valid
        sequence_length = next(iter(data['values'].values())).shape[1]
        for position in positions:
            if position >= sequence_length:
                print(f"Error: Position {position} is out of range. Max position is {sequence_length-1}.")
                return
        
        # Create a figure with subplots in a single row
        fig, axes = plt.subplots(1, len(positions), figsize=(6*len(positions), 8), sharey=True)
        
        # Handle single subplot case
        if len(positions) == 1:
            axes = [axes]
        
        if plot_type == 'histogram':
            if bins is None:
                start, stop, step = data['bin_range']
                bins = np.arange(start, stop, step)
            max_freq = 0
            
            # First pass to determine max frequency for shared y-axis
            for position in positions:
                for model in data['models']:
                    position_values = data['values'][model][:, position]
                    if metric_type == 'jsd':
                        position_values = position_values[~np.isnan(position_values)]
                    hist, _ = np.histogram(position_values, bins=bins, density=True)
                    max_freq = max(max_freq, np.max(hist))
            
            # Second pass to plot histograms
            for i, position in enumerate(positions):
                ax = axes[i]
                
                for model in data['models']:
                    position_values = data['values'][model][:, position]
                    
                    if metric_type == 'jsd':
                        self._check_nans(position_values, model, metric_type)
                        position_values = position_values[~np.isnan(position_values)]
                    
                    ax.hist(position_values, bins=bins, alpha=0.5, label=model)
                
                ax.set_xlabel(data['label'])
                if i == 0:
                    ax.set_ylabel('Frequency')
                ax.set_title(f'Position {position}')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                if log_scale:
                    ax.set_yscale('log')
        
        elif plot_type == 'boxplot':
            for i, position in enumerate(positions):
                ax = axes[i]
                plot_data = []
                labels = []
                
                for model in data['models']:
                    position_values = data['values'][model][:, position]
                    
                    if metric_type == 'jsd':
                        self._check_nans(position_values, model, metric_type)
                        position_values = position_values[~np.isnan(position_values)]
                    
                    plot_data.append(position_values)
                    labels.append(model)
                
                ax.boxplot(plot_data, labels=labels, patch_artist=True)
                ax.set_xlabel('Model')
                if i == 0:
                    ax.set_ylabel(data['label'])
                ax.set_title(f'Position {position}')
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                if log_scale:
                    ax.set_yscale('log')
        
        else:
            raise ValueError(f"Invalid plot_type '{plot_type}'. Must be 'histogram' or 'boxplot'")
        
        # Add a common title
        fig.suptitle(f'Distribution of {data["title_prefix"]} by Position - Dataset: {dataset_name}', 
                    fontsize=16)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        
        # Save the figure
        positions_str = '_'.join(str(p) for p in positions)
        log_suffix = "_log" if log_scale else ""
        output_path = self.output_dir / f"{metric_type}_at_positions_{positions_str}_{plot_type}{log_suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if visualize:
            plt.show()
        plt.close()
        
        print(f"Saved {data['label']} at positions {positions} plot to {output_path}")

    def visualize_token_sequences_for_extreme_metric_values_in_html(self, metric_type, model_or_pair, top_k=10, highest=True, 
                                                            save_path=None, tokenizer='gpt2'):
        """Visualize the top k token sequences for a given metric."""
        if tokenizer == 'gpt2':
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        indices, values = self.get_top_k_indices(metric_type, model_or_pair, top_k=top_k, highest=highest)
        tokens_list = [self.data['tokens'][idx[0], :] for idx in indices]
        highlight_idxs = [idx[1] for idx in indices]
        if highest: 
            extreme_metric_type = 'highest'
        else:
            extreme_metric_type = 'lowest'
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{extreme_metric_type}_{top_k}_{metric_type}_{model_or_pair}_highlighted_tokens_visualization')
        
        html_objects = []
        if metric_type == 'jsd':
            for tokens, highlight_idx in zip(tokens_list, highlight_idxs):
                full_text_pre_highlight = tokenizer.decode(tokens[:highlight_idx], skip_special_tokens=False)
                full_text_post_highlight = tokenizer.decode(tokens[highlight_idx+2:], skip_special_tokens=False)
                highlighted_token = tokenizer.decode([tokens[highlight_idx]], skip_special_tokens=False)
                tokens = [full_text_pre_highlight, highlighted_token, full_text_post_highlight]
                values = [0, 1, 0]
                html_objects.append(colored_tokens([repr(token)[1:-1] for token in tokens], values, min_value=-1, max_value=1, positive_color='red', negative_color='blue'))
        elif metric_type == 'ce_diff' or metric_type == 'ce':
            for tokens, highlight_idx in zip(tokens_list, highlight_idxs):
                full_text_pre_highlight = tokenizer.decode(tokens[:highlight_idx], skip_special_tokens=False)
                full_text_post_highlight = tokenizer.decode(tokens[highlight_idx+2:], skip_special_tokens=False)
                highlighted_token = tokenizer.decode([tokens[highlight_idx]], skip_special_tokens=False)
                following_token = tokenizer.decode([tokens[highlight_idx+1]], skip_special_tokens=False)
                tokens = [full_text_pre_highlight, highlighted_token, following_token, full_text_post_highlight]
                values = [0, 1, -1, 0]
                html_objects.append(colored_tokens([repr(token)[1:-1] for token in tokens], values, min_value=-1, max_value=1, positive_color='red', negative_color='blue'))
        
        combined = combine_rendered_html(*html_objects, layout="vertical", background_color="white", 
                                        title=f'{extreme_metric_type} {top_k} {metric_type} {model_or_pair}', labels=None)
        display(combined)
        return combined
        
    def visualize_token_sequences_for_extreme_metric_values_differences_in_html(self, metric_type, model_or_pair_1, model_or_pair_2, top_k=10, highest=True, 
                                                            save_path=None, tokenizer='gpt2'):
        """Visualize the top k token sequences for a given metric."""
        if tokenizer == 'gpt2':
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        indices, values = self.get_top_k_indices_diff(metric_type, model_or_pair_1, model_or_pair_2, top_k=top_k, highest=highest)
        tokens_list = [self.data['tokens'][idx[0], :] for idx in indices]
        highlight_idxs = [idx[1] for idx in indices]
        if highest: 
            extreme_metric_type = 'highest'
        else:
            extreme_metric_type = 'lowest'
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{extreme_metric_type}_{top_k}_{metric_type}_{model_or_pair_1}_VS_{model_or_pair_2}_highlighted_tokens_visualization')
        
        html_objects = []
        if metric_type == 'jsd':
            for tokens, highlight_idx in zip(tokens_list, highlight_idxs):
                full_text_pre_highlight = tokenizer.decode(tokens[:highlight_idx], skip_special_tokens=False)
                full_text_post_highlight = tokenizer.decode(tokens[highlight_idx+1:], skip_special_tokens=False)
                highlighted_token = tokenizer.decode([tokens[highlight_idx]], skip_special_tokens=False)
                tokens = [full_text_pre_highlight, highlighted_token, full_text_post_highlight]
                values = [0, 1, 0]
                html_objects.append(colored_tokens([repr(token)[1:-1] for token in tokens], values, min_value=-1, max_value=1, positive_color='red', negative_color='blue'))
        elif metric_type == 'ce_diff' or metric_type == 'ce':
            for tokens, highlight_idx in zip(tokens_list, highlight_idxs):
                full_text_pre_highlight = tokenizer.decode(tokens[:highlight_idx], skip_special_tokens=False)
                full_text_post_highlight = tokenizer.decode(tokens[highlight_idx+2:], skip_special_tokens=False)
                highlighted_token = tokenizer.decode([tokens[highlight_idx]], skip_special_tokens=False)
                following_token = tokenizer.decode([tokens[highlight_idx+1]], skip_special_tokens=False)
                tokens = [full_text_pre_highlight, highlighted_token, following_token, full_text_post_highlight]
                values = [0, 1, -1, 0]
                html_objects.append(colored_tokens([repr(token)[1:-1] for token in tokens], values, min_value=-1, max_value=1, positive_color='red', negative_color='blue'))
        
        combined = combine_rendered_html(*html_objects, layout="vertical", background_color="white", 
                                        title=f'{extreme_metric_type} {top_k} {metric_type} {model_or_pair_1} VS {model_or_pair_2}', labels=None)
        # Display in notebook
        display(combined)
        return combined
    
    def run_all_analysis(self):
        if self.metadata['dataset'] == 'apollo-owt':
            bins_ce = np.arange(0, 40, 0.5)
            bins_ce_diff = np.arange(-40, 40, 0.5)
            bins_jsd = np.arange(0, 0.8, 0.01)
        else:
            bins_ce = np.arange(0, 90, 0.5)
            bins_ce_diff = np.arange(-90, 90, 0.5)
            bins_jsd = np.arange(0, 0.8, 0.01)
        self.plot_average_metric_values_by_position('ce', visualize=True)
        self.plot_average_metric_values_by_position('ce_diff', visualize=True)
        self.plot_average_metric_values_by_position('jsd', visualize=True)

        self.plot_metric_values_distribution('ce', log_scale=True, plot_type='histogram', visualize=True, bins=bins_ce)
        self.plot_metric_values_distribution('ce_diff', log_scale=True, plot_type='histogram', visualize=True, bins=bins_ce_diff)
        self.plot_metric_values_distribution('jsd', log_scale=True, plot_type='histogram', visualize=True, bins=bins_jsd)

        self.plot_metric_values_distribution('ce', log_scale=False, plot_type='boxplot', visualize=True)
        self.plot_metric_values_distribution('ce_diff', log_scale=False, plot_type='boxplot', visualize=True)
        self.plot_metric_values_distribution('jsd', log_scale=False, plot_type='boxplot', visualize=True)

        self.plot_metric_values_distribution_by_position('ce', positions=[0, 1, 5, 20, 45], log_scale=True, plot_type='histogram', visualize=True, bins=bins_ce)
        self.plot_metric_values_distribution_by_position('ce_diff', positions=[0, 1, 5, 20, 45], log_scale=True, plot_type='histogram', visualize=True, bins=bins_ce_diff)
        self.plot_metric_values_distribution_by_position('jsd', positions=[0, 1, 5, 20, 45], log_scale=True, plot_type='histogram', visualize=True, bins=bins_jsd)
        
        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce', 'baseline', top_k=10, highest=True)
        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce', 'finetuned', top_k=10, highest=True)
        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce', 'noLN', top_k=10, highest=True)

        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce_diff', 'baseline_vs_finetuned', top_k=10, highest=True)
        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce_diff', 'baseline_vs_noLN', top_k=10, highest=True)
        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce_diff', 'finetuned_vs_noLN', top_k=10, highest=True)

        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce_diff', 'baseline_vs_finetuned', top_k=10, highest=False)
        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce_diff', 'baseline_vs_noLN', top_k=10, highest=False)
        self.visualize_token_sequences_for_extreme_metric_values_in_html('ce_diff', 'finetuned_vs_noLN', top_k=10, highest=False)

        self.visualize_token_sequences_for_extreme_metric_values_in_html('jsd', 'baseline_vs_finetuned', top_k=10, highest=True)
        self.visualize_token_sequences_for_extreme_metric_values_in_html('jsd', 'baseline_vs_noLN', top_k=10, highest=True)
        self.visualize_token_sequences_for_extreme_metric_values_in_html('jsd', 'finetuned_vs_noLN', top_k=10, highest=True)

        self.visualize_token_sequences_for_extreme_metric_values_differences_in_html('jsd', 'baseline_vs_noLN', 'baseline_vs_finetuned', top_k=10, highest=True)
    


config = {
    'dataset': 'apollo-pile',
    'num_samples': 10000,
    'sequence_length': 50,
}

h5_file_path = f'/workspace/removing-layer-norm/mech_interp/data/inference_logs/dataset_{config["dataset"]}_samples_{config["num_samples"]}_seqlen_{config["sequence_length"]}/inference_results.h5'
output_dir = f'/workspace/removing-layer-norm/mech_interp/data/inference_logs/dataset_{config["dataset"]}_samples_{config["num_samples"]}_seqlen_{config["sequence_length"]}/analysis_results'
analyzer = H5DataAnalyzer(h5_file_path, output_dir)
analyzer.run_all_analysis()

config = {
    'dataset': 'apollo-owt',
    'num_samples': 10000,
    'sequence_length': 50,
}

h5_file_path = f'/workspace/removing-layer-norm/mech_interp/data/inference_logs/dataset_{config["dataset"]}_samples_{config["num_samples"]}_seqlen_{config["sequence_length"]}/inference_results.h5'
output_dir = f'/workspace/removing-layer-norm/mech_interp/data/inference_logs/dataset_{config["dataset"]}_samples_{config["num_samples"]}_seqlen_{config["sequence_length"]}/analysis_results'
analyzer = H5DataAnalyzer(h5_file_path, output_dir)
analyzer.run_all_analysis()


# %%
