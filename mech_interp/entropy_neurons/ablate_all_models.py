# %%
import sys
import copy
import os
import numpy as np
import torch as t
import matplotlib.pylab as plt
from matplotlib.patches import Patch
import seaborn as sns
import einops
from tqdm.auto import tqdm


parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from load_dataset import DataLoader
from load_models import ModelFactory


# Create output directories
os.makedirs('figures', exist_ok=True)
   
    
class AblationAnalyzer:
    def __init__(self, model, model_name, batch_size=512):
        """
        Initialize the AblationAnalyzer with either a model or a model_loader.
        
        Args:
            model: A pre-loaded transformer model
            model_loader: A function that loads the model
            batch_size: Batch size for processing to avoid OOM issues
        """
        self.model = model
        self.n_layers = model.cfg.n_layers
       
        for param in self.model.parameters():
            param.requires_grad_(False)
            
        self.model_name = model_name
        self.batch_size = batch_size
        
        self.dataloader = DataLoader(
                dataset_name="luca-pile",
                batch_size=1,
                max_context=512,
                num_samples=250,
                prepend_bos=False
                ).create_dataloader()        
        
        # Initialize containers for activations and results
        self.targets = None
        self.resid_inp = None
        self.attn_out = None
        self.mlp_mid = None
        self.ref_loss = None
        self.simulated_loss = None
        
        # Save important weights for ablation analysis
        self.Wout = None
        self.b_out = None
        self.unembed = None
        
        # Store results
        self.results = {}
        
    def calculate_entropy(self, logits):
        # Convert logits to probabilities
        probs = t.nn.functional.softmax(logits, dim=-1)
        # Calculate entropy: -sum(p * log(p))
        # Adding a small epsilon to avoid log(0)
        log_probs = t.log(probs + 1e-10)
        entropy = -t.sum(probs * log_probs, dim=-1)
        return entropy
        
    def extract_activations(self, hook_names=None):
        """
        Extract the relevant activations and ground truth loss.
        
        Args:
            dataloader: DataLoader for the data
            hook_names: List of hook names to extract
            
        Returns:
            self for method chaining
        """
        if hook_names is None:
            hook_names = [
                f"blocks.{self.n_layers - 1}.hook_resid_pre",
                f"blocks.{self.n_layers - 1}.hook_attn_out",
                f"blocks.{self.n_layers - 1}.mlp.hook_post",
                f"blocks.{self.n_layers - 1}.hook_mlp_out"
            ]
        
        targets = []
        resid_inp = []
        attn_out = []
        mlp_mid = []
        gt_loss = []

        with t.no_grad():
            for batch in tqdm(self.dataloader, desc="Extracting activations"):
                device = next(self.model.parameters()).device 
                batch = batch.to(device)
                
                logits, cache = self.model.run_with_cache(
                    batch,
                    names_filter=lambda name: name in hook_names
                )

                # Crop to match target token positions
                target_token = batch[:, 1:].reshape(-1)
                
                # Extract the residual stream, attention out, and middle of mlp final
                resid_pre = cache[f"blocks.{self.n_layers - 1}.hook_resid_pre"][:, :-1, :]
                attn = cache[f"blocks.{self.n_layers - 1}.hook_attn_out"][:, :-1, :]
                mid_mlp = cache[f"blocks.{self.n_layers - 1}.mlp.hook_post"][:, :-1, :]
                logits = logits[:, :-1, :]
                
                # Reshape to (batch * seq_len, ...)
                resid_pre = resid_pre.reshape(-1, resid_pre.size(-1))
                attn = attn.reshape(-1, attn.size(-1))
                mid_mlp = mid_mlp.reshape(-1, mid_mlp.size(-1))
                logits = logits.reshape(-1, logits.size(-1))
                loss = t.nn.functional.cross_entropy(
                    logits,
                    target_token,
                    reduction='none'
                )

                # Store for later
                targets.append(target_token)
                resid_inp.append(resid_pre)
                attn_out.append(attn)
                mlp_mid.append(mid_mlp)
                gt_loss.append(loss)

        # Concatenate across all batches
        self.targets = t.cat(targets, dim=0)
        self.resid_inp = t.cat(resid_inp, dim=0)
        self.attn_out = t.cat(attn_out, dim=0)
        self.mlp_mid = t.cat(mlp_mid, dim=0)
        self.ref_loss = t.cat(gt_loss, dim=0)

        # Save important weights for ablation analysis
        self.Wout = self.model.blocks[-1].mlp.W_out.clone()
        self.b_out = self.model.blocks[-1].mlp.b_out.clone()
        self.unembed = copy.deepcopy(self.model.unembed)
        
        # Delete model and empty cache to free up gpu memory
        del self.model
        t.cuda.empty_cache()
        
        return self
    
    def run_simulation(self):
        """
        Run the 'simulated' version of the final layer with the extracted activations 
        and compare loss to ground truth.
        
        Returns:
            self for method chaining
        """
        if self.mlp_mid is None:
            raise ValueError("No activations found. Run extract_activations first.")
        
        # Calculate original statistics batch wise to avoid OOM issues
        num_samples = self.mlp_mid.shape[0]
        all_losses = []
        all_entropies = []

        # Process in batches
        for start_idx in tqdm(range(0, num_samples, self.batch_size), desc="Running simulation"):
            end_idx = min(start_idx + self.batch_size, num_samples)
            
            # Get batch slices
            batch_mlp_mid = self.mlp_mid[start_idx:end_idx]
            batch_resid_inp = self.resid_inp[start_idx:end_idx]
            batch_attn_out = self.attn_out[start_idx:end_idx]
            batch_targets = self.targets[start_idx:end_idx]
            
            # Calculate for this batch
            batch_mlp_out = einops.einsum(self.Wout, batch_mlp_mid, 'dmlp dmodel, seq dmlp -> seq dmodel') + self.b_out
            batch_x_orig = batch_resid_inp + batch_attn_out + batch_mlp_out
            if 'noLN' in self.model_name: # No centering for LN free model
                batch_x_centred = batch_x_orig 
            else:
                batch_x_mean = batch_x_orig.mean(dim=-1, keepdim=True)
                batch_x_var = (batch_x_orig - batch_x_mean).pow(2).mean(dim=-1, keepdim=True)
                batch_x_centred = (batch_x_orig - batch_x_mean) / (batch_x_var + 1e-12).sqrt()
            batch_logits = self.unembed(batch_x_centred)
            
            # Calculate loss
            batch_loss = t.nn.functional.cross_entropy(
                batch_logits,
                batch_targets,
                reduction='none'
            )
            all_losses.append(batch_loss)
            
            # Calculate entropy
            batch_entropy = self.calculate_entropy(batch_logits)
            all_entropies.append(batch_entropy)
            
            
            # Free up memory
            del batch_mlp_mid, batch_resid_inp, batch_attn_out
            del batch_mlp_out, batch_x_orig, batch_logits
            t.cuda.empty_cache()

        # Combine all results
        self.simulated_loss = t.cat(all_losses)
        self.simulated_entropy = t.cat(all_entropies)
        
        # Check that loss from original model is close to our 'simulation'
        print(f"Reference loss: {self.simulated_loss.mean().item():.4f}, Simulated loss: {self.simulated_loss.mean().item():.4f}")
        print(f"Simulated entropy: {self.simulated_entropy.mean().item():.4f}")
 
        return self
    
    def _perform_ablation(self, indices_to_ablate):
        """
        Helper method to perform ablation on specified indices.
        
        Args:
            indices_to_ablate: List of neurons to ablate (always treated as a list)
            
        Returns:
            Total effect of the ablation
        """
        # Calculate mean activation for mean ablation
        mean_mlp_act = self.mlp_mid.mean(dim=0, keepdim=True)
        
        # Clone to avoid modifying original
        ablated_mid = self.mlp_mid.clone()
        
        # Ensure indices_to_ablate is a list
        if not isinstance(indices_to_ablate, (list, tuple)):
            indices_to_ablate = [indices_to_ablate]
        
        # Apply ablation for all indices in the list
        for idx in indices_to_ablate:
            ablated_mid[:, idx] = mean_mlp_act[:, idx]
        
        # Process in batches to avoid OOM
        total_loss_ablated = []
        total_entropy_ablated = []
        
        for start_idx in range(0, len(self.targets), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.targets))
            
            # Get correct batch indices
            batch_targets = self.targets[start_idx:end_idx]
            batch_resid_inp = self.resid_inp[start_idx:end_idx]
            batch_attn_out = self.attn_out[start_idx:end_idx]
            batch_ablated_mlp_mid = ablated_mid[start_idx:end_idx]
            
            # Compute logits 
            batch_ablated_mlp_out = einops.einsum(
                self.Wout, 
                batch_ablated_mlp_mid,
                'dmlp dmodel, seq dmlp -> seq dmodel'
            ) + self.b_out
            
            batch_x_ablated = batch_resid_inp + batch_attn_out + batch_ablated_mlp_out
            
            # Calculate normalized logits
            if 'noLN' in self.model_name: # No centering for LN free model
                batch_x_ablated_centred = batch_x_ablated
            else:
                batch_x_ablated_mean = batch_x_ablated.mean(dim=-1, keepdim=True)
                batch_x_ablated_var = (batch_x_ablated - batch_x_ablated_mean).pow(2).mean(dim=-1, keepdim=True)
                batch_x_ablated_centred = (batch_x_ablated - batch_x_ablated_mean) / (batch_x_ablated_var + 1e-12).sqrt()
            batch_logits_ablated = self.unembed(batch_x_ablated_centred)
            
            # Calculate ablated loss
            batch_loss = t.nn.functional.cross_entropy(
                batch_logits_ablated, 
                batch_targets,
                reduction='none'
            )
            total_loss_ablated.append(batch_loss)
            
            # Calculate ablated entropy
            batch_entropy = self.calculate_entropy(batch_logits_ablated)
            total_entropy_ablated.append(batch_entropy)
            
        # Combine results
        total_loss_ablated = t.cat(total_loss_ablated, dim=0)
        total_entropy_ablated = t.cat(total_entropy_ablated, dim=0)
        total_effect_loss = (self.simulated_loss - total_loss_ablated).abs().cpu().numpy()
        total_effect_entropy = (self.simulated_entropy - total_entropy_ablated).abs().cpu().numpy()
            
        return {'loss_effect' : total_effect_loss,
                'entropy_effect': total_effect_entropy
        }
    
    def run_ablation(self, neuron_indices):
        """
        Run ablation for the specified neuron indices.
        
        Args:
            neuron_indices: List of neuron indices to ablate. If 'all' is in the list,
                          all specified neurons will be ablated together.
        
        Returns:
            self for method chaining
        """
        if self.simulated_loss is None:
            raise ValueError("No simulation has been run. Run run_simulation first.")

        # Run individual neuron ablations
        for idx in tqdm(neuron_indices, desc="Running individual neuron ablations"):
            effect_dict = self._perform_ablation([idx])
            self.results[str(idx)] = effect_dict
        
        return self.results        


def plot_model_comparison(model_results, model_type='small', save_path=None, metric='loss_effect'):
    """
    Create a single box plot comparing total effects across models for each neuron.
    
    Args:
        model_results: Dictionary mapping model names to their ablation results
        save_path: Path to save the figure
        metric: Which metric to plot ('loss_effect' or 'entropy_effect')
    """
    # Set larger font sizes globally
    plt.rcParams.update({
        'font.size': 16,           # Base font size
        'axes.titlesize': 20,      # Title font size
        'axes.labelsize': 18,      # Axis label font size
        'xtick.labelsize': 16,     # X tick label size
        'ytick.labelsize': 16,     # Y tick label size
        'legend.fontsize': 16,     # Legend font size
        'legend.title_fontsize': 16  # Legend title font size
    }) 
    model_type = model_type.capitalize()
    model_labels = {
            'baseline': f'GPT2-{model_type} original',
            'finetuned': f'GPT2-{model_type} vanilla FT',
            'noLN': f'GPT2-{model_type} LN-free FT'
        }
    
    # Set plot strings based on metric
    if metric == 'loss_effect':
        y_label = '|Δ CE Loss|'
        title = 'Entropy Neuron Mean Ablations (CE Loss)'
    elif metric == 'entropy_effect':
        y_label = '|Δ Entropy|'
        title = 'Entropy Neuron Mean Ablations (Entropy)'
    else:
        raise ValueError(f"Metric {metric} not supported. Use 'loss_effect' or 'entropy_effect'.")
    
    # Extract neuron indices from results (excluding 'All')
    neuron_indices = []
    for model_name in model_results:
        for key in model_results[model_name]:
            if key not in neuron_indices:
                neuron_indices.append(key)
    
    # Sort neuron indices for consistent display
    neuron_indices = sorted(neuron_indices, key=lambda x: int(x))
    
    # Get default matplotlib colors
    colors = sns.color_palette("colorblind")
    
    # Set up explicit colors for each model using default matplotlib colors
    model_colors = {
        'baseline': colors[0],
        'finetuned': colors[1],
        'noLN': colors[2]
    }
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define width of bars and positions
    model_names = list(model_results.keys())
    n_models = len(model_names)
    n_categories = len(neuron_indices)
    width = 0.3 / n_models  # Width of each box
    
    # For each neuron (+ "All"), prepare data for all models
    all_data = []
    all_positions = []
    all_colors = []
    all_labels = []
    x_positions = []
    x_labels = []
    
    # Process individual neuron data first
    for i, neuron_idx in enumerate(neuron_indices):
        category_center = i + 1
        x_positions.append(category_center)
        x_labels.append(f"Neuron {neuron_idx}")
        
        # Each model gets a box in this category
        for j, model_name in enumerate(model_names):
            if neuron_idx in model_results[model_name]:
                position = category_center + (j - n_models/2 + 0.5) * width
                all_positions.append(position)
                all_data.append(model_results[model_name][neuron_idx][metric])
                all_colors.append(model_colors[model_name])
                all_labels.append(model_name)
    
    # Create boxplots
    bplots = []
    for data, pos, color, label in zip(all_data, all_positions, all_colors, all_labels):
        bp = ax.boxplot(data, positions=[pos], patch_artist=True, 
                       widths=width*0.8, showfliers=False)
        
        # Set color for this box with semi-transparency
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.7)
            
        # Set median line to black
        for median in bp['medians']:
            median.set(color='black', linewidth=1, alpha=0.7)
        
        bplots.append(bp)
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis labels and limits
    ax.set_ylabel(y_label)
    #ax.set_title(title)
    
    # Set x-ticks at category centers
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    
    # Add vertical separators between neuron categories
    for i in range(1, n_categories):
        ax.axvline(x=i + 0.5, color='gray', linestyle='-', alpha=0.3)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=model_colors[model_name],
              edgecolor='black',
              label=model_labels[model_name])
        for model_name in model_names
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    
    return fig

def print_ablation_summary(model_results):
    """
    Simple function to print ablation summary with relative changes.
    
    Args:
        model_results: Dictionary of ablation results per model and neuron
        analyzers: Dictionary mapping model names to AblationAnalyzer instances 
                  (or just their reference metrics)
    """
    
    print("\n=== ABLATION EFFECTS SUMMARY ===\n")
    
    # Get neuron indices
    neuron_indices = []
    for model_name in model_results:
        for key in model_results[model_name]:
            if key not in neuron_indices:
                neuron_indices.append(key)
    
    # Sort neuron indices
    neuron_indices = sorted(neuron_indices, key=lambda x: int(x))
    
    for neuron_idx in neuron_indices:
        print(f"NEURON {neuron_idx}:")
        
        for model_name in model_results:
            print(f"Model name : {model_name}")
            if neuron_idx in model_results[model_name]:
                # Get absolute effect values
                loss_effect = model_results[model_name][neuron_idx]['loss_effect']
                entropy_effect = model_results[model_name][neuron_idx]['entropy_effect']
                
                # Calculate statistics for absolute effects
                loss_median = np.median(loss_effect)
                loss_q1 = np.percentile(loss_effect, 25)
                loss_q3 = np.percentile(loss_effect, 75)
                
                entropy_median = np.median(entropy_effect)
                entropy_q1 = np.percentile(entropy_effect, 25)
                entropy_q3 = np.percentile(entropy_effect, 75)
                
                # Print absolute change
                print(f"    CE Loss Abs:   {loss_median:.4f} [{loss_q1:.4f} - {loss_q3:.4f}]")
                print(f"    Entropy Abs:   {entropy_median:.4f} [{entropy_q1:.4f} - {entropy_q3:.4f}]") 
            print()  # Empty line between models
        print()  # Empty line between neurons 
   
    
if __name__ == "__main__":
    entropy_neuron_indices = [584, 2123, 2870]  # Small
    #entropy_neuron_indices = [3144, 1083, 1108] # Medium
    models = ['baseline', 'finetuned', 'noLN']
    model_size = 'small'
    save_path = f'figures/{model_size}'
    os.makedirs(save_path, exist_ok=True)

    model_results = {}
    for model_type in models:
        # Load each model individually to avoid OOM
        model = ModelFactory([model_type],
                             model_dir='../models',
                             model_size=model_size).models[model_type]
        # Create analyzer
        analyzer = AblationAnalyzer(model=model, model_name=model_type)
        # Run analysis
        analyzer.extract_activations()
        analyzer.run_simulation()
        results = analyzer.run_ablation(entropy_neuron_indices)
       
        # Save model results 
        model_results[model_type] = results
        
        # Free up cache for next model
        del model, analyzer
        t.cuda.empty_cache()

    # Print summary results
    print_ablation_summary(model_results)
    
    # Plot loss effect
    plot_model_comparison(
        model_results,
        model_size,
        save_path=f'{save_path}/all_models_loss_effect.png',
        metric='loss_effect'
    )
    
    # Plot entropy effect
    plot_model_comparison(
        model_results,
        model_size,
        save_path=f'{save_path}/all_models_entropy_effect.png',
        metric='entropy_effect'
    )
# %%
