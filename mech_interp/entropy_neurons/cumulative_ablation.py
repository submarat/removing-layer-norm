# %%
import sys
import copy
import os
import numpy as np
import pandas as pd
import torch as t
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
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
        for param in self.model.parameters():
            param.requires_grad_(False)
            
        self.model_name = model_name
        self.batch_size = batch_size
        
        self.dataloader = DataLoader(
                dataset_name="luca-pile",
                batch_size=5,
                max_context=512,
                num_samples=1000,
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
                "blocks.11.hook_resid_pre",
                "blocks.11.hook_attn_out",
                "blocks.11.mlp.hook_post",
                "blocks.11.hook_mlp_out"
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
                resid_pre = cache["blocks.11.hook_resid_pre"][:, :-1, :]
                attn = cache["blocks.11.hook_attn_out"][:, :-1, :]
                mid_mlp = cache["blocks.11.mlp.hook_post"][:, :-1, :]
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
        print(f"Reference loss: {self.ref_loss.mean().item():.4f}, Simulated loss: {self.simulated_loss.mean().item():.4f}")
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
            
        loss_mean = total_loss_ablated.mean().cpu().item()
        loss_std = total_loss_ablated.std().cpu().item()
        entropy_mean = total_entropy_ablated.mean().cpu().item()
        entropy_std = total_entropy_ablated.std().cpu().item()
            
        return (loss_mean, loss_std), (entropy_mean, entropy_std)    
    
    
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

        # Always run ablation on all neurons together
        loss, entropy = self._perform_ablation(neuron_indices)
        
        return loss, entropy
   
    
if __name__ == "__main__":
    entropy_neuron_indices = [584, 2123, 2870]
    model_types = ['baseline', 'finetuned', 'noLN']
    model_results = {}
    
    for model_type in model_types: 
        # Load each model individually to avoid OOM
        model = ModelFactory([model_type],
                             model_dir='../models').models[model_type]
    
        # Create analyzer
        analyzer = AblationAnalyzer(model=model, model_name=model_type)
        # Run analysis
        analyzer.extract_activations()
        analyzer.run_simulation()
        
        # Store initial stats - format: [loss_stats, entropy_stats]
        # Each stats is a tuple of (mean, std)
        loss_stats = (analyzer.simulated_loss.mean().item(), analyzer.simulated_loss.std().item())
        entropy_stats = (analyzer.simulated_entropy.mean().item(), analyzer.simulated_entropy.std().item())
        model_results[model_type] = [[loss_stats, entropy_stats]]
        
        for idx in tqdm(range(1, len(entropy_neuron_indices) + 1), desc="Neuron Ablations"):
            indices = entropy_neuron_indices[:idx]
            
            # Get loss and entropy statistics as tuples
            loss_stats, entropy_stats = analyzer.run_ablation(indices)
            model_results[model_type].append([loss_stats, entropy_stats])
            
    # %% 
    # Get default matplotlib colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # FIRST PLOT: Cross-Entropy Loss
    plt.figure(figsize=(10, 6))
    
    for idx, model_type in enumerate(model_types):
        # Extract means and stds for loss
        ce_losses_mean = [r[0][0] for r in model_results[model_type]]  # r[0] is loss_stats, [0] is mean
        
        # Create x-axis values (number of ablated neurons)
        num_ablated = list(range(len(ce_losses_mean)))
        
        # Plot CE Loss
        plt.plot(num_ablated, ce_losses_mean, 'o-', lw=2,
                 color=colors[idx], alpha=0.8, label=model_type)
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of Entropy Neurons Ablated', fontsize=14)
    plt.ylabel('Cross-Entropy Loss', fontsize=14)
    plt.title('Effect of Neuron Ablation on Cross-Entropy Loss', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the CE Loss plot
    plt.savefig('figures/cumulative_ce_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # SECOND PLOT: Entropy
    plt.figure(figsize=(10, 6))
    
    for idx, model_type in enumerate(model_types):
        # Extract means and stds for entropy
        entropies_mean = [r[1][0] for r in model_results[model_type]]  # r[1] is entropy_stats, [0] is mean
        
        # Create x-axis values (number of ablated neurons)
        num_ablated = list(range(len(entropies_mean)))
        
        # Plot Entropy
        plt.plot(num_ablated, entropies_mean, 'o-', lw=2,
                 color=colors[idx], alpha=0.8, label=model_type)    

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of Entropy Neurons Ablated', fontsize=14)
    plt.ylabel('Entropy', fontsize=14)
    plt.title('Effect of Neuron Ablation on Entropy', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the Entropy plot
    plt.savefig('figures/cumulative_entropy_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
# %%
