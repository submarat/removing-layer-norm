# %% [markdown]
# This notebook is used to visualize the attention patterns of the original GPT-2 model and the model without LayerNorm. Most attention patterns are similar except for 11.11 attention sinks are visibly reduced.

# %%
import torch
from std_dicts import std_dicts
from transformers import GPT2LMHeadModel, AutoTokenizer
from utils import remove_layernorm_by_scaling, calculate_sink_rate
from prepare_dataset import prepare_dataset
from tqdm import tqdm
import numpy as np
from scipy import stats

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# %%
def load_model(model_id, remove_ln=False):
    # Load the model and tokenizer using your Hugging Face Hub model ID
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(model_id)
    model = model.to(device)

    if remove_ln:
        model = remove_layernorm_by_scaling("gpt2", model)
    
    # Load the tokenizer
    model_type = model.config.model_type
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.eos_token = tokenizer.eos_token
    
    return model, tokenizer

def sample(model, tokenizer, prompt, max_length=50, num_return_sequences=1, temperature=0.7):
    # Example usage
    text = prompt
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)
    
    # Generate text using sampling
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,  # Adjust max length as needed
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,  # Adjust temperature to control randomness (higher = more random)
        attention_mask=inputs["attention_mask"],
    )
    
    # Decode the outputs
    decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(decoded_outputs)

# %%
from utils import calculate_sink_rate 

def visualize_attention_comparison(model1, tokenizer1, model2, tokenizer2, prompt):
    # %%
    model, tokenizer = load_model("gpt2", remove_ln=False)
    sample(model, tokenizer, "to be or not to")

    # %%
    model, tokenizer = load_model("submarat/gpt2-noln-aux")
    sample(model, tokenizer, "to be or not to")

    # Prepare input for both models
    inputs1 = tokenizer1(prompt, return_tensors="pt")
    inputs1 = {k: v.to(device) for k,v in inputs1.items()}
    
    inputs2 = tokenizer2(prompt, return_tensors="pt")
    inputs2 = {k: v.to(device) for k,v in inputs2.items()}
    
    # Get attention patterns for both models
    with torch.no_grad():
        outputs1 = model1(**inputs1, output_attentions=True)
        outputs2 = model2(**inputs2, output_attentions=True)
    
    # Get attention tensors
    attention1 = outputs1.attentions
    attention2 = outputs2.attentions

    # Calculate the sink rate for both models
    sink_rate1 = calculate_sink_rate(model1, attention1)
    sink_rate2 = calculate_sink_rate(model2, attention2)
    print("Sink rates: ", sink_rate1, sink_rate2)
    
    # Plot attention patterns side by side
    import matplotlib.pyplot as plt
    
    num_layers = len(attention1)
    num_heads = attention1[0].shape[1]
    tokens1 = tokenizer1.convert_ids_to_tokens(inputs1['input_ids'][0])
    tokens2 = tokenizer2.convert_ids_to_tokens(inputs2['input_ids'][0])
    
    for layer in range(num_layers):
        fig, axes = plt.subplots(num_heads, 2, figsize=(12, 3*num_heads))
        fig.suptitle(f'Layer {layer} Attention Patterns Comparison')
        
        for head in range(num_heads):
            # Get attention weights for both models
            attn1 = attention1[layer][0, head].cpu()
            attn2 = attention2[layer][0, head].cpu()
            
            # Plot heatmaps side by side
            im1 = axes[head, 0].imshow(attn1, cmap='viridis')
            axes[head, 0].set_title(f'Original GPT-2: Head {head}')
            
            im2 = axes[head, 1].imshow(attn2, cmap='viridis')
            axes[head, 1].set_title(f'Without LayerNorm: Head {head}')
            
            # Set tick labels for first column
            if len(tokens1) < 10:  # Only show ticks for reasonably short sequences
                axes[head, 0].set_xticks(range(len(tokens1)))
                axes[head, 0].set_yticks(range(len(tokens1)))
                axes[head, 0].set_xticklabels(tokens1, rotation=45, fontsize=8)
                axes[head, 0].set_yticklabels(tokens1, fontsize=8)
                
                axes[head, 1].set_xticks(range(len(tokens2)))
                axes[head, 1].set_yticks(range(len(tokens2)))
                axes[head, 1].set_xticklabels(tokens2, rotation=45, fontsize=8)
                axes[head, 1].set_yticklabels(tokens2, fontsize=8)
            else:
                # For longer sequences, skip detailed labels
                axes[head, 0].set_xticks([])
                axes[head, 0].set_yticks([])
                axes[head, 1].set_xticks([])
                axes[head, 1].set_yticks([])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

def test_visualize_attention_comparison():
    # Load both models
    print("Loading models...")
    model_original, tokenizer_original = load_model("gpt2", remove_ln=False)
    model_no_ln, tokenizer_no_ln = load_model("submarat/gpt2-noln-aux")

    # Example usage with side-by-side comparison
    prompt = "The quick brown fox jumps over the lazy dog"
    print(f"Comparing attention patterns for prompt: '{prompt}'")
    visualize_attention_comparison(model_original, tokenizer_original, model_no_ln, tokenizer_no_ln, prompt)

def calculate_batch_sink_rates(model, input_ids, attention_mask=None):
    """Calculate sink rates for a batch of sequences."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True
        )
    
    # Get attention tensors
    attentions = outputs.attentions
    
    # Calculate sink rate for the batch
    sink_rate = calculate_sink_rate(model, attentions)
    return sink_rate

def main():
    # Load both models
    print("Loading models...")
    model_original, tokenizer_original = load_model("gpt2", remove_ln=False)
    model_no_ln, tokenizer_no_ln = load_model("submarat/gpt2-noln-aux")
    
    # Load and prepare dataset
    print("Loading dataset...")
    tokenized, data_collator = prepare_dataset()
    
    # Get 256 samples from test split
    test_samples = tokenized["test"].select(range(256))
    
    # Print sequence length information
    first_sample = test_samples[0]
    print(f"\nSequence length information:")
    print(f"Length of first sequence: {len(first_sample['input_ids'])}")
    print(f"Sample sequence lengths: {[len(sample['input_ids']) for sample in test_samples[:5]]}")
    
    # Process samples in batches
    batch_size = 8
    original_sink_rates = []
    no_ln_sink_rates = []
    
    print("Calculating sink rates...")
    for i in tqdm(range(0, len(test_samples), batch_size)):
        batch_samples = test_samples["input_ids"][i:i + batch_size]
        batch = torch.stack([torch.tensor(sample, device=device) for sample in batch_samples])
        input_ids = batch.to(device)
        
        # Calculate sink rates for both models
        original_sink_rate = calculate_batch_sink_rates(model_original, input_ids)
        no_ln_sink_rate = calculate_batch_sink_rates(model_no_ln, input_ids)
        
        original_sink_rates.append(original_sink_rate.item())
        no_ln_sink_rates.append(no_ln_sink_rate.item())
    
    # Convert to numpy arrays for statistics
    original_sink_rates = np.array(original_sink_rates)
    no_ln_sink_rates = np.array(no_ln_sink_rates)
    
    # Calculate basic statistics
    original_mean = np.mean(original_sink_rates)
    no_ln_mean = np.mean(no_ln_sink_rates)
    
    # Calculate 95% confidence intervals using bootstrap
    n_bootstrap = 1000
    original_ci = np.percentile(
        [np.mean(np.random.choice(original_sink_rates, len(original_sink_rates))) 
         for _ in range(n_bootstrap)],
        [2.5, 97.5]
    )
    no_ln_ci = np.percentile(
        [np.mean(np.random.choice(no_ln_sink_rates, len(no_ln_sink_rates))) 
         for _ in range(n_bootstrap)],
        [2.5, 97.5]
    )
    
    # Calculate IQR statistics
    original_q1, original_q3 = np.percentile(original_sink_rates, [25, 75])
    no_ln_q1, no_ln_q3 = np.percentile(no_ln_sink_rates, [25, 75])
    
    print("\nSink Rate Statistics:")
    print(f"Original GPT-2:")
    print(f"  Mean = {original_mean:.4f}")
    print(f"  95% CI = [{original_ci[0]:.4f}, {original_ci[1]:.4f}]")
    print(f"  IQR = [{original_q1:.4f}, {original_q3:.4f}]")
    print(f"  Median = {np.median(original_sink_rates):.4f}")
    
    print(f"\nWithout LayerNorm:")
    print(f"  Mean = {no_ln_mean:.4f}")
    print(f"  95% CI = [{no_ln_ci[0]:.4f}, {no_ln_ci[1]:.4f}]")
    print(f"  IQR = [{no_ln_q1:.4f}, {no_ln_q3:.4f}]")
    print(f"  Median = {np.median(no_ln_sink_rates):.4f}")
    
    print(f"\nDifference in means: {no_ln_mean - original_mean:.4f}")
    print(f"Difference in medians: {np.median(no_ln_sink_rates) - np.median(original_sink_rates):.4f}")

if __name__ == "__main__":
    main()


