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
from pile_eval import preprocess_pile_dataset

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
    model, tokenizer = load_model("gpt2", remove_ln=False)
    sample(model, tokenizer, "to be or not to")

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
    
    # Calculate sink rate for all layers
    all_layers_sink_rate = calculate_sink_rate(model, attentions)
    
    # Calculate sink rate for layer 3 only
    layer3_attentions = [attentions[2]]  # Layer 3 is index 2 (0-based)
    layer3_sink_rate = calculate_sink_rate(model, layer3_attentions)
    
    return all_layers_sink_rate, layer3_sink_rate

def main():
    # Define model names
    original_model_name = "schaeff/gpt-2medium_vanilla500"
    no_ln_model_name = "submarat/gpt2-medium-noln-aux"
    
    # Load both models
    print("Loading models...")
    model_original, tokenizer_original = load_model(original_model_name)
    model_no_ln, tokenizer_no_ln = load_model(no_ln_model_name)
    
    # Load Pile dataset
    print("Loading Pile dataset...")
    processed_examples, _ = preprocess_pile_dataset("pile-apollo-luca", "gpt2-medium", 1000)
    
    # Process samples in batches
    batch_size = 8
    original_all_sink_rates = []
    original_layer3_sink_rates = []
    no_ln_all_sink_rates = []
    no_ln_layer3_sink_rates = []
    
    print("Calculating sink rates...")
    for i in tqdm(range(0, len(processed_examples), batch_size)):
        batch_examples = processed_examples[i:i + batch_size]
        # Truncate to 512 tokens
        input_ids = torch.stack([torch.tensor(example[:512]) for example in batch_examples]).to(device)
        
        # Calculate sink rates for both models
        original_all, original_layer3 = calculate_batch_sink_rates(model_original, input_ids)
        no_ln_all, no_ln_layer3 = calculate_batch_sink_rates(model_no_ln, input_ids)
        
        original_all_sink_rates.append(original_all.item())
        original_layer3_sink_rates.append(original_layer3.item())
        no_ln_all_sink_rates.append(no_ln_all.item())
        no_ln_layer3_sink_rates.append(no_ln_layer3.item())
    
    # Convert to numpy arrays for statistics
    original_all_sink_rates = np.array(original_all_sink_rates)
    original_layer3_sink_rates = np.array(original_layer3_sink_rates)
    no_ln_all_sink_rates = np.array(no_ln_all_sink_rates)
    no_ln_layer3_sink_rates = np.array(no_ln_layer3_sink_rates)
    
    # Calculate statistics for all layers
    print("\nSink Rate Statistics (All Layers):")
    print(f"{original_model_name}:")
    print(f"  Mean = {np.mean(original_all_sink_rates):.4f}")
    print(f"  95% CI = [{np.percentile(original_all_sink_rates, 2.5):.4f}, {np.percentile(original_all_sink_rates, 97.5):.4f}]")
    print(f"  IQR = [{np.percentile(original_all_sink_rates, 25):.4f}, {np.percentile(original_all_sink_rates, 75):.4f}]")
    print(f"  Median = {np.median(original_all_sink_rates):.4f}")
    
    print(f"\n{no_ln_model_name}:")
    print(f"  Mean = {np.mean(no_ln_all_sink_rates):.4f}")
    print(f"  95% CI = [{np.percentile(no_ln_all_sink_rates, 2.5):.4f}, {np.percentile(no_ln_all_sink_rates, 97.5):.4f}]")
    print(f"  IQR = [{np.percentile(no_ln_all_sink_rates, 25):.4f}, {np.percentile(no_ln_all_sink_rates, 75):.4f}]")
    print(f"  Median = {np.median(no_ln_all_sink_rates):.4f}")
    
    # Calculate statistics for layer 3
    print("\nSink Rate Statistics (Layer 3):")
    print(f"{original_model_name}:")
    print(f"  Mean = {np.mean(original_layer3_sink_rates):.4f}")
    print(f"  95% CI = [{np.percentile(original_layer3_sink_rates, 2.5):.4f}, {np.percentile(original_layer3_sink_rates, 97.5):.4f}]")
    print(f"  IQR = [{np.percentile(original_layer3_sink_rates, 25):.4f}, {np.percentile(original_layer3_sink_rates, 75):.4f}]")
    print(f"  Median = {np.median(original_layer3_sink_rates):.4f}")
    
    print(f"\n{no_ln_model_name}:")
    print(f"  Mean = {np.mean(no_ln_layer3_sink_rates):.4f}")
    print(f"  95% CI = [{np.percentile(no_ln_layer3_sink_rates, 2.5):.4f}, {np.percentile(no_ln_layer3_sink_rates, 97.5):.4f}]")
    print(f"  IQR = [{np.percentile(no_ln_layer3_sink_rates, 25):.4f}, {np.percentile(no_ln_layer3_sink_rates, 75):.4f}]")
    print(f"  Median = {np.median(no_ln_layer3_sink_rates):.4f}")

def visualize_low_sink_rate_attention(model, tokenizer, prompt, sink_rate_threshold=0.1):
    """Visualize attention patterns for cases where sink rate is below threshold."""
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k,v in inputs.items()}
    
    # Get attention patterns
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention tensors
    attentions = outputs.attentions
    
    # Calculate sink rate for each layer
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Plot attention patterns
    import matplotlib.pyplot as plt
    
    for layer in range(num_layers):
        # Calculate sink rate for this layer
        layer_attentions = [attentions[layer]]
        layer_sink_rate = calculate_sink_rate(model, layer_attentions)
        
        # Only visualize if sink rate is below threshold
        if layer_sink_rate < sink_rate_threshold:
            fig, axes = plt.subplots(num_heads, 1, figsize=(8, 3*num_heads))
            fig.suptitle(f'Layer {layer} Attention Patterns (Sink Rate: {layer_sink_rate:.3f})')
            
            for head in range(num_heads):
                # Get attention weights
                attn = attentions[layer][0, head].cpu()
                
                # Plot heatmap
                im = axes[head].imshow(attn, cmap='viridis')
                axes[head].set_title(f'Head {head}')
                
                # Set tick labels
                if len(tokens) < 10:  # Only show ticks for reasonably short sequences
                    axes[head].set_xticks(range(len(tokens)))
                    axes[head].set_yticks(range(len(tokens)))
                    axes[head].set_xticklabels(tokens, rotation=45, fontsize=8)
                    axes[head].set_yticklabels(tokens, fontsize=8)
                else:
                    axes[head].set_xticks([])
                    axes[head].set_yticks([])
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show()

def test_low_sink_rate_visualization():
    # Load model
    print("Loading model...")
    model, tokenizer = load_model("gpt2", remove_ln=False)
    
    # Example prompts that might have low sink rates
    prompts = [
        "The quick brown fox jumps over the lazy dog",
        "In a world where technology advances rapidly",
        "The sun sets behind the mountains, casting long shadows",
        "She carefully arranged the flowers in the vase",
        "The ancient ruins stood silently in the desert"
    ]
    
    for prompt in prompts:
        print(f"\nAnalyzing prompt: '{prompt}'")
        visualize_low_sink_rate_attention(model, tokenizer, prompt)

if __name__ == "__main__":
    test_low_sink_rate_visualization()


