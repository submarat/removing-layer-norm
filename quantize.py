import torch
import sys
import os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.modeling_utils import load_sharded_checkpoint
import torch.nn as nn

# [LOCAL IMPORTS] - Ensure these files exist in your environment
# Assuming these files contain the necessary training/utility logic
import train 
from utils import get_device 
from pile_eval import preprocess_pile_dataset, evaluate_model_on_pile
from prepare_dataset import prepare_dataset

# --- 1. Configuration and Constants ---
# Use parameters that align with your original script's defaults
BASE_MODEL_NAME = "gpt2" # Used for tokenizer/config
LN_FREE_MODEL_ID = "schaeff/gpt2-small_LNFree300" # A known LN-free variant
STANDARD_MODEL_ID = "gpt2"
DATASET_NAME = "pile-apollo" 
NUM_SAMPLES = 5000
BATCH_SIZE = 16
DEVICE = get_device()

# Define the 8-bit Quantization Configuration
quantization_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# --- 2. Model Loading Functions ---

def load_quantized_model(model_id: str, q_config: BitsAndBytesConfig):
    print(f"Loading {model_id} in 8-bit on device_map='auto'...")
    # Load using GPT2LMHeadModel as it's the specific architecture
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=q_config
    )
    return model

def load_fp_model(model_id: str):
    """Loads a model in full precision (FP32/FP16/BF16) and moves to the target device."""
    print(f"Loading {model_id} in Full Precision (FP/Baseline)...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
  
    return model.to(DEVICE)


def run_quantization_comparison(dataset_name, num_samples, batch_size):
    """Executes the full comparison across the four model variants."""
    
    # --- A. Load Models ---
    print("\n--- 1. Loading All Four Model Variants ---")
    
    # 1. Standard (Full Precision)
    standard_fp = load_fp_model(STANDARD_MODEL_ID)
    standard_8bit = load_quantized_model(STANDARD_MODEL_ID, quantization_config_8bit)
    
    LN_free_fp = load_fp_model(LN_FREE_MODEL_ID)
    LN_free_8bit = load_quantized_model(LN_FREE_MODEL_ID, quantization_config_8bit)

    models_to_evaluate = {
        "Standard (FP)": standard_fp,
        "Standard (8-bit)": standard_8bit,
        "LN-free (FP)": LN_free_fp,
        "LN-free (8-bit)": LN_free_8bit,
    }

    # --- B. Load and Preprocess Dataset ---
    print(f"\n--- 2. Loading and Preprocessing Dataset: {DATASET_NAME} ---")
    
    # Use the shared preprocessing function from pile_eval.py/prepare_dataset.py
    # We use the BASE_MODEL_NAME for tokenization and config purposes
    try:
        # Assuming prepare_dataset is used for the openwebtext variant of The Pile
        processed_examples, _ = preprocess_pile_dataset(
            dataset_name=DATASET_NAME, 
            model_name=BASE_MODEL_NAME, 
            num_samples=num_samples
        )
    except NameError:
        print("Error: Could not find preprocess_pile_dataset. Ensure pile_eval.py is in path.")
        return

    # --- C. Evaluation Loop ---
    print("\n--- 3. Running Evaluation for All Models ---")
    results = {}

    for name, model in models_to_evaluate.items():
        # Call the external evaluation function
        # This function should return Cross-Entropy Loss
        ce_loss = evaluate_model_on_pile(model, processed_examples, batch_size)
        
        # PPL is exp(CE Loss)
        perplexity = torch.exp(torch.tensor(ce_loss)).item()
        
        results[name] = {"Loss": ce_loss, "PPL": perplexity}
        print(f"Results for {name}: Loss={ce_loss:.4f}, PPL={perplexity:.2f}")

    # --- D. Print Final Comparison Table ---
    print("\n" + "="*80)
    print("      Quantization vs. LN Removal: Perplexity and Loss Comparison      ")
    print("="*80)
    print(f"{'Model Variant':<20} | {'Precision':<10} | {'CE Loss':<10} | {'Perplexity':<12} | {'PPL Degradation':<20}")
    print("-" * 80)

    # Calculate Baselines
    standard_fp_ppl = results["Standard (FP)"]["PPL"]
    LN_free_fp_ppl = results["LN-free (FP)"]["PPL"]

    # Display Results and Degradation
    for name in models_to_evaluate:
        precision = "FP" if "(FP)" in name else "8-bit"
        loss = results[name]["Loss"]
        ppl = results[name]["PPL"]
        
        # Calculate degradation relative to its FP counterpart
        if "Standard" in name and "8-bit" in name:
            degradation = ppl - standard_fp_ppl
        elif "LN-free" in name and "8-bit" in name:
            degradation = ppl - LN_free_fp_ppl
        else:
            degradation = 0 # Baseline
            
        degradation_str = f"+{degradation:.3f}" if degradation > 0.001 else f"{degradation:.3f}"
        
        print(f"{name.split('(')[0].strip():<20} | {precision:<10} | {loss:<10.4f} | {ppl:<12.3f} | {degradation_str:<20}")

    print("="*80)

if __name__ == "__main__":
    # Note: In a real script, these arguments would come from docopt or argparse
    run_quantization_comparison(
        dataset_name=DATASET_NAME, 
        num_samples=NUM_SAMPLES, 
        batch_size=BATCH_SIZE
    )
