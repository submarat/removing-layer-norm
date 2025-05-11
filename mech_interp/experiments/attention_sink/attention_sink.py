#%%
import sys
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import load_nln_model, load_finetuned_model
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer


no_ln = load_nln_model()
finetuned = load_finetuned_model()

#%%
prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "In a world of artificial intelligence, humans still play a crucial role.",
    "The sun rises in the east and sets in the west, marking the passage of time.",
    "Four score and seven years ago our fathers brought forth on this continent a new nation."
]
with torch.no_grad():
    _, cache_nln = no_ln.run_with_cache(prompts)
    _, cache_finetuned = finetuned.run_with_cache(prompts)


attention_patterns_nln = {}
for name, tensor in cache_nln.items():
    if 'pattern' in name:
        attention_patterns_nln[name] = tensor

attention_patterns_finetuned = {}
for name, tensor in cache_finetuned.items():
    if 'pattern' in name:
        attention_patterns_finetuned[name] = tensor

#%%
attention_patterns_finetuned.keys()
for k in attention_patterns_finetuned.keys():
    print(k)
    print(attention_patterns_finetuned[k].shape)
    attn_patterns = attention_patterns_finetuned[k].reshape(-1, 19,19) - attention_patterns_nln[k].reshape(-1, 19,19)
    for j in range(len(attn_patterns)):
        plt.imshow(attn_patterns[j].cpu().numpy())
        plt.show()

#%%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Function to get attention patterns for a model using TransformerLens caching
def get_attention_patterns(model, prompt):
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Run the model with caching
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids)
        
    # Extract attention patterns from cache
    attention_patterns = {}
    for name, tensor in cache.items():
        if 'pattern' in name:
            attention_patterns[name] = tensor
    
    return attention_patterns
    