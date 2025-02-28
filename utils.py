import torch
from std_dicts import std_dicts

def remove_layernorm(model_name, model):
    """Remove LayerNorm from either GPT-2 or Pythia model"""
    is_pythia = "pythia" in model_name.lower()
    
    if is_pythia:
        # For Pythia models
        for block in model.gpt_neox.layers:
            # Effectively disable layernorm by setting epsilon very high
            block.input_layernorm.eps = 1e12
            # Scale the weights to reduce their impact
            block.input_layernorm.weight.data *= 1e6
            
            block.post_attention_layernorm.eps = 1e12
            block.post_attention_layernorm.weight.data *= 1e6
            
        # Final layer norm
        model.gpt_neox.final_layer_norm.eps = 1e12
        model.gpt_neox.final_layer_norm.weight.data *= 1e6
    else:
        # For GPT-2 models
        for block in model.transformer.h:
            block.ln_1.weight.data = block.ln_1.weight.data * 1e6
            block.ln_1.eps = 1e12
            block.ln_2.weight.data = block.ln_2.weight.data * 1e6
            block.ln_2.eps = 1e12
        
        model.transformer.ln_f.weight.data = model.transformer.ln_f.weight.data * 1e6
        model.transformer.ln_f.eps = 1e12
