import torch
from std_dicts import std_dicts

def remove_layernorm(model_name: str, model_hf):
    std_dict = std_dicts[model_name]['std_dict']
    
    # Now kill the layer norm by setting layer_norm_epsilon to 1e12, and multiplied the ln scaling parameters by 1e6
    n_layers = len(model_hf.transformer.h)
    for id, block in enumerate(model_hf.transformer.h):
        with torch.no_grad():
            # Get the standard deviations from the std_dict
            ln1_std = std_dict[f'blocks.{id}.hook_resid_pre']
            ln2_std = std_dict[f'blocks.{id}.hook_resid_mid']
            block.ln_1.weight.data *= 1e6 / ln1_std
            block.ln_2.weight.data *= 1e6 / ln2_std
            block.ln_1.eps = 1e12
            block.ln_2.eps = 1e12
    with torch.no_grad():
        lnf_std = std_dict[f'blocks.{n_layers-1}.hook_resid_post']
        model_hf.transformer.ln_f.weight.data *= 1e6 / lnf_std
        model_hf.transformer.ln_f.eps = 1e12
    return model_hf
