import os
import torch

from train import load_model
# from transformers.modeling_utils import load_sharded_checkpoint


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def extract_std_from_checkpoint(model_name, ckpt_path):
    # Load model and replace with FakeLayerNorm
    ckpt_model = load_model(model_name=model_name, remove_ln=True)
    # Load checkpoint to load in std values
    try:
        # First try loading a single pytorch model file
        missing, unexpected = ckpt_model.load_state_dict(torch.load(os.path.join(ckpt_path, 'pytorch_model.bin')), strict=False)
    except FileNotFoundError:
        # try:
        #     # If that fails, try loading a sharded checkpoint
        #     missing, unexpected = load_sharded_checkpoint(ckpt_model, ckpt_path, strict=False)
        # except Exception as e:
            raise ValueError(f"Could not load checkpoint from {ckpt_path}. Error: FileNotFoundError")

    if missing:
        print(f"Missing keys when loading checkpoint: {len(missing)} keys")
    if unexpected:
        print(f"Unexpected keys when loading checkpoint: {len(unexpected)} keys")

    state_dict = ckpt_model.state_dict()

    std_dict = {}
    # Extract 1st index in case there is BOS special treatment applied
    for id, block in enumerate(ckpt_model.transformer.h):
        # add std to the dict with appropriate key.
        std_dict[f'blocks.{id}.hook_resid_pre'] = state_dict[f'transformer.h.{id}.ln_1.average_std_buffer'][1].item()
        std_dict[f'blocks.{id}.hook_resid_mid'] = state_dict[f'transformer.h.{id}.ln_2.average_std_buffer'][1].item()
    std_dict[f'blocks.{id}.hook_resid_post'] = state_dict['transformer.ln_f.average_std_buffer'][1].item()

    return std_dict


def remove_layernorm_by_scaling(model_hf, std_dict):

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
    # Set the layer_norm_epsilon to 1e12 as it is not a buffer or learned parameter
    model_hf.config.layer_norm_epsilon = 1e12
    return model_hf


def calculate_sink_rate(model, attentions: list[torch.Tensor], eps=0.3):
    """
    Proportion of heads that attend to the sink on average with a coefficient of at least eps.
    As defined in https://openreview.net/pdf?id=78Nn4QJTEN

    Args:
        model: The model to calculate the sink rate for.
        attentions: A list of tensors of shape (B, H, T, T) containing the attentions for each layer.
        eps: The threshold for the sink rate.

    Returns:
        A tensor of shape (B,) containing the sink rate for each input in the batch.
    """

    # Get config from model
    config = model.config
    T = attentions[0].shape[2]
    H = config.n_head
    L = config.n_layer
    B = attentions[0].shape[0]

    # Calculate the sink rate for each input in the batch
    sink_rate = torch.zeros((B,), device=attentions[0].device)
    for attn in attentions:
        # Heads attending to sink
        y = attn[:,:,:,0] > eps # Bool[B, T, H]
        # Number of heads attending to sink for each position
        y = y.sum(dim=(1,2)).float()
        # Proportion of heads attending to sink
        y = y/T
        sink_rate += y

    # Average over the batch of inputs
    sink_rate = sink_rate.mean()

    # Scale the sink rate
    sink_rate = sink_rate/(L*H)
    return sink_rate