"""
Low-rank LoRA adapters for GPT-2 Conv1D layers.

GPT-2 Conv1D forward: y = x @ W + b,  where W is (d_in, d_out).
We wrap this as:     y = x @ W_frozen + b_frozen + (x @ lora_A) @ lora_B

lora_A: (d_in, r)   initialized with Kaiming uniform
lora_B: (r, d_out)  initialized to zeros

At init the adapter output is zero so the model starts at pretrained
behaviour.  Gradients flow through A from the first step because B's
gradient is A^T @ x^T @ dL/dy which is nonzero when A != 0.
"""

import torch
import torch.nn as nn
import math
from transformers.pytorch_utils import Conv1D


class LoRAConv1D(nn.Module):
    """Wraps a frozen Conv1D with a trainable low-rank additive adapter."""

    def __init__(self, original: Conv1D, rank: int = 64):
        super().__init__()
        self.d_in, self.d_out = original.weight.shape
        self.rank = rank

        self.register_buffer("weight_frozen", original.weight.data.clone())
        if original.bias is not None:
            self.register_buffer("bias_frozen", original.bias.data.clone())
        else:
            self.bias_frozen = None

        dev = original.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.d_in, rank, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.d_out, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        orig_dtype = x.dtype
        y = x.to(self.weight_frozen.dtype) @ self.weight_frozen
        if self.bias_frozen is not None:
            y = y + self.bias_frozen
        adapter_out = (x.float() @ self.lora_A) @ self.lora_B
        y = y.float() + adapter_out
        return y.to(orig_dtype)


def inject_lora_adapters(model, rank: int = 64):
    """Replace every Conv1D in the model with a LoRAConv1D wrapper.

    Returns the number of trainable and frozen parameters.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, Conv1D):
            replacements.append((name, module))

    for name, original in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRAConv1D(original, rank=rank))

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_bufs = sum(b.numel() for b in model.buffers())

    print(f"LoRA injection complete (rank={rank}):")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {frozen:,}")
    print(f"  Buffers          : {total_bufs:,}")
    return trainable, frozen
