"""
Usage:
    config.py list
    config.py show <config_name>
"""

from docopt import docopt
from pydantic import BaseModel, Field
from typing import Optional, Dict

class FinetuneConfig(BaseModel):
    # Architecture params
    model_name: str
    n_layers: int
    
    # Training params
    base_batch_size: int
    max_steps: int
    early_stop_step: int = 1_000_000 # i.e. don't stop early by default
    block_size: int = 1024
    target_batch_tokens: int = Field(default=2**19, description="Desired total tokens per batch")
    warmup_steps: int = 100
    weight_decay: float = 0.01
    learning_rate: float = 6e-4
    lr_scheduler_type: str = 'cosine_with_min_lr' #'constant_with_warmup'
    lr_scheduler_kwargs: dict = {"min_lr": 3e-4}
    save_steps: int = 100  # Save checkpoint every 100 steps, for larger models less often
    
    # Evaluation params
    num_eval_samples: int = 1000
    eval_steps: int = 100
    
    # Derived training params
    batch_size: int
    gradient_accumulation_steps: int
    
    # Auxiliary loss params
    aux_loss_weight: float = Field(default=0.0, description="Weight for the auxiliary loss to encourage uniform residual norms")
    gradient_checkpointing: bool = Field(default=False, description="Use gradient checkpointing to save memory")

    # Momentum for recomputing the moving average std which will be fixed at LN removal
    # low momentum: 
    momentum: float = Field(default=0.9, description="Recompute momentum")

    # Layernorm schedule params
    gap_ln2: Optional[int]
    gap_ln1qk: Optional[int]
    gap_ln1v: Optional[int]
    gap_lnf: Optional[int]
    gap_eot: Optional[int]
    gap_bos: Optional[int]
    start_ln2: int
    start_ln1qk: int
    start_ln1v: int
    start_lnf: int
    start_eot: int
    start_bos: int

def make_gpt2_standard():
    # Fast schedule
    # Architecture params
    model_name = "gpt2"
    n_layers = 12
    
    # Training params
    base_batch_size = 40 # A100 with 80 GB VRAM, adapt for other chips
    max_steps = 300
    block_size = 1024
    target_batch_tokens = 2**19
    warmup_steps = 20
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule
    gap_ln2 = 2
    gap_ln1qk = 2
    gap_ln1v = 3
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0

    start_ln2 = 20
    start_ln1qk = start_ln2 + 12 * gap_ln2
    start_ln1v = start_ln1qk + 12 * gap_ln1qk
    start_lnf = start_ln1v + 12 * gap_ln1v
    start_eot = start_lnf + 2
    start_bos = start_eot + 10

    # early_stop_step = start_bos + 40
    
    return FinetuneConfig(**locals())

def make_gpt2_standard_aux():
    # Fast schedule
    # Architecture params
    model_name = "gpt2"
    n_layers = 12
    
    # Training params
    base_batch_size = 32 # A100 with 80 GB VRAM, adapt for other chips
    max_steps = 300
    block_size = 1024
    target_batch_tokens = 2**19
    warmup_steps = 25
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    gradient_checkpointing = True

    # Calculate layernorm schedule
    gap_ln2 = 2
    gap_ln1qk = 2
    gap_ln1v = 3
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0

    start_ln2 = 20
    start_ln1qk = start_ln2 + 12 * gap_ln2
    start_ln1v = start_ln1qk + 12 * gap_ln1qk
    start_lnf = start_ln1v + 12 * gap_ln1v
    start_eot = start_lnf + 2
    start_bos = start_eot + 10

    aux_loss_weight = 0.1
    
    return FinetuneConfig(**locals())

def make_gpt2_medium_fasttune():
    # Architecture params
    model_name = "gpt2-medium"
    n_layers = 24
    
    # Training params
    base_batch_size = 22 # A100 with 80 GB VRAM, adapt for other chips
    max_steps = 500
    block_size = 1024
    target_batch_tokens = 2**19
    warmup_steps = 10  # Shorter warmup due to accelerated schedule
    save_steps = 100

    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule
    gap_ln2 = 2
    gap_ln1qk = 2
    gap_ln1v = 3
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0
    
    start_ln2 = 20
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 2
    start_bos = start_eot + 10

    # early_stop_step = start_bos + 10
    
    return FinetuneConfig(**locals())

def make_gpt2_medium_fasttune_aux():
    # Architecture params
    model_name = "gpt2-medium"
    n_layers = 24
    
    # Training params
    base_batch_size = 22 # A100 with 80 GB VRAM, adapt for other chips
    max_steps = 500
    block_size = 1024
    target_batch_tokens = 2**19
    warmup_steps = 10  # Shorter warmup due to accelerated schedule
    
    gradient_checkpointing = True

    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule
    gap_ln2 = 2
    gap_ln1qk = 2
    gap_ln1v = 3
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0
    
    start_ln2 = 20
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 2
    start_bos = start_eot + 10

    aux_loss_weight = 0.1
    
    return FinetuneConfig(**locals())

def make_gpt2_large():
    # Architecture params
    model_name = "gpt2-large"
    n_layers = 36
    
    # Training params
    base_batch_size = 28 # B200 with 180 GB VRAM, adapt for other chips
    max_steps = 1200
    block_size = 1024
    target_batch_tokens = 2**19
    save_steps = 100
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    gradient_checkpointing = False
    
    learning_rate: float = 3e-4
    lr_scheduler_type: str = 'cosine_with_min_lr' #'constant_with_warmup'
    lr_scheduler_kwargs: dict = {"min_lr": 4e-5}

    warmup_steps = 15
    momentum = 0.9**(base_batch_size/32)
    
    # Calculate layernorm schedule
    gap_ln2 = 4
    gap_ln1qk = 4
    gap_ln1v = 6
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0
    
    start_ln2 = 30
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 2
    start_bos = start_eot + 10

    # early_stop_step = start_bos + 10
    
    return FinetuneConfig(**locals())

def make_gpt2_large_aux():
    # Architecture params
    model_name = "gpt2-large"
    n_layers = 36
    
    # Training params
    base_batch_size = 28 # B200 with 180 GB VRAM, adapt for other chips
    max_steps = 1200
    block_size = 1024
    target_batch_tokens = 2**19
    save_steps = 100
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    gradient_checkpointing = False
    
    learning_rate: float = 3e-4
    lr_scheduler_type: str = 'cosine_with_min_lr' #'constant_with_warmup'
    lr_scheduler_kwargs: dict = {"min_lr": 4e-5}

    warmup_steps = 15
    momentum = 0.9**(base_batch_size/32)

    # Calculate layernorm schedule
    gap_ln2 = 4
    gap_ln1qk = 4
    gap_ln1v = 6
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0
    
    start_ln2 = 30
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 2
    start_bos = start_eot + 10
    
    aux_loss_weight = 0.03

    return FinetuneConfig(**locals())

def make_gpt2_xl():
    # Architecture params
    model_name = "gpt2-xl"
    n_layers = 48
    
    # Training params
    base_batch_size = 18 # B200 with 180 GB VRAM, adapt for other chips
    max_steps = 1200
    block_size = 1024
    target_batch_tokens = 2**19
    save_steps = 100
        
    learning_rate: float = 1e-4
    lr_scheduler_type: str = 'cosine_with_min_lr' #'constant_with_warmup'
    lr_scheduler_kwargs: dict = {"min_lr": 2e-5}
    warmup_steps = 20
    momentum = 0.9**(base_batch_size/32)
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    gradient_checkpointing = False
    
    # Calculate layernorm schedule
    gap_ln2 = 2
    gap_ln1qk = 2
    gap_ln1v = 3
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0
    
    start_ln2 = 20
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 2
    start_bos = start_eot + 10
    
    return FinetuneConfig(**locals())

def make_gpt2_xl_aux():
    # Architecture params
    model_name = "gpt2-xl"
    n_layers = 48
    
    # Training params
    base_batch_size = 18 # B200 with 180 GB VRAM, adapt for other chips
    max_steps = 1200
    block_size = 1024
    target_batch_tokens = 2**19
    save_steps = 100
        
    learning_rate: float = 1e-4
    lr_scheduler_type: str = 'cosine_with_min_lr' #'constant_with_warmup'
    lr_scheduler_kwargs: dict = {"min_lr": 2e-5}
    warmup_steps = 20
    momentum = 0.9**(base_batch_size/32)
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    gradient_checkpointing = False
    
    # Calculate layernorm schedule
    gap_ln2 = 4
    gap_ln1qk = 4
    gap_ln1v = 6
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0
    
    start_ln2 = 50
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 2
    start_bos = start_eot + 10
    
    aux_loss_weight = 0.01

    return FinetuneConfig(**locals())

FINETUNE_CONFIGS = {
    "gpt2": make_gpt2_standard(),
    "gpt2_aux": make_gpt2_standard_aux(),
    "gpt2-medium_fasttune": make_gpt2_medium_fasttune(),
    "gpt2-medium_fasttune_aux": make_gpt2_medium_fasttune_aux(),
    "gpt2-large": make_gpt2_large(),
    "gpt2-large_aux": make_gpt2_large_aux(),
    "gpt2-xl": make_gpt2_xl(),
    "gpt2-xl_aux": make_gpt2_xl_aux(),
}

def main():
    args = docopt(__doc__)
    
    if args["list"]:
        print("\nAvailable configurations:")
        for name in FINETUNE_CONFIGS:
            print(f"  {name}")
        print()
    elif args["show"]:
        config_name = args["<config_name>"]
        if config_name not in FINETUNE_CONFIGS:
            print(f"Error: Configuration '{config_name}' not found")
            return
        config = FINETUNE_CONFIGS[config_name]
        print(f"\nConfiguration '{config_name}':")
        print(config.model_dump_json(indent=2))
        print()

if __name__ == "__main__":
    main() 
