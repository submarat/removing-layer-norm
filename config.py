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

# Add new Pythia model configurations
def make_pythia_70m():
    # Architecture params
    model_name = "EleutherAI/pythia-70m"
    n_layers = 6
    
    # Training params
    base_batch_size = 16
    max_steps = 300
    block_size = 2048
    target_batch_tokens = 2**19
    warmup_steps = 20
    save_steps = 20

    learning_rate = 3e-4  # Correct learning rate for Pythia-70m
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": 1e-4}  # 10th of learning rate

    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule
    gap_ln2 = 1
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
    aux_loss_weight = 0.01
    
    return FinetuneConfig(**locals())

def make_pythia_70m_simultaneous_lns():
    # Architecture params
    model_name = "EleutherAI/pythia-70m"
    n_layers = 6
    
    # Training params
    base_batch_size = 16
    max_steps = 300
    block_size = 2048
    target_batch_tokens = 2**19
    warmup_steps = 20
    save_steps = 20

    learning_rate = 2e-4  # Correct learning rate for Pythia-70m
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": 1e-4}  # 10th of learning rate

    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule
    gap_ln2 = 3
    gap_ln1qk = 4
    gap_ln1v = 4
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0

    start_ln2 = 50
    start_ln1qk = start_ln2 + n_layers * gap_ln1qk
    start_ln1v = start_ln2 + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v + 10
    start_eot = start_lnf + 2
    start_bos = start_eot + 10
    aux_loss_weight = 0.01
    
    return FinetuneConfig(**locals())


def make_pythia_70m_test():
    # Architecture params
    model_name = "EleutherAI/pythia-70m"
    n_layers = 6
    
    # Training params - minimal values for testing
    base_batch_size = 1
    max_steps = 10
    block_size = 1024  # Fixed: should match actual sequence length
    target_batch_tokens = 2**12
    warmup_steps = 2
    save_steps = 5  # Save checkpoints frequently
    
    # Learning rate and scheduler params
    learning_rate = 1e-10  # Correct learning rate for Pythia-70m
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": 5e-11}  # Half of learning rate
    weight_decay = 0.01
    momentum = 0.9
    
    # Evaluation params
    eval_steps = 100
    num_eval_samples = 1000
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule - minimal gaps for quick testing
    gap_ln2 = 1
    gap_ln1qk = 1
    gap_ln1v = 1
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0
    
    start_ln2 = 2
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 1
    start_bos = start_eot + 1  # Shorter gap for testing
    
    return FinetuneConfig(**locals())


def make_pythia_160m():
    # Architecture params
    model_name = "EleutherAI/pythia-160m"
    n_layers = 12
    
    # Training params
    base_batch_size = 32
    max_steps = 2000
    block_size = 1024
    target_batch_tokens = 2**19
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule
    gap_ln2 = 20
    gap_ln1qk = 20
    gap_ln1v = 30
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0

    start_ln2 = 200
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 20
    start_bos = start_eot + 100
    
    return FinetuneConfig(**locals())

def make_pythia_410m():
    # Architecture params
    model_name = "EleutherAI/pythia-410m"
    n_layers = 24
    
    # Training params
    base_batch_size = 16
    max_steps = 1500
    block_size = 1024
    target_batch_tokens = 2**19
    
    # Calculate derived training params
    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    
    # Calculate layernorm schedule
    gap_ln2 = 10
    gap_ln1qk = 10
    gap_ln1v = 15
    gap_lnf = None
    gap_eot = 0
    gap_bos = 0
    
    start_ln2 = 200
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 20
    start_bos = start_eot + 100
    
    return FinetuneConfig(**locals())

# Add these functions to your config.py file

FINETUNE_CONFIGS = {
    "gpt2": make_gpt2_standard(),
    "gpt2_aux": make_gpt2_standard_aux(),
    "gpt2-medium_fasttune": make_gpt2_medium_fasttune(),
    "gpt2-medium_fasttune_aux": make_gpt2_medium_fasttune_aux(),
    "gpt2-large": make_gpt2_large(),
    "gpt2-large_aux": make_gpt2_large_aux(),
    "gpt2-xl": make_gpt2_xl(),
    "gpt2-xl_aux": make_gpt2_xl_aux(),
    "pythia-70m": make_pythia_70m(),
    "pythia-70m_test": make_pythia_70m_test(),
}

def make_pythia_70m_start10():
    """Early LayerNorm removal start (step 10)"""
    config = make_pythia_70m()
    config.start_ln2 = 10
    config.start_ln1qk = config.start_ln2 + config.n_layers * config.gap_ln2
    config.start_ln1v = config.start_ln1qk + config.n_layers * config.gap_ln1qk
    config.start_lnf = config.start_ln1v + config.n_layers * config.gap_ln1v
    config.start_eot = config.start_lnf + 2
    config.start_bos = config.start_eot + 10
    return config

def make_pythia_70m_start50():
    """Medium LayerNorm removal start (step 50)"""
    config = make_pythia_70m()
    config.start_ln2 = 50
    config.start_ln1qk = config.start_ln2 + config.n_layers * config.gap_ln2
    config.start_ln1v = config.start_ln1qk + config.n_layers * config.gap_ln1qk
    config.start_lnf = config.start_ln1v + config.n_layers * config.gap_ln1v
    config.start_eot = config.start_lnf + 2
    config.start_bos = config.start_eot + 10
    return config

def make_pythia_70m_start100():
    """Late LayerNorm removal start (step 100)"""
    config = make_pythia_70m()
    config.start_ln2 = 100
    config.start_ln1qk = config.start_ln2 + config.n_layers * config.gap_ln2
    config.start_ln1v = config.start_ln1qk + config.n_layers * config.gap_ln1qk
    config.start_lnf = config.start_ln1v + config.n_layers * config.gap_ln1v
    config.start_eot = config.start_lnf + 2
    config.start_bos = config.start_eot + 10
    return config

def make_pythia_70m_aux():
    """With auxiliary loss (0.1 weight)"""
    config = make_pythia_70m()
    config.aux_loss_weight = 0.1
    return config

def make_pythia_70m_start10_aux():
    """Early start + auxiliary loss"""
    config = make_pythia_70m_start10()
    config.aux_loss_weight = 0.1
    return config

def make_pythia_70m_start50_aux():
    """Medium start + auxiliary loss"""
    config = make_pythia_70m_start50()
    config.aux_loss_weight = 0.1
    return config

def make_pythia_70m_start100_aux():
    """Late start + auxiliary loss"""
    config = make_pythia_70m_start100()
    config.aux_loss_weight = 0.1
    return config

# Add these to your FINETUNE_CONFIGS dictionary:
FINETUNE_CONFIGS.update({
    "pythia-70m_start10": make_pythia_70m_start10(),
    "pythia-70m_start50": make_pythia_70m_start50(),
    "pythia-70m_start100": make_pythia_70m_start100(),
    "pythia-70m_aux": make_pythia_70m_aux(),
    "pythia-70m_start10_aux": make_pythia_70m_start10_aux(),
    "pythia-70m_start50_aux": make_pythia_70m_start50_aux(),
    "pythia-70m_start100_aux": make_pythia_70m_start100_aux(),
    "pythia-70m_simultaneous_lns": make_pythia_70m_simultaneous_lns(),
})

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
