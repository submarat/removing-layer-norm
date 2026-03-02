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
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")
    lora_rank: int = Field(default=64, description="LoRA adapter rank")

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

def make_gpt2_lora(lr, rank=64, skip_eot=False):
    """GPT-2 small with low-rank LoRA adapters."""
    model_name = "gpt2"
    n_layers = 12
    lora_rank = rank

    base_batch_size = 40
    block_size = 1024
    target_batch_tokens = 2**19
    warmup_steps = 30
    max_grad_norm = 0.3

    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)

    learning_rate = lr
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": lr / 2}

    gap_ln2 = 5
    gap_ln1qk = 5
    gap_ln1v = 8
    gap_lnf = None

    start_ln2 = 30
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v

    if skip_eot:
        gap_eot = 10
        gap_bos = 10
        start_eot = 999999
        start_bos = 999999
        max_steps = start_lnf + 200
    else:
        gap_eot = 10
        gap_bos = 10
        start_eot = start_lnf + 30
        start_bos = start_eot + n_layers * gap_eot + 30
        max_steps = start_bos + n_layers * gap_bos + 200

    kw = {k: v for k, v in locals().items() if k not in ('rank', 'skip_eot')}
    return FinetuneConfig(**kw)

def make_gpt2_medium_lora(lr, rank=64, max_steps=700):
    """GPT-2 medium with low-rank LoRA adapters."""
    model_name = "gpt2-medium"
    n_layers = 24
    lora_rank = rank

    base_batch_size = 20
    block_size = 1024
    target_batch_tokens = 2**19
    warmup_steps = 30
    save_steps = 100
    max_grad_norm = 0.3

    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)

    learning_rate = lr
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": lr / 2}

    gap_ln2 = 5
    gap_ln1qk = 5
    gap_ln1v = 8
    gap_lnf = None
    gap_eot = 5
    gap_bos = 5

    start_ln2 = 30
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 30
    start_bos = start_eot + n_layers * gap_eot + 30

    kw = {k: v for k, v in locals().items() if k != 'rank'}
    return FinetuneConfig(**kw)

def make_gpt2_large_lora(lr, rank=64, conservative=False):
    """GPT-2 large with low-rank LoRA adapters."""
    model_name = "gpt2-large"
    n_layers = 36
    lora_rank = rank

    base_batch_size = 10
    block_size = 1024
    target_batch_tokens = 2**19
    save_steps = 100

    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)

    learning_rate = lr
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": lr / 2}

    if conservative:
        warmup_steps = 50
        max_grad_norm = 0.2
        gap_ln2 = 8
        gap_ln1qk = 8
        gap_ln1v = 12
        gap_eot = 8
        gap_bos = 8
        start_ln2 = 40
    else:
        warmup_steps = 30
        max_grad_norm = 0.3
        gap_ln2 = 5
        gap_ln1qk = 5
        gap_ln1v = 8
        gap_eot = 5
        gap_bos = 5
        start_ln2 = 30

    gap_lnf = None
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 40
    start_bos = start_eot + n_layers * gap_eot + 40
    max_steps = start_bos + n_layers * gap_bos + 200

    kw = {k: v for k, v in locals().items() if k not in ('rank', 'conservative')}
    return FinetuneConfig(**kw)

def make_gpt2_xl_lora(lr, rank=64, conservative=False):
    """GPT-2 XL with low-rank LoRA adapters."""
    model_name = "gpt2-xl"
    n_layers = 48
    lora_rank = rank

    base_batch_size = 4
    block_size = 1024
    target_batch_tokens = 2**19
    gradient_checkpointing = True
    save_steps = 100

    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)

    learning_rate = lr
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": lr / 2}

    if conservative:
        warmup_steps = 50
        max_grad_norm = 0.2
        gap_ln2 = 8
        gap_ln1qk = 8
        gap_ln1v = 12
        gap_eot = 8
        gap_bos = 8
        start_ln2 = 40
    else:
        warmup_steps = 30
        max_grad_norm = 0.3
        gap_ln2 = 6
        gap_ln1qk = 6
        gap_ln1v = 10
        gap_eot = 5
        gap_bos = 5
        start_ln2 = 40

    gap_lnf = None
    start_ln1qk = start_ln2 + n_layers * gap_ln2
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    start_lnf = start_ln1v + n_layers * gap_ln1v
    start_eot = start_lnf + 40
    start_bos = start_eot + n_layers * gap_eot + 40
    max_steps = start_bos + n_layers * gap_bos + 200

    kw = {k: v for k, v in locals().items() if k not in ('rank', 'conservative')}
    return FinetuneConfig(**kw)


def make_gpt2_xl_lora_resume_from_500():
    """Resume XL from checkpoint-500 with more conservative ln1qk schedule.
    Original conservative: gap_ln1qk=8, at step 500 blocks 0-8 had ln1qk disabled.
    This config: start_ln1qk=392, gap_ln1qk=12 so at step 500 same state, then
    remaining 39 blocks get 12 steps each (vs 8) for gentler removal."""
    model_name = "gpt2-xl"
    n_layers = 48
    lora_rank = 64

    base_batch_size = 4
    block_size = 1024
    target_batch_tokens = 2**19
    gradient_checkpointing = True
    save_steps = 100

    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)

    learning_rate = 5e-4
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": 2.5e-4}
    warmup_steps = 50
    max_grad_norm = 0.2

    # Match original conservative ln2 schedule (already done by step 500)
    gap_ln2 = 8
    start_ln2 = 40
    # Adjusted ln1qk: at step 500, (500-392)/12=9 blocks done, matching checkpoint
    start_ln1qk = 392
    gap_ln1qk = 12
    # Remaining phases with conservative gaps
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk  # 968
    gap_ln1v = 12
    gap_lnf = None
    start_lnf = start_ln1v + n_layers * gap_ln1v
    gap_eot = 8
    gap_bos = 8
    start_eot = start_lnf + 40
    start_bos = start_eot + n_layers * gap_eot + 40
    max_steps = start_bos + n_layers * gap_bos + 200

    return FinetuneConfig(**locals())


def make_gpt2_xl_lora_resume_from_500_v2():
    """Resume XL from checkpoint-500, v2: even more conservative to get past ln1qk.
    - gap_ln1qk=16 (vs 12), LR=3e-4 (vs 5e-4), max_grad_norm=0.15, aux_loss=0.003."""
    model_name = "gpt2-xl"
    n_layers = 48
    lora_rank = 64

    base_batch_size = 4
    block_size = 1024
    target_batch_tokens = 2**19
    gradient_checkpointing = True
    save_steps = 100

    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)

    learning_rate = 3e-4
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": 1.5e-4}
    warmup_steps = 50
    max_grad_norm = 0.15
    aux_loss_weight = 0.0  # Disabled - was causing loss explosion; LR+gap changes first

    gap_ln2 = 8
    start_ln2 = 40
    # At step 500: (500-356)/16=9 blocks done
    start_ln1qk = 356
    gap_ln1qk = 16
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk  # 1124
    gap_ln1v = 16
    gap_lnf = None
    start_lnf = start_ln1v + n_layers * gap_ln1v
    gap_eot = 8
    gap_bos = 8
    start_eot = start_lnf + 40
    start_bos = start_eot + n_layers * gap_eot + 40
    max_steps = start_bos + n_layers * gap_bos + 200

    return FinetuneConfig(**locals())


def make_gpt2_xl_lora_resume_from_600_v3():
    """Resume XL from checkpoint-600 (last good before collapse), v3: lower LR=2e-4."""
    model_name = "gpt2-xl"
    n_layers = 48
    lora_rank = 64

    base_batch_size = 4
    block_size = 1024
    target_batch_tokens = 2**19
    gradient_checkpointing = True
    save_steps = 100

    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)

    learning_rate = 2e-4
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": 1e-4}
    warmup_steps = 50
    max_grad_norm = 0.15
    aux_loss_weight = 0.0

    gap_ln2 = 8
    start_ln2 = 40
    # At step 600: (600-360)/16=15 blocks done
    start_ln1qk = 360
    gap_ln1qk = 16
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk
    gap_ln1v = 16
    gap_lnf = None
    start_lnf = start_ln1v + n_layers * gap_ln1v
    gap_eot = 8
    gap_bos = 8
    start_eot = start_lnf + 40
    start_bos = start_eot + n_layers * gap_eot + 40
    max_steps = start_bos + n_layers * gap_bos + 200

    return FinetuneConfig(**locals())


def make_gpt2_xl_lora_resume_from_600_v4():
    """Resume XL from checkpoint-600, v4: gap_ln1qk=24, LR=1.5e-4, save_steps=50."""
    model_name = "gpt2-xl"
    n_layers = 48
    lora_rank = 64

    base_batch_size = 4
    block_size = 1024
    target_batch_tokens = 2**19
    gradient_checkpointing = True
    save_steps = 50

    batch_size = base_batch_size
    desired_batch_size = target_batch_tokens / block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)

    learning_rate = 1.5e-4
    lr_scheduler_type = 'cosine_with_min_lr'
    lr_scheduler_kwargs = {"min_lr": 7.5e-5}
    warmup_steps = 50
    max_grad_norm = 0.15
    aux_loss_weight = 0.0

    gap_ln2 = 8
    start_ln2 = 40
    # At step 600: (600-240)/24=15 blocks done
    start_ln1qk = 240
    gap_ln1qk = 24
    start_ln1v = start_ln1qk + n_layers * gap_ln1qk  # 1392
    gap_ln1v = 24
    gap_lnf = None
    start_lnf = start_ln1v + n_layers * gap_ln1v
    gap_eot = 8
    gap_bos = 8
    start_eot = start_lnf + 40
    start_bos = start_eot + n_layers * gap_eot + 40
    max_steps = start_bos + n_layers * gap_bos + 200

    return FinetuneConfig(**locals())


def make_gpt2_xl_lora_resume_from_600_v5():
    """Resume XL from checkpoint-600, v5: same as v4 but base_batch_size=8 (flash attn saves memory)."""
    cfg = make_gpt2_xl_lora_resume_from_600_v4()
    # Override batch size - with flash attention we can use larger batches
    base_batch_size = 8
    batch_size = base_batch_size
    desired_batch_size = cfg.target_batch_tokens / cfg.block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    d["batch_size"] = batch_size
    d["gradient_accumulation_steps"] = gradient_accumulation_steps
    d["base_batch_size"] = base_batch_size
    return FinetuneConfig(**d)


def make_gpt2_xl_lora_resume_from_600_v6():
    """Resume XL, v6: base_batch_size=16 for better GPU utilization (~40GB on A100 80GB)."""
    cfg = make_gpt2_xl_lora_resume_from_600_v4()
    base_batch_size = 16
    batch_size = base_batch_size
    desired_batch_size = cfg.target_batch_tokens / cfg.block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    d["batch_size"] = batch_size
    d["gradient_accumulation_steps"] = gradient_accumulation_steps
    d["base_batch_size"] = base_batch_size
    return FinetuneConfig(**d)


def make_gpt2_xl_lora_resume_from_600_v7():
    """Resume XL, v7: base_batch_size=32, no gradient checkpointing - max GPU utilization."""
    cfg = make_gpt2_xl_lora_resume_from_600_v4()
    base_batch_size = 32
    batch_size = base_batch_size
    desired_batch_size = cfg.target_batch_tokens / cfg.block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    d["batch_size"] = batch_size
    d["gradient_accumulation_steps"] = gradient_accumulation_steps
    d["base_batch_size"] = base_batch_size
    d["gradient_checkpointing"] = False
    return FinetuneConfig(**d)


def make_gpt2_xl_lora_resume_from_600_v8():
    """Resume XL, v8: base_batch_size=64 - fill 80GB (v7 used ~47GB)."""
    cfg = make_gpt2_xl_lora_resume_from_600_v4()
    base_batch_size = 64
    batch_size = base_batch_size
    desired_batch_size = cfg.target_batch_tokens / cfg.block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    d["batch_size"] = batch_size
    d["gradient_accumulation_steps"] = gradient_accumulation_steps
    d["base_batch_size"] = base_batch_size
    d["gradient_checkpointing"] = False
    return FinetuneConfig(**d)


def make_gpt2_xl_lora_resume_from_600_v9():
    """Resume XL, v9: base_batch_size=48 (between v7=32 and v8=64)."""
    cfg = make_gpt2_xl_lora_resume_from_600_v4()
    base_batch_size = 48
    batch_size = base_batch_size
    desired_batch_size = cfg.target_batch_tokens / cfg.block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    d["batch_size"] = batch_size
    d["gradient_accumulation_steps"] = gradient_accumulation_steps
    d["base_batch_size"] = base_batch_size
    d["gradient_checkpointing"] = False
    return FinetuneConfig(**d)


def make_gpt2_xl_lora_resume_from_600_v10():
    """Resume XL, v10: base_batch_size=128, grad_accum=4 - works when resuming (LNs disabled)."""
    cfg = make_gpt2_xl_lora_resume_from_600_v4()
    base_batch_size = 128
    batch_size = base_batch_size
    desired_batch_size = cfg.target_batch_tokens / cfg.block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    d["batch_size"] = batch_size
    d["gradient_accumulation_steps"] = gradient_accumulation_steps
    d["base_batch_size"] = base_batch_size
    d["gradient_checkpointing"] = False
    return FinetuneConfig(**d)


def make_gpt2_xl_lora_resume_from_600_v11():
    """Resume XL, v11: base_batch_size=256, grad_accum=2 - higher utilization when resuming."""
    cfg = make_gpt2_xl_lora_resume_from_600_v4()
    base_batch_size = 256
    batch_size = base_batch_size
    desired_batch_size = cfg.target_batch_tokens / cfg.block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    d["batch_size"] = batch_size
    d["gradient_accumulation_steps"] = gradient_accumulation_steps
    d["base_batch_size"] = base_batch_size
    d["gradient_checkpointing"] = False
    return FinetuneConfig(**d)


def make_gpt2_xl_lora_resume_from_600_v12():
    """Resume XL, v12: base_batch_size=512, grad_accum=1 - max utilization when resuming."""
    cfg = make_gpt2_xl_lora_resume_from_600_v4()
    base_batch_size = 512
    batch_size = base_batch_size
    desired_batch_size = cfg.target_batch_tokens / cfg.block_size
    gradient_accumulation_steps = int(desired_batch_size // batch_size)  # 512/512=1
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    d["batch_size"] = batch_size
    d["gradient_accumulation_steps"] = gradient_accumulation_steps
    d["base_batch_size"] = base_batch_size
    d["gradient_checkpointing"] = False
    return FinetuneConfig(**d)


FINETUNE_CONFIGS = {
    "gpt2": make_gpt2_standard(),
    "gpt2_aux": make_gpt2_standard_aux(),
    "gpt2-medium_fasttune": make_gpt2_medium_fasttune(),
    "gpt2-medium_fasttune_aux": make_gpt2_medium_fasttune_aux(),
    "gpt2-large": make_gpt2_large(),
    "gpt2-large_aux": make_gpt2_large_aux(),
    "gpt2-xl": make_gpt2_xl(),
    "gpt2-xl_aux": make_gpt2_xl_aux(),
    "gpt2_lora_lr1e-3": make_gpt2_lora(1e-3),
    "gpt2_lora_lr3e-3": make_gpt2_lora(3e-3),
    "gpt2_lora_lr5e-3": make_gpt2_lora(5e-3),
    "gpt2_lora_r128_lr1e-3": make_gpt2_lora(1e-3, rank=128),
    "gpt2_lora_r128_lr3e-3": make_gpt2_lora(3e-3, rank=128),
    "gpt2_lora_r256_lr1e-3": make_gpt2_lora(1e-3, rank=256),
    "gpt2_lora_noeot_lr1e-3": make_gpt2_lora(1e-3, skip_eot=True),
    "gpt2_lora_noeot_lr3e-3": make_gpt2_lora(3e-3, skip_eot=True),
    "gpt2-medium_lora_lr1e-3": make_gpt2_medium_lora(1e-3, max_steps=1200),
    "gpt2-medium_lora_lr1e-3_700": make_gpt2_medium_lora(1e-3, max_steps=700),
    "gpt2-medium_lora_lr3e-3": make_gpt2_medium_lora(3e-3),
    "gpt2-large_lora_lr1e-3": make_gpt2_large_lora(1e-3),
    "gpt2-large_lora_conservative": make_gpt2_large_lora(5e-4, conservative=True),
    "gpt2-large_lora_lr3e-3": make_gpt2_large_lora(3e-3),
    "gpt2-xl_lora_lr1e-3": make_gpt2_xl_lora(1e-3),
    "gpt2-xl_lora_conservative": make_gpt2_xl_lora(5e-4, conservative=True),
    "gpt2-xl_lora_resume_from_500": make_gpt2_xl_lora_resume_from_500(),
    "gpt2-xl_lora_resume_from_500_v2": make_gpt2_xl_lora_resume_from_500_v2(),
    "gpt2-xl_lora_resume_from_600_v3": make_gpt2_xl_lora_resume_from_600_v3(),
    "gpt2-xl_lora_resume_from_600_v4": make_gpt2_xl_lora_resume_from_600_v4(),
    "gpt2-xl_lora_resume_from_600_v5": make_gpt2_xl_lora_resume_from_600_v5(),
    "gpt2-xl_lora_resume_from_600_v6": make_gpt2_xl_lora_resume_from_600_v6(),
    "gpt2-xl_lora_resume_from_600_v7": make_gpt2_xl_lora_resume_from_600_v7(),
    "gpt2-xl_lora_resume_from_600_v8": make_gpt2_xl_lora_resume_from_600_v8(),
    "gpt2-xl_lora_resume_from_600_v9": make_gpt2_xl_lora_resume_from_600_v9(),
    "gpt2-xl_lora_resume_from_600_v10": make_gpt2_xl_lora_resume_from_600_v10(),
    "gpt2-xl_lora_resume_from_600_v11": make_gpt2_xl_lora_resume_from_600_v11(),
    "gpt2-xl_lora_resume_from_600_v12": make_gpt2_xl_lora_resume_from_600_v12(),
    "gpt2-xl_lora_lr3e-3": make_gpt2_xl_lora(3e-3),
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
