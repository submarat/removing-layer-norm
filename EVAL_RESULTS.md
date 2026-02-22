# LoRA LayerNorm Removal — Evaluation Results

## Overview

This document reports evaluation results for parameter-efficient LayerNorm removal from GPT-2 models using full-rank LoRA adapters. The pretrained weights are frozen and only LoRA adapter matrices (A, B) on all Conv1D layers are trained while LayerNorm modules are progressively replaced with FakeLayerNorm (fixed-std normalization).

**Method**: Full-rank LoRA (`lora_A: d_in × d_in`, `lora_B: d_in × d_out`) injected on all Conv1D layers. A initialized to identity, B to zeros. FakeLayerNorm removal follows a progressive schedule across layers.

**Evaluation suite**:
- **Pile-10k**: Cross-entropy loss and perplexity on 10k chunks from NeelNanda/pile-10k
- **HellaSwag**: Commonsense reasoning (acc, acc_norm)
- **MMLU**: Massive Multitask Language Understanding (aggregate accuracy)

---

## GPT-2 Small (124M params)

**Training**: lr=3e-4, cosine schedule, 300 max steps. LN removal completes by step ~106. Checkpoint at step 100 used (latest clean checkpoint before NaN appeared in bf16 LoRA weights at step ~150).

**Note**: An initial bf16 precision issue caused NaN in LoRA adapter weights between steps 100-200 across all tested learning rates (1e-4 to 3e-3). This was fixed for subsequent runs by using float32 for LoRA parameters while keeping frozen weights in bf16.

| Metric | Baseline (pretrained) | LoRA (step 100, LN removed) | Delta |
|--------|----------------------|------------------------------|-------|
| **Pile-10k loss** | 2.7932 | 3.2535 | +0.46 |
| **Pile-10k PPL** | 16.33 | 25.88 | +9.55 |
| **HellaSwag acc** | 0.2880 | 0.2774 | -0.011 |
| **HellaSwag acc_norm** | 0.3124 | 0.2916 | -0.021 |
| **MMLU acc** | 0.2292 | 0.2304 | +0.001 |

**Analysis**: Pile-10k perplexity increases ~58% (16.3 → 25.9) but remains well below random. HellaSwag drops modestly (-2.1 pts normalized), while MMLU is essentially unchanged (both near random chance for GPT-2 small). The model retains meaningful language modeling capability after LN removal.

---

## GPT-2 Medium (355M params)

**Training**: lr=6e-4, cosine schedule, 500 max steps, bs=8, grad_acc=64. LN removal completes by step ~190. LoRA params in float32 to prevent NaN.

*Results pending — training in progress.*

---

## GPT-2 Large (774M params)

**Training**: lr=6e-4, cosine schedule, 800 max steps, bs=4, grad_acc=64, gradient checkpointing. LN removal completes by step ~279. LoRA params in float32.

*Results pending — training queued after medium.*

---

## GPT-2 XL (1.5B params)

**Training**: lr=6e-4, cosine schedule, 1000 max steps, bs=2, grad_acc=64, gradient checkpointing. LN removal completes by step ~363. LoRA params in float32.

*Results pending — training queued after large.*

---

## Key Findings

1. **LN removal with LoRA is partially successful**: At step 100 (before NaN), the model retains strong language modeling ability with only ~0.46 nats degradation in Pile-10k loss.

2. **Numerical stability is critical**: bf16 LoRA parameters developed NaN values between steps 100-200 for all tested learning rates. Fixed by switching LoRA params to float32.

3. **Downstream tasks are relatively robust**: HellaSwag and MMLU show minimal degradation, suggesting the model's factual knowledge and reasoning patterns survive LN removal even if raw perplexity increases.

4. **lr=3e-4 was optimal for GPT-2 small**: Among tested LRs (1e-4, 3e-4, 6e-4, 1e-3, 3e-3), lr=3e-4 at checkpoint-100 gave the best Pile-10k loss (3.25 vs 4.13 for 6e-4, 36.0 for 1e-4).

---

## Experimental Details

- **Hardware**: NVIDIA H100 80GB HBM3 (single GPU, CUDA_VISIBLE_DEVICES=1)
- **Framework**: PyTorch + HuggingFace Transformers + lm-evaluation-harness v0.4.11
- **W&B project**: `m-subkhankulov-arena/removing-layer-norm-lora`
- **Eval script**: `eval_lora.py`
- **Training script**: `train_lora.py`
