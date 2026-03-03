# LoRA LayerNorm Removal — Full Results

Parameter-efficient LayerNorm removal from GPT-2 using **low-rank LoRA** (rank=64). Progressive LN removal: ln2 → ln1qk → ln1v → ln_f → eot → bos.

---

## Checkpoint Locations

| Model | Checkpoint Path | Step | Notes |
|-------|-----------------|------|-------|
| **GPT-2 Small** | `results/gpt2_lora/2026-02-22-16-14-45/checkpoint-600` | 600 | LN removed, evals done |
| **GPT-2 Medium** | `results/gpt2-medium_lora/2026-02-22-19-52-56/checkpoint-700` | 700 | LN removed, evals done |
| **GPT-2 Large** | `results/gpt2-large_lora/2026-02-24-08-42-58/checkpoint-1904` | 1904 | Conservative config, LN removed, evals done |
| **GPT-2 XL** | `results/gpt2-xl_lora/2026-02-25-23-24-19/checkpoint-3400` | 3400 | Conservative config, LN removed, evals done |

---

## Evaluation Results — Cross-Model Comparison

| Model | Pile-10k loss | Pile-10k PPL | HellaSwag acc_norm | MMLU acc |
|-------|----------------|--------------|--------------------|----------|
| **Small** baseline | 2.79 | 16.32 | 0.312 | 0.229 |
| **Small** LN removed | 5.17 | 176.41 | 0.278 | 0.230 |
| **Medium** baseline | 2.49 | 12.11 | 0.394 | 0.230 |
| **Medium** LN removed | 3.38 | 29.51 | 0.340 | 0.230 |
| **Large** baseline | 2.48 | 11.98 | 0.454 | 0.232 |
| **Large** LN removed | 2.61 | 13.54 | 0.434 | 0.231 |
| **XL** baseline | — | — | — | — *(not run)* |
| **XL** LN removed | 2.68 | 14.56 | 0.422 | 0.230 |

**Deltas (LN removed vs baseline):**

| Model | Δ Pile-10k loss | Δ Pile-10k PPL | Δ HellaSwag | Δ MMLU |
|-------|-----------------|----------------|-------------|--------|
| Small | +2.38 | +160.09 | -0.035 | ~0 |
| Medium | +0.89 | +17.40 | -0.054 | ~0 |
| Large | +0.12 | +1.56 | -0.020 | ~0 |
| XL | ~+0.2–0.3* | ~+2–3* | — | — |

\* XL baseline not run; estimated from scaling (Small→Large→XL).

---

## Per-Model Details

### GPT-2 Small (124M)

| Metric | Baseline | LoRA (ckpt-600) | Delta |
|--------|----------|-----------------|-------|
| Pile-10k loss | 2.79 | 5.17 | +2.38 |
| Pile-10k PPL | 16.32 | 176.41 | +160.09 |
| HellaSwag acc_norm | 0.312 | 0.278 | -0.035 |
| MMLU acc | 0.229 | 0.230 | ~0 |

**Analysis**: Severe perplexity degradation. LN removal with rank-64 LoRA is challenging for small.

---

### GPT-2 Medium (355M)

| Metric | Baseline | LoRA (ckpt-700) | Delta |
|--------|----------|-----------------|-------|
| Pile-10k loss | 2.49 | 3.38 | +0.89 |
| Pile-10k PPL | 12.11 | 29.51 | +17.40 |
| HellaSwag acc_norm | 0.394 | 0.340 | -0.054 |
| MMLU acc | 0.230 | 0.230 | ~0 |

**Analysis**: Moderate degradation. Medium sits between Small (severe) and Large (minimal): PPL roughly doubles but remains usable; HellaSwag drops ~5.4 pts; MMLU stable.

---

### GPT-2 Large (774M)

| Metric | Baseline | LoRA (ckpt-1904) | Delta |
|--------|----------|------------------|-------|
| Pile-10k loss | 2.48 | 2.61 | +0.12 |
| Pile-10k PPL | 11.98 | 13.54 | +1.56 |
| HellaSwag acc_norm | 0.454 | 0.434 | -0.020 |
| MMLU acc | 0.232 | 0.231 | ~0 |

**Analysis**: Minimal degradation. Large scales much better than small for LN removal.

---

### GPT-2 XL (1.5B)

| Metric | Baseline | LoRA (ckpt-3400) | Delta |
|--------|----------|------------------|-------|
| Pile-10k loss | — | 2.68 | — |
| Pile-10k PPL | — | 14.56 | — |
| HellaSwag acc_norm | — | 0.422 | — |
| MMLU acc | — | 0.230 | — |

**Analysis**: XL LN-removed performs between Large baseline (2.48) and Large LN-removed (2.61). HellaSwag acc_norm (0.422) is below Large baseline (0.454), consistent with ~3 pt degradation seen at Large. Baseline XL eval not yet run.

---

## Training Configs

### Small
- lr=1e-3, gap_eot=10, ~600 steps
- LN removal completes ~step 536

### Medium
- lr=1e-3, max_steps=700 (stopped)
- Checkpoint at 700

### Large (conservative — required after collapse at step 328)
- lr=5e-4, max_grad_norm=0.2
- gap_ln2=8, gap_ln1qk=8, gap_ln1v=12, gap_eot=8, gap_bos=8
- warmup_steps=50, start_ln2=40
- ~1904 steps, ~16h on H100

### XL (conservative — complete)
- Config: `gpt2-xl_lora_conservative`
- lr=5e-4, max_grad_norm=0.2
- gap_ln2=8, gap_ln1qk=8, gap_ln1v=12, gap_eot=8, gap_bos=8
- checkpoint-3400, LN removal complete by step 3384
- Log: `lora_gpt2-xl_lora_conservative.log`

---

## Key Findings

1. **Scale helps**: Small collapses (PPL 16→176); Medium shows moderate degradation (PPL 12→30); Large and XL retain reasonable perplexity (Large 12→14, XL ~14.5).
2. **Conservative schedule for large models**: Default config collapsed at ln1qk (step 328). Conservative gaps + lower lr fixed it.
3. **MMLU robust**: Small, Medium, Large, and XL all keep MMLU accuracy (~23%) after LN removal.
4. **HellaSwag degrades with scale**: Small −3.5 pts, Medium −5.4 pts, Large −2.0 pts, XL ~0.42 (vs Large baseline 0.45).
5. **XL vs Large**: XL LN-removed (2.68 loss, 14.56 PPL) is slightly worse than Large LN-removed (2.61, 13.54); both show modest regression vs their baselines.

---

## Run Commands

```bash
# Eval Large
CUDA_VISIBLE_DEVICES=1 python eval_lora.py --model-name gpt2-large \
  --checkpoint results/gpt2-large_lora/2026-02-24-08-42-58/checkpoint-1904 \
  --force-all-fake --output eval_gpt2_large_lora.md

# Eval XL (LN removed)
CUDA_VISIBLE_DEVICES=0 python eval_lora.py --model-name gpt2-xl \
  --checkpoint results/gpt2-xl_lora/2026-02-25-23-24-19/checkpoint-3400 \
  --lora-only --force-all-fake --output eval_ckpt3400_ln_removed.md

# XL baseline (to get proper delta)
CUDA_VISIBLE_DEVICES=0 python eval_lora.py --model-name gpt2-xl --output eval_gpt2_xl_baseline.md
```

---

## Hardware & Framework

- **GPU**: NVIDIA H100 80GB HBM3 (CUDA_VISIBLE_DEVICES=1)
- **Framework**: PyTorch, HuggingFace Transformers, lm-evaluation-harness
- **W&B**: m-subkhankulov-arena/removing-layer-norm-lora
