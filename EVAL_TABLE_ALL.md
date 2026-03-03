# LoRA LayerNorm Removal — Full Evaluation Table

All results across metrics and model combinations. `—` = not evaluated.

**Bold** = all LayerNorms removed (LN removal schedule completed).

---

## Core Metrics

| Model | Config | Pile-10k loss | Pile-10k PPL | HellaSwag acc | HellaSwag acc_norm | MMLU acc |
|-------|--------|---------------|--------------|---------------|--------------------|----------|
| **gpt2** | baseline | 2.79 | 16.32 | 0.288 | 0.312 | 0.229 |
| **gpt2** | **LoRA ckpt-600 (all LN removed)** | 5.17 | 176.41 | 0.274 | 0.278 | 0.230 |
| **gpt2-medium** | baseline | 2.49 | 12.11 | 0.333 | 0.394 | 0.230 |
| **gpt2-medium** | **LoRA ckpt-700 (all LN removed)** | 3.38 | 29.51 | 0.306 | 0.340 | 0.230 |
| **gpt2-large** | baseline | 2.48 | 11.98 | 0.364 | 0.454 | 0.232 |
| **gpt2-large** | **LoRA ckpt-1904 (all LN removed)** | 2.61 | 13.54 | 0.352 | 0.434 | 0.231 |
| **gpt2-xl** | baseline | — | — | — | — | — |
| **gpt2-xl** | **LoRA ckpt-3400 (all LN removed)** | 2.68 | 14.56 | 0.353 | 0.422 | 0.230 |

---

## MMLU Subcategories (where available)

| Model | Config | mmlu_humanities | mmlu_stem | mmlu_social_sciences | mmlu_other |
|-------|--------|-----------------|-----------|----------------------|------------|
| **gpt2** | baseline | 0.242 | 0.213 | 0.217 | 0.239 |
| **gpt2** | **LoRA ckpt-600 (all LN removed)** | 0.243 | 0.213 | 0.216 | 0.241 |
| **gpt2-medium** | baseline | 0.242 | 0.215 | 0.217 | 0.239 |
| **gpt2-medium** | **LoRA ckpt-700 (all LN removed)** | 0.247 | 0.207 | 0.217 | 0.241 |
| **gpt2-large** | baseline | 0.245 | 0.216 | 0.216 | 0.243 |
| **gpt2-large** | **LoRA ckpt-1904 (all LN removed)** | 0.243 | 0.215 | 0.219 | 0.243 |
| **gpt2-xl** | **LoRA ckpt-3400 (all LN removed)** | 0.240 | 0.216 | 0.221 | 0.241 |

---

## Pivot: By Metric

### Pile-10k loss

| | gpt2 | gpt2-medium | gpt2-large | gpt2-xl |
|--|------|------------|------------|---------|
| **baseline** | 2.79 | 2.49 | 2.48 | — |
| **LoRA (all LN removed)** | **5.17** (ckpt-600) | **3.38** (ckpt-700) | **2.61** | **2.68** |

### HellaSwag acc_norm

| | gpt2 | gpt2-medium | gpt2-large | gpt2-xl |
|--|------|------------|------------|---------|
| **baseline** | 0.312 | 0.394 | 0.454 | — |
| **LoRA (all LN removed)** | **0.278** (ckpt-600) | **0.340** (ckpt-700) | **0.434** | **0.422** |

### MMLU acc

| | gpt2 | gpt2-medium | gpt2-large | gpt2-xl |
|--|------|------------|------------|---------|
| **baseline** | 0.229 | 0.230 | 0.232 | — |
| **LoRA (all LN removed)** | **0.230** (ckpt-600) | **0.230** (ckpt-700) | **0.231** | **0.230** |

---

## Data Sources

- `EVAL_RESULTS.json` — gpt2 baseline, gpt2 LoRA ckpt-600
- `eval_results.json` — gpt2 baseline
- `eval_gpt2_large_lora.json` — gpt2-large baseline, gpt2-large LoRA ckpt-1904
- `eval_ckpt3400_ln_removed.json` — gpt2-xl LoRA ckpt-3400
- `eval_gpt2_medium_lora.json` — gpt2-medium baseline, gpt2-medium LoRA ckpt-700

**Not yet evaluated:** gpt2-xl baseline, OWT loss, Pile-filtered loss.
