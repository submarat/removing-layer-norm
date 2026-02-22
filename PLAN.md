# Plan: Parameter-Efficient LayerNorm Removal via Full-Rank LoRA

## Motivation

The original codebase removes LayerNorm from GPT-2 by finetuning **all** model
parameters while progressively replacing real LayerNorm with FakeLayerNorm (fixed
std).  This works but updates every weight in the model (~124M params for
GPT-2 small).

The question: can we achieve the same LN removal while keeping the pretrained
weights **frozen** and only training lightweight LoRA-style additive adapters on
the linear layers?  The adapters compensate for the distribution shift caused by
removing the normalization.

## Approach

### 1. Full-rank LoRA adapters on every linear layer

For every `nn.Linear` (or GPT-2's `Conv1D`) in the model we inject a parallel
additive path:

```
y = W_frozen @ x + b_frozen          # original (frozen)
  + B @ (A @ x)                       # adapter  (trained)
```

Where `A ∈ R^{r x d_in}` and `B ∈ R^{d_out x r}` with `r = d_in` (full rank).
"Full-rank LoRA" means the adapter is not low-rank — it can in principle
represent any linear correction.  We initialize `A = I` (or small random) and
`B = 0` so the adapter is an identity/zero at init (output unchanged).

Target modules (GPT-2 small, per block):
- `c_attn`  (768 → 2304, i.e. QKV projection)
- `c_proj`  (768 → 768, attention output projection)
- `mlp.c_fc`   (768 → 3072)
- `mlp.c_proj`  (3072 → 768)

Plus the final `lm_head` is tied to `wte` so we skip it.

Total adapter params: ~same order as frozen params, but **only the adapters
are optimized**, so the optimizer state is over adapter params only.

### 2. FakeLayerNorm removal schedule — reused as-is

The existing `LNRemover` / `LNRemoverCallback` machinery and the per-layer
schedule (`start_ln2`, `gap_ln2`, etc.) are reused unchanged.  The only
difference is that during training the model backbone is frozen and only the
LoRA adapter weights receive gradients.

### 3. Freezing strategy

- All pretrained GPT-2 parameters: **frozen** (`requires_grad=False`)
- FakeLayerNorm weight & bias: **frozen** (they carry the original LN gamma/beta
  and are consumed by the fixed-std path)
- LoRA `A` and `B` matrices: **trained**
- Embeddings (`wte`, `wpe`): **frozen**

### 4. LR sweep

Because the adapter learning dynamics differ from full finetuning, we sweep
learning rate.  The existing cosine-with-min-lr scheduler is reused.
We sweep over: `{1e-4, 3e-4, 6e-4, 1e-3, 3e-3}`.

### 5. Configs

We create new config entries in `config.py`:
- `gpt2_lora_lr1e-4` through `gpt2_lora_lr3e-3`

Each uses the **same LN removal schedule** as `gpt2` (the standard config)
but with the LoRA-specific learning rate.

### 6. What we log to W&B

Everything the original logs, plus:
- `lora_trainable_params` — number of trainable adapter parameters
- `lora_frozen_params` — number of frozen parameters
- Run names include the LR for easy comparison.

### 7. Files changed / added

| File | Change |
|------|--------|
| `lora.py` | **NEW** — `LoRALinear` module, injection helpers |
| `train_lora.py` | **NEW** — training entrypoint for LoRA experiments |
| `config.py` | **MODIFIED** — add `gpt2_lora_*` configs |
| `run_lora_sweep.sh` | **NEW** — launcher for the LR sweep on GPU 1 |

### 8. Execution

```bash
CUDA_VISIBLE_DEVICES=1 bash run_lora_sweep.sh
```

This launches one run per LR sequentially on GPU 1 only.

## Results

All 5 LR sweep runs completed on GPU 1 (H100 80GB).  W&B project:
`m-subkhankulov-arena/removing-layer-norm-lora`.

### Summary Table

| Config  | Step 100 | Step 200 | Step 300 | Final loss | Max loss | Verdict |
|---------|----------|----------|----------|------------|----------|---------|
| lr=1e-4 | 3.20     | 3.46     | 3.33     | 11088      | 11088    | Catastrophic divergence |
| lr=3e-4 | 3.18     | 3.41     | 3.27     | 115        | 115      | Divergence (milder) |
| lr=6e-4 | 3.18     | 3.37     | 3.25     | 3699       | 3699     | Best at step 300, late explosion |
| lr=1e-3 | 3.23     | 3.44     | 3.34     | 9.0        | 9.0      | Moderate divergence |
| lr=3e-3 | 3.89     | 13.27    | —        | 8.1 (s237) | 25.7     | Unstable from early on |

GPT-2 small baseline (with LN): ~3.17.

### Key Findings

1. **Gradient dead zone fixed**: The initial implementation used A=0, B=0
   initialization which created a dead gradient saddle (both gradients are
   zero when both matrices are zero). Fixed by initializing A=I (identity),
   B=0 so that `grad(B) = A^T x^T dL/dy = x^T dL/dy ≠ 0`.

2. **All runs eventually diverge**: Even with the best LR (6e-4), the loss
   remains near baseline (~3.25) through the 300-step training window but
   explodes catastrophically afterward. The adapters cannot prevent residual
   stream norm growth once all LayerNorms are removed.

3. **The explosion is delayed, not immediate**: Loss stays near 3.2–3.4
   during and shortly after the LN removal schedule (steps 20–116).
   Divergence starts 100–500 steps later, suggesting accumulated instability
   in the residual stream rather than an acute failure at the point of
   removal.

4. **Higher LR is better during removal but can hurt stability**: lr=3e-3
   destabilizes even during the removal phase. lr=6e-4 offers the best
   trade-off during removal but still diverges later.

### Why Full-Rank LoRA Alone Is Insufficient

The original method works because it updates *all* weights (including
embeddings, attention, MLP) simultaneously, allowing the entire network
to co-adapt to the absence of normalization.  With frozen weights + LoRA
adapters:

- The adapters can only add a linear correction `x @ A @ B` to each
  layer's output. They cannot change how the frozen weights respond to
  the changed input distribution (no longer mean-centered and
  variance-normalized).
- Without LayerNorm, the residual stream norms grow unboundedly. The
  frozen weights amplify this growth, and the adapters cannot counteract
  it fast enough.
- The problem is fundamentally non-linear: LayerNorm's mean-centering
  and variance normalization are non-linear operations that a linear
  adapter cannot replicate.

### Possible Next Steps

- **Unfreeze LN gamma/beta** alongside LoRA adapters (hybrid approach)
- **Slower LN removal schedule** to give adapters more time per layer
- **Add a norm-stabilization auxiliary loss** (as in the `_aux` configs)
  to explicitly penalize residual stream norm growth
- **Train embeddings** alongside adapters (wte, wpe account for the
  initial distribution shift)
