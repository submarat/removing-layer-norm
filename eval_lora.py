"""
Evaluate LoRA-finetuned GPT-2 models (with FakeLayerNorm) on:
  1. Pile-10k perplexity
  2. HellaSwag (via lm-evaluation-harness)
  3. MMLU (via lm-evaluation-harness)

Also evaluates the vanilla pretrained baseline for comparison.

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval_lora.py \
        --model-name gpt2 \
        --checkpoint results/gpt2_lora/2026-02-21-16-44-04/checkpoint-300 \
        --output eval_results.md

    # Baseline only:
    CUDA_VISIBLE_DEVICES=1 python eval_lora.py --model-name gpt2 --baseline-only --output eval_results.md
"""

import argparse
import json
import math
import os
import sys
import time
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from pile_eval import preprocess_pile_dataset, evaluate_model_on_pile
from train_lora import load_model_lora
from lora import LoRAConv1D


def load_lora_checkpoint(model_name, checkpoint_path, force_all_fake=False):
    """Reconstruct the LoRA+FakeLayerNorm model and load checkpoint weights."""
    model, trainable, frozen = load_model_lora(model_name)
    sd = torch.load(
        os.path.join(checkpoint_path, "pytorch_model.bin"),
        map_location="cpu",
    )
    nan_keys = [k for k, v in sd.items() if torch.is_floating_point(v) and torch.isnan(v).any()]
    if nan_keys:
        print(f"  WARNING: {len(nan_keys)} keys with NaN in checkpoint!")
        print(f"    Examples: {nan_keys[:3]}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    if force_all_fake:
        print("  Forcing all FakeLayerNorm to is_fake=True")
        for block in model.transformer.h:
            block.ln_1.is_fake.fill_(True)
            block.ln_1.attn_v_is_fake.fill_(True)
            block.ln_2.is_fake.fill_(True)
        model.transformer.ln_f.is_fake.fill_(True)

    model.eval()
    return model


def eval_pile10k(model, model_name, num_samples=10000, batch_size=8):
    """Return average CE loss on Pile-10k.
    Uses a custom eval loop for LoRA models to avoid dtype mismatches."""
    processed_examples, _ = preprocess_pile_dataset(
        "pile-10k", model_name, num_samples=num_samples
    )

    has_lora = any("lora_" in n for n, _ in model.named_parameters())
    if not has_lora:
        loss = evaluate_model_on_pile(model, processed_examples, batch_size=batch_size)
    else:
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = []
        for i in range(0, len(processed_examples), batch_size):
            chunk = processed_examples[i:i+batch_size]
            if len(chunk) < batch_size:
                continue
            batches.append(torch.stack(chunk))
        with torch.no_grad():
            for i, batch in enumerate(batches):
                batch = batch.to(device)
                outputs = model(input_ids=batch, labels=batch)
                total_loss += outputs.loss.item() * batch.numel()
                total_tokens += batch.numel()
                if i % 50 == 0:
                    print(f"  Pile eval batch {i}/{len(batches)}: loss={total_loss/total_tokens:.4f}")
        loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        print(f"  Pile eval final loss: {loss:.4f}")

    ppl = math.exp(loss) if loss < 100 else float("inf")
    return {"pile10k_loss": round(loss, 4), "pile10k_ppl": round(ppl, 2)}


def eval_lm_harness(model, model_name, tasks, batch_size=16):
    """Run lm-evaluation-harness tasks on a pre-built HF model."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=batch_size,
        dtype="bfloat16",
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
    )

    parsed = {}
    for task_name, task_result in results["results"].items():
        for metric, value in task_result.items():
            if metric in ("alias",):
                continue
            key = f"{task_name}/{metric}"
            if isinstance(value, float):
                parsed[key] = round(value, 4)
            else:
                parsed[key] = value

    return parsed


def eval_model(model, model_name, label, batch_size=16):
    """Run full eval suite on a model. Returns dict of metrics."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"{'='*60}")

    results = {"label": label, "model_name": model_name}

    print("\n--- Pile-10k perplexity ---")
    t0 = time.time()
    pile_res = eval_pile10k(model, model_name, batch_size=batch_size)
    results.update(pile_res)
    print(f"  Loss: {pile_res['pile10k_loss']}, PPL: {pile_res['pile10k_ppl']}  ({time.time()-t0:.0f}s)")

    print("\n--- HellaSwag ---")
    t0 = time.time()
    hs_res = eval_lm_harness(model, model_name, ["hellaswag"], batch_size=batch_size)
    results.update(hs_res)
    print(f"  {hs_res}  ({time.time()-t0:.0f}s)")

    print("\n--- MMLU ---")
    t0 = time.time()
    mmlu_res = eval_lm_harness(model, model_name, ["mmlu"], batch_size=batch_size)
    results.update(mmlu_res)
    print(f"  {mmlu_res}  ({time.time()-t0:.0f}s)")

    return results


def format_results_md(all_results):
    """Format results list into a markdown string."""
    lines = ["# LoRA LayerNorm Removal — Evaluation Results\n"]

    for res in all_results:
        lines.append(f"## {res['label']}\n")
        lines.append(f"**Model**: `{res['model_name']}`\n")

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in sorted(res.items()):
            if k in ("label", "model_name"):
                continue
            lines.append(f"| {k} | {v} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to LoRA checkpoint dir (contains pytorch_model.bin)")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--lora-only", action="store_true")
    parser.add_argument("--output", default="eval_results.md")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force-all-fake", action="store_true",
                        help="Force all FakeLayerNorm to is_fake=True (for early checkpoints)")
    args = parser.parse_args()

    all_results = []
    existing_results = []

    if os.path.exists(args.output):
        print(f"Output file {args.output} exists, will append new results.")

    if not args.lora_only:
        print(f"\nLoading baseline model: {args.model_name}")
        baseline = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        baseline.eval()
        baseline_res = eval_model(baseline, args.model_name,
                                  f"{args.model_name} baseline (pretrained)",
                                  batch_size=args.batch_size)
        all_results.append(baseline_res)
        del baseline
        torch.cuda.empty_cache()

    if not args.baseline_only and args.checkpoint:
        print(f"\nLoading LoRA checkpoint: {args.checkpoint}")
        lora_model = load_lora_checkpoint(
            args.model_name, args.checkpoint, force_all_fake=args.force_all_fake)
        ckpt_name = os.path.basename(args.checkpoint)
        lora_res = eval_model(lora_model, args.model_name,
                              f"{args.model_name} LoRA lr=6e-4 ({ckpt_name}, LN removed)",
                              batch_size=args.batch_size)
        all_results.append(lora_res)
        del lora_model
        torch.cuda.empty_cache()

    md = format_results_md(all_results)
    print(f"\n{'='*60}")
    print(md)

    with open(args.output, "a" if os.path.exists(args.output) else "w") as f:
        f.write(md + "\n")
    print(f"\nResults written to {args.output}")

    results_json = args.output.replace(".md", ".json")
    with open(results_json, "a" if os.path.exists(results_json) else "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"JSON results appended to {results_json}")


if __name__ == "__main__":
    main()
