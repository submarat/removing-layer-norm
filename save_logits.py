#!/usr/bin/env python3
"""
Parse high loss samples and save corresponding logits.

Supported input formats:
- Evaluator logs (e.g., 'high_loss_samples.txt') with lines like:
  "Sample i, item j: input_ids=[...]" and optional "Decoded: ..."
- Token:value CSV as in 'high_loss_sample_1.txt', where entries look like
  "220:N/A, 220:7.62, ..." — we reconstruct input_ids by taking the token ids
  in order and ignoring the values.

This does NOT change the evaluator's token/view logic. It reuses model-loading helpers
and recomputes logits for the flagged samples.

Usage examples:
  python save_logits_for_high_loss_samples.py --model schaeff/gpt2-xl_LNFree800 --format transformers --model-name gpt2-xl
  python save_logits_for_high_loss_samples.py --model results/checkpoint-1200 --format fakeln_checkpoint --model-name gpt2

Outputs per sample into 'bad_samples_logits/'.
If input_ids are present in the file, those are used directly. Otherwise, uses the
decoded text to re-tokenize with the provided tokenizer.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

from utils import get_device
from eval_pile import load_hf_model, load_nln_hf_model, load_fakeln_checkpoint
from transformer_lens import HookedTransformer


def parse_high_loss_file(path: str) -> List[Tuple[Optional[List[int]], Optional[str]]]:
    """Parse 'high_loss_samples.txt' and return list of (input_ids, decoded_text).

    Prefers exact input_ids when present; otherwise, falls back to decoded text.
    Supports two observed formats in the repo:
      - "Sample {i}: <numpy-array-str>" and "Decoded: ..."
      - "Sample {i}, item {j}: input_ids=[...]" and "Decoded: ..."
    """
    samples: List[Tuple[Optional[List[int]], Optional[str]]] = []
    input_ids_pattern = re.compile(r"input_ids=(\[.*\])")

    current_ids: Optional[List[int]] = None
    current_decoded: Optional[str] = None

    def commit():
        nonlocal current_ids, current_decoded
        if current_ids is not None or current_decoded is not None:
            samples.append((current_ids, current_decoded))
            current_ids, current_decoded = None, None

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fast-path: token:value CSV format (as in high_loss_sample_1.txt)
    # Detect by presence of many ":" and commas without 'Sample' or 'input_ids=['
    if (":" in content and "," in content) and ("Sample" not in content) and ("input_ids=[" not in content):
        # Each comma-separated entry is like "220:N/A" or "10494:12220.764".
        parts = [p.strip() for p in content.split(",") if p.strip()]
        ids: List[int] = []
        for part in parts:
            if ":" not in part:
                continue
            tok_str, _val = part.split(":", 1)
            tok_str = tok_str.strip()
            if not tok_str:
                continue
            try:
                tok_id = int(tok_str)
            except ValueError:
                continue
            ids.append(tok_id)
        if ids:
            return [(ids, None)]
        # Fall through to general parser if no ids parsed

    # General evaluator-log parser
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            # Blank line separates samples in typical logs
            commit()
            continue

        # New sample header lines; commit previous if any
        if line.startswith("Sample "):
            # Try to parse explicit input_ids if present on the same line
            m = input_ids_pattern.search(line)
            if m:
                try:
                    current_ids = list(ast.literal_eval(m.group(1)))
                except Exception:
                    # ignore parse errors; leave as None
                    current_ids = None
            continue

        if line.startswith("Decoded:"):
            # Everything after 'Decoded:' is the text
            current_decoded = line[len("Decoded:"):].strip()
            continue

        # Legacy format: "Sample {i}: <numpy-array-as-str>" then later a 'Loss:' line
        # We do not attempt to parse the numpy array string due to ambiguous formatting
        # and rely on 'Decoded:' when available.

    # Commit last block if file didn't end with blank line
    commit()

    # Filter out empty entries
    return [(ids, txt) for (ids, txt) in samples if ids is not None or (txt is not None and txt != "")]


def load_model_any(format_type: str, model_path: str, base_model_name: str):
    if format_type == 'transformers':
        model = load_hf_model(model_path, base_model_name)
    elif format_type == 'noLN_HF_model':
        model = load_nln_hf_model(model_name=base_model_name, name=model_path)
    elif format_type == 'fakeln_checkpoint':
        model = load_fakeln_checkpoint(model_name=base_model_name, ckpt_path=model_path)
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model checkpoint path or HF repo id")
    parser.add_argument("--format", default="transformers", choices=["transformers", "noLN_HF_model", "fakeln_checkpoint"], help="Model format used in eval")
    parser.add_argument("--model-name", default="gpt2", help="Base model name for tokenizer and helpers (e.g., gpt2, gpt2-medium, gpt2-xl)")
    parser.add_argument("--input-file", default="high_loss_sample_1.txt", help="Path to the text file with high loss samples")
    parser.add_argument("--outdir", default="bad_samples_logits", help="Output directory for per-sample logits")
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading model ({args.format}) from: {args.model}")
    model = load_model_any(args.format, args.model, args.model_name).to(device)
    model.eval()

    # Tokenizer for re-tokenization when only decoded text available
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Parse samples
    samples = parse_high_loss_file(args.input_file)
    print(f"Found {len(samples)} high-loss sample entries in '{args.input_file}'")

    saved = 0
    with torch.no_grad():
        for idx, (maybe_ids, maybe_text) in enumerate(samples):
            if maybe_ids is not None:
                input_ids = torch.tensor(maybe_ids, dtype=torch.long).unsqueeze(0).to(device)
            elif maybe_text is not None:
                enc = tokenizer(maybe_text, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
            else:
                continue

            # Run model forward to get logits
            if isinstance(model, HookedTransformer):
                logits, cache = model.run_with_cache(input_ids)
                resid_stream = cache["resid_post", model.cfg.n_layers - 1]
                print(resid_stream.shape)
                norm_resid = resid_stream.norm(dim=-1)
            else:
                outputs = model(input_ids=input_ids)
                logits = outputs.logits

            logits_cpu = logits.detach().cpu()
            input_ids_cpu = input_ids.detach().cpu()

            out_path = os.path.join(args.outdir, f"sample_{idx:05d}_logits.pt")
            torch.save({
                "input_ids": input_ids_cpu,
                "logits": logits_cpu,
                "decoded": maybe_text,
                "resid_norm": norm_resid,
            }, out_path)
            saved += 1

    print(f"Saved logits for {saved} samples to '{args.outdir}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


