# Goal: remove layernorm from a new generation model

Requirements:
1. Needs to be foldable using transformer_lens
2. Needs to be smallish, so that fine-tuning can be done relatively quick
3. Needs to be supported by transformers
4. Needs to be newer than GPT-2

Decision: use Qwen-3-0.6B from huggingface transformers

Steps:
- Load and update requirements
- Evaluate
    - [x] hellaswag etc using llm evals
    - [x] Loss on variants of the pile and OWT
    - Ensure that transformer_lens folding works with the model
- Remove LayerNorm through fine tuning
    - Update FakeLayerNorm to use the correct norm
    - Update schedule 
- Replicate mech interp experiments - DLA, Attribute patching

## Eval

```
python eval_pile.py --model-name "Qwen/Qwen3-0.6B" -m "Qwen/Qwen3-0.6B" -b 1 -d pile-10k --ctx-len 1024
```