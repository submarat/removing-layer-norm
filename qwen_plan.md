# Goal: remove layernorm from a new generation model

Requirements:
1. Needs to be foldable using transformer_lens
2. Needs to be smallish, so that fine-tuning can be done relatively quick
3. Needs to be supported by transformers
4. Needs to be newer than GPT-2

Decision: use Qwen 3 0.6

Steps:
- Load and update requirements
- Remove LayerNorm through fine tuning
    - Update FakeLayerNorm to use the correct norm
    - Update schedule 
- Evaluate
    - hellaswag etc using llm evals
    - Loss on the pile
- Replicate mech interp experiments - DLA, Attribute patching