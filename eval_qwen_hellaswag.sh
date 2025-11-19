#!/bin/bash
# Evaluate Qwen 3 0.6B on HellaSwag benchmark

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-0.6B \
    --tasks hellaswag \
    --device cuda \
    --batch_size 8 \
    --output_path ./results

