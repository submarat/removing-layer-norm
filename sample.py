"""
Sample from a huggingface model or checkpoint

Example:
    python sample.py -m results/checkpoint-300 -p "to be or not to"

Usage:
    sample.py -m MODEL [-c CHECKPOINT] [-p PROMPT] [-n]

Options:
    -h --help                               Show this help message
    -m MODEL --model MODEL                  Pretrained model name [REQUIRED]
    -c CHECKPOINT --checkpoint CHECKPOINT   Checkpoint to load [REQUIRED]
    -p PROMPT --prompt PROMPT               Prompt for completion [default: As the last leaf fell from the tree ]
    -r --remove_ln_by_scaling               Remove ln by scaling weights and eps [default: False]
"""

import torch
from std_dicts import std_dicts
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModel, GPT2Config
from transformers.modeling_utils import load_sharded_checkpoint
from train import load_model, replace_layernorm_with_fake_layernorm

from utils import remove_layernorm_by_scaling, extract_std_from_checkpoint, get_device


def main():
    from docopt import docopt
    args = docopt(__doc__)
    
    # Load the model and tokenizer using your Hugging Face Hub model ID
    model_name = args['--model']
    ckpt = args['--checkpoint']
    prompt = args['--prompt']
    remove_ln = args['--remove_ln_by_scaling']

    device = get_device()
    std_dict = extract_std_from_checkpoint(model_name, ckpt)
    
    model = load_model(ckpt)
    if remove_ln:
        model = remove_layernorm_by_scaling(model, std_dict)

    model.to(device)

    # Load the tokenizer
    model_type = model.config.model_type
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.eos_token = tokenizer.eos_token
    
    # Example usage
    text = prompt
    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)
    
    # Generate text using sampling
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,  # Adjust max length as needed
        num_return_sequences=1,
        do_sample=True,
        temperature=1.0,  # Adjust temperature to control randomness (higher = more random)
        attention_mask=inputs["attention_mask"],
    )
    
    # Decode the outputs
    decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(decoded_outputs)

if __name__ == '__main__':
    main()
