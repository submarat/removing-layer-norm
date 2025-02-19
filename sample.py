"""
Sample from a huggingface model or checkpoint

Example:
    python sample.py -m results/checkpoint-300 -p "to be or not to"

Usage:
    sample.py -m MODEL [-p PROMPT] [--remove-ln]

Options:
    -h --help                       Show this help message
    -m MODEL --model MODEL          Model checkpoint path or model id [REQUIRED]
    -p PROMPT --prompt PROMPT       Prompt for completion [default: "to be or not to"]
    --remove_ln                     Undo eps scale hack [default: False]
"""

import os
import torch
from std_dicts import std_dicts
from transformers import GPT2LMHeadModel, AutoTokenizer
from utils import remove_layernorm


def main():
    from docopt import docopt
    args = docopt(__doc__)
    
    # Load the model and tokenizer using your Hugging Face Hub model ID
    model_id = args['--model']
    prompt = args['--prompt']
    remove_ln = args['--remove-ln']
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(model_id)
    model = model.to(device)
    
    print(model)

    if remove_ln:
        model = remove_layernorm("gpt2-medium", model)
    
    # Load the tokenizer
    model_type = model.config.model_type
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    import pdb; pdb.set_trace()
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
        temperature=0.7,  # Adjust temperature to control randomness (higher = more random)
        attention_mask=inputs["attention_mask"],
    )
    
    # Decode the outputs
    decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(decoded_outputs)

if __name__ == '__main__':
    main()
