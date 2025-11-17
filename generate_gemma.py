"""
Download and sample from the gemma-3-1b-it model.

Example:
    python generate_gemma.py --prompt "What is machine learning?"

Usage:
    generate_gemma.py [--prompt PROMPT] [--max-length MAX_LENGTH] [--temperature TEMP] [--num-sequences NUM]

Options:
    -h --help                           Show this help message
    -p PROMPT --prompt PROMPT           Prompt for completion [default: The future of AI is]
    -m MAX_LENGTH --max-length MAX_LENGTH   Maximum length of generated text [default: 100]
    -t TEMP --temperature TEMP           Sampling temperature [default: 0.7]
    -n NUM --num-sequences NUM         Number of sequences to generate [default: 1]
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    """Get the appropriate device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    from docopt import docopt
    args = docopt(__doc__)
    
    # Parse arguments
    prompt = args['--prompt']
    max_length = int(args['--max-length'])
    temperature = float(args['--temperature'])
    num_sequences = int(args['--num-sequences'])
    
    # Model name - try the exact name user specified first
    model_name = "google/gemma-3-1b-it"
    
    # Try to load the model
    try:
        print(f"Attempting to load model: {model_name}")
        device = get_device()
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print("Downloading and loading model (this may take a while on first run)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        
        if device.type != "cuda":
            model = model.to(device)
        
        model.eval()
        print("Model loaded successfully!")
        
        # Tokenize input
        print(f"\nGenerating text for prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=num_sequences,
                do_sample=True,
                temperature=temperature,
                attention_mask=inputs.get("attention_mask"),
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode and print outputs
        print("\n" + "="*80)
        print("Generated text:")
        print("="*80)
        for i, output in enumerate(outputs):
            decoded_text = tokenizer.decode(output, skip_special_tokens=True)
            if num_sequences > 1:
                print(f"\nSequence {i+1}:")
            print(decoded_text)
        print("="*80)
        
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        print("\nTrying alternative model names...")
        # Try alternative model names if the first one fails
        alternative_names = [
            "google/gemma-1.1-1b-it",
            "google/gemma-2b-it",
            "google/gemma-1b-it"
        ]
        
        success = False
        for alt_name in alternative_names:
            if alt_name == model_name:
                continue
            try:
                print(f"Trying: {alt_name}")
                model_name = alt_name
                device = get_device()
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    device_map="auto" if device.type == "cuda" else None,
                )
                
                if device.type != "cuda":
                    model = model.to(device)
                
                model.eval()
                
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        num_return_sequences=num_sequences,
                        do_sample=True,
                        temperature=temperature,
                        attention_mask=inputs.get("attention_mask"),
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                print("\n" + "="*80)
                print("Generated text:")
                print("="*80)
                for i, output in enumerate(outputs):
                    decoded_text = tokenizer.decode(output, skip_special_tokens=True)
                    if num_sequences > 1:
                        print(f"\nSequence {i+1}:")
                    print(decoded_text)
                print("="*80)
                success = True
                break
            except Exception as e2:
                print(f"Failed to load {alt_name}: {e2}")
                continue
        
        if not success:
            raise RuntimeError(f"Could not load any Gemma model. Tried: {model_name} and alternatives.")


if __name__ == '__main__':
    main()

