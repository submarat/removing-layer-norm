import os
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Load the model and tokenizer using your Hugging Face Hub model ID
model_id = "model-without-ln"

# Load the model
model = GPT2LMHeadModel.from_pretrained(model_id)

print(model)

# Undo scale hack
if os.getenv('SCALE', '') != '':
    with torch.no_grad():
        for head in model.transformer.h:
            head.ln_1.weight /= 1e6
            head.ln_1.eps = 1e-5
            head.ln_2.weight /= 1e6
            head.ln_2.eps = 1e-5
        model.transformer.ln_f.weight /= 1e6
        model.transformer.ln_f.eps = 1e-5

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.eos_token = tokenizer.eos_token

# Example usage
text = "Penguin is a mamal, crocodile is a"
inputs = tokenizer(text, return_tensors="pt")

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
