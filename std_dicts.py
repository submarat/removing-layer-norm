import json

# Load standard deviations for each model
std_dicts = {}
for model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:  # Add other model names as needed
    try:
        with open(f'{model_name}_std_dicts.json', 'r') as f:
            std_dicts[model_name] = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find std_dicts file for {model_name}")


__all__ = ['std_dicts']