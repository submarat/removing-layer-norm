"""
Evaluate language model performance on The Pile dataset variants or OpenWebText.

Example:
    python eval_pile.py -m ckpt1200.pt -f nanogpt -d pile-10k -n 20000 -b 16
    python eval_pile.py -m gpt2 -f transformers -d pile-ANONYMIZED -b 8
    python eval_pile.py -m results/checkpoint-1200 -f transformers
    python eval_pile.py -m gpt2 -f transformers -d openwebtext -b 8

Usage:
    eval_pile.py -m MODEL [-f FORMAT] [-d DATASET] [-n NUM_SAMPLES] [-b BATCH_SIZE] [--model-name MODEL_NAME]
    eval_pile.py -h | --help

Options:
    -h --help                       Show this help message
    -m MODEL --model MODEL          Model checkpoint path or model id [REQUIRED]
    -f FORMAT --format FORMAT       Model format: transformers/noLN_HF_model/fakeln_checkpoint [default: transformers]
    -d DATASET --dataset DATASET    Dataset variant: pile-10k/pile-ANONYMIZED/pile-ANONYMIZED-ANONYMIZED/pile-uncopyrighted/openwebtext [default: pile-ANONYMIZED]
    -n NUM --num-samples NUM        Number of samples to evaluate [default: 10000]
    -b BATCH_SIZE --batch-size BATCH_SIZE  Batch size for evaluation [default: 8]
    --model-name MODEL_NAME         Base model name [default: gpt2]
"""

import os
import sys
import torch
import train
from transformer_lens import HookedTransformer
from utils import get_device
from pile_eval import preprocess_pile_dataset, evaluate_model_on_pile
from transformers import GPT2LMHeadModel, AutoModelForCausalLM, logging
from transformers.modeling_utils import load_sharded_checkpoint
from prepare_dataset import prepare_dataset

# Load model with appropriate device
device = get_device()

def load_saved_model(model_name: str, model_path=None):
    if model_path is not None: 
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        # Load OpenAI's GPT model - use specific model name like "gpt2-large" or "gpt2-xl"
        model = AutoModelForCausalLM.from_pretrained(model_name)  # or other OpenAI model version
    return model


def load_hf_model(model_id_or_ckpt_path, model_name):
    """ Loads huggingface transformers model and removes layernorm """

    model_hf = GPT2LMHeadModel.from_pretrained(model_id_or_ckpt_path)

    return model_hf

def load_fakeln_checkpoint(ckpt_path, model_name):

    # Load model and replace with FakeLayerNorm
    ckpt_model = train.load_model(model_name=model_name, remove_ln=True)
    # Load checkpoint to load in std values
    try:
        # First try loading a single pytorch model file
        missing, unexpected = ckpt_model.load_state_dict(torch.load(os.path.join(ckpt_path, 'pytorch_model.bin'), map_location=get_device()), strict=False)
    except FileNotFoundError:
        try:
            # If that fails, try loading a sharded checkpoint
            missing, unexpected = load_sharded_checkpoint(ckpt_model, ckpt_path, strict=False)
        except Exception as e:
            raise ValueError(f"Could not load checkpoint from {ckpt_path}. Error: {str(e)}")

    if missing:
        print(f"Missing keys when loading checkpoint: {len(missing)} keys")
    if unexpected:
        print(f"Unexpected keys when loading checkpoint: {len(unexpected)} keys")

    return ckpt_model


def load_nln_hf_model(model_name, name=None, model=None):
    if model is None and name is None or model is not None and name is not None:
        raise ValueError("Either name or model must be provided, but not both")
    if model is not None:
        model = model.to("cpu")
    else:
        model = GPT2LMHeadModel.from_pretrained(name).to("cpu")
    
    for block in model.transformer.h:
        block.ln_1.weight.data = block.ln_1.weight.data / 1e6
        block.ln_1.eps = 1e-5
        block.ln_2.weight.data = block.ln_2.weight.data / 1e6
        block.ln_2.eps = 1e-5
    model.transformer.ln_f.weight.data = model.transformer.ln_f.weight.data / 1e6
    model.transformer.ln_f.eps = 1e-5
    
    # Properly replace LayerNorms by Identities
    class HookedTransformerNoLN(HookedTransformer):
        def removeLN(self):
            for i in range(len(self.blocks)):
                self.blocks[i].ln1 = torch.nn.Identity()
                self.blocks[i].ln2 = torch.nn.Identity()
            self.ln_final = torch.nn.Identity()
    
    hooked_model = HookedTransformerNoLN.from_pretrained(model_name, hf_model=model, fold_ln=True, center_unembed=False).to("cpu")

    hooked_model.removeLN()
    hooked_model.cfg.normalization_type = None
    print(f"loaded {name} and removed LN hack, returning transformerlens hooked model")
    return hooked_model


def main():
    from docopt import docopt
    args = docopt(__doc__)
    
    # Parse arguments
    model_path = args['--model']
    format_type = args['--format']
    dataset_name = args['--dataset']
    num_samples = int(args['--num-samples'])
    batch_size = int(args['--batch-size'])
    model_name = args['--model-name'] or "gpt2"
    
    # Create cache directory if it doesn't exist
    os.makedirs("processed_datasets", exist_ok=True)
    
    # Load model based on format
    if format_type == 'transformers':
        # Load model from huggingface or local, standard format
        model = load_hf_model(model_path, model_name)
    elif format_type == 'noLN_HF_model':
        # Load model that has scale trick applied to disable layer norm
        model = load_nln_hf_model(model_name=model_name, name=model_path)
    elif format_type == 'fakeln_checkpoint':
        # Load local model checkpoint assuming it has FakeLayerNorms and special state_dict
        model = load_fakeln_checkpoint(model_name=model_name, ckpt_path=model_path)
    else:
        raise ValueError(f"Unknown format type: {format_type}")

    model = model.to(device)


    if dataset_name == 'openwebtext':
        # Load OpenWebText dataset
        print("Loading OpenWebText dataset...")
        tokenized, _ = prepare_dataset(model_name)
        # Convert to list of tensors for evaluation
        processed_examples = [torch.tensor(example["input_ids"]) for example in tokenized["test"]]
    else:
        # Using shared preprocessing function for Pile datasets
        processed_examples, _ = preprocess_pile_dataset(dataset_name, model_name, num_samples)
    
    # Using shared evaluation function
    ce_loss = evaluate_model_on_pile(model, processed_examples, batch_size)
    output_string = f"Final Cross-Entropy Loss on {dataset_name}: {ce_loss:.4f}\n"
    command_used = " ".join(sys.argv) + "\n--------------\n"
    
    with open("eval.txt", "a") as f:
        f.write(output_string)
        f.write(command_used)

    print(f"Results appended to eval.txt")

if __name__ == "__main__":
    main()