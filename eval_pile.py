import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from transformer_lens import HookedTransformer


def load_saved_model(model_path=None):
    if model_path is not None: 
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        # Load OpenAI's GPT model - use specific model name like "gpt2-large" or "gpt2-xl"
        model = AutoModelForCausalLM.from_pretrained("gpt2")  # or other OpenAI model version    
    return model

def load_nln_hf_model(name=None):
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
    
    hooked_model = HookedTransformerNoLN.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=False).to("cpu")

    hooked_model.removeLN()
    hooked_model.cfg.normalization_type = None
    print(f"loaded {name} and removed LN hack, returning transformerlens hooked model")
    return hooked_model

def custom_tokenizer(examples):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Add EOS token to the end of each example
    examples["text"] = [text + tokenizer.eos_token for text in examples["text"]]

    # Tokenize the examples
    tokenized_examples = tokenizer(
        examples["text"], truncation=False, padding=False, return_tensors=None
    )

    # Concatenate the tokenized examples
    concatenated = []
    for seq in tokenized_examples["input_ids"]:
        concatenated.extend(seq)

    # Chunk into 1024 token chunks, dropping any remainder
    n_chunks = (
        len(concatenated) // 1024
    )  # Integer division to get complete chunks only
    chunks = [concatenated[i * 1024 : (i + 1) * 1024] for i in range(n_chunks)]

    return {"input_ids": chunks}


def evaluate_on_pile_ce(model, dataset_name, num_samples=5000, device=None):
    print(f"Evaluating on {dataset_name}, using {num_samples} samles")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_name == "pile-apollo":
        dataset = load_dataset("apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2", streaming=True, split="train")
        dataset = dataset.shuffle(seed=42)
        dataset = list(dataset.take(num_samples))
    elif dataset_name == "pile-uncopyrighted":
        dataset = load_dataset("monology/pile-uncopyrighted", streaming=True, split="train")
        dataset = dataset.shuffle(seed=42)
        dataset = list(dataset.take(num_samples))
    elif dataset_name == "pile-10k":
        dataset = load_dataset("NeelNanda/pile-10k", streaming=False, split="train")
    
    if dataset_name == "pile-apollo":
        processed_examples = []
        for i, example in enumerate(tqdm(dataset)):
            processed_examples.append(torch.tensor(example["input_ids"]))
    else:
        # Process dataset in fixed-size chunks
        # processed_examples = []
        # for i in range(0, len(dataset), 1024):
        #     batch = dataset[i:i + 1024]
        #     processed = custom_tokenizer({"text": [ex["text"] for ex in batch]})
        #     processed_examples.extend(processed["input_ids"])
        # Process in chunks without limiting total size
        batch_size = 1024
        processed_examples = []
        
        # Use iterator to process dataset in chunks
        dataset_iterator = iter(dataset)
        while True:
            try:
                batch = []
                # for _ in range(batch_size):
                #     batch.append(next(dataset_iterator)["text"])
                while len(batch) < batch_size:
                    example = next(dataset_iterator)
                    # Filter out OpenWebText2
                    if 1: # example.get('meta', {}).get('pile_set_name') != "OpenWebText2":
                        batch.append(example["text"])
                processed = custom_tokenizer({"text": batch})
                print(len(processed["input_ids"][0]))
                processed_examples.extend(processed["input_ids"])
            except StopIteration:
                break
    
    print(f"Processed data shape: {(len(processed_examples), len(processed_examples[0]))}")
    
    # For testing purposes, print the first 100 examples
    # for i, example in enumerate(dataset):
    #     if i >= 100:
    #         break
    #     print(tokenizer.decode(example["input_ids"]))
    
    model.to(device)
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, chunk in enumerate(tqdm(processed_examples, desc="Evaluating")):
            input_ids = torch.tensor(chunk).unsqueeze(0).to(device)
            
            if isinstance(model, HookedTransformer):
                logits = model(input_ids)
                loss = model.loss_fn(logits, input_ids, per_token=True)
                
                total_loss += loss.sum().item()
                total_tokens += input_ids.numel()
            else:
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item() * len(chunk)
                total_tokens += len(chunk)
            if i % 100 == 0:
                print(
                    f"Processed {i} examples. Current ce loss: {total_loss/total_tokens:.2f}"
                )
            
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def main():
    # model_path = "./model_checkpoints"  # Path to your saved model
    # model = load_saved_model(model_path)
    model = load_saved_model()
    # model = load_nln_hf_model("apollo-research/gpt2_noLN")
    dataset_name = "pile-10k"
    num_samples = 20000
    ce_loss = evaluate_on_pile_ce(model, dataset_name, num_samples)
    print(f"Final Cross-Entropy Loss on PILE: {ce_loss:.4f}")

if __name__ == "__main__":
    main()
