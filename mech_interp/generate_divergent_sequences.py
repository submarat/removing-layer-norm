import pandas as pd
import torch as t
from transformers import GPT2TokenizerFast
from models.load_models import load_baseline, load_nln_model
from dataclasses import dataclass
from typing import Any
from tqdm import tqdm

class DivergenceAnalyzer:
    def __init__(self,
                examples_path: str,
                raw_df_path: str,
                subsampled_df_path: str, 
                device: str = None):
        # Set device
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialise the tokenizer           
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            
        # Load dataframes
        self.interesting_df = pd.read_parquet(examples_path)
        self.raw_df = pd.read_parquet(raw_df_path)
        self.subsampled_df = pd.read_parquet(subsampled_df_path)
        
        # Load models
        self.baseline = load_baseline().to(self.device)
        self.noln = load_nln_model().to(self.device)
        
        print(f"Models loaded on {self.device}")

    @dataclass
    class GeneratedExample:
        text: str  # full original text
        input_sequence: list[int]  # token indices
        target_token: int  # the token where models diverge
        baseline_continuation: str
        noln_continuation: str
        js_divergence: float
        
        def __str__(self):
            # Get the subsampled text from input_sequence
            subsampled_text = self.tokenizer.decode(self.input_sequence)
            
            # Find where this appears in the full text
            start_idx = self.text.find(subsampled_text)
            end_idx = start_idx + len(subsampled_text)
            
            # Split the text
            before = self.text[:end_idx]
            divergent_token = self.tokenizer.decode([self.target_token])
            after = self.text[end_idx:]
            
            divider = "="*100
            return (
                f"\n{divider}\n"
                f"JS Divergence: {self.js_divergence:.2f}\n"
                f"\n"
                f"Original Text (divergence marked with **):\n"
                f"{before}**{divergent_token}**{after}\n"
                f"\n"
                f"Continuations from divergence point:\n"
                f"\n"
                f"[Baseline]\n"
                f"{self.baseline_continuation}\n"
                f"\n"
                f"[NoLN]\n"
                f"{self.noln_continuation}\n"
                f"{divider}\n"
            )

    def generate_continuations(self, max_new_tokens: int = 256) -> list[GeneratedExample]:
        """Generate continuations for examples with high divergence"""
        examples = []
        
        for row in tqdm(self.interesting_df.itertuples(index=True),
                         total=len(self.interesting_df),
                         desc="Generating continuations"):
            # Get input sequence and target token
            subsampled_row = self.subsampled_df.iloc[row.Index]
            input_sequence = subsampled_row.input_sequence
            target_token = subsampled_row.target_token

            # Get original text representation of the original sequence and subsequence
            full_text = '<|endoftext|>' + self.raw_df.iloc[row.original_sequence_id].text
            sub_text = self.tokenizer.decode(subsampled_row.input_sequence)
            target_token = self.tokenizer.decode(target_token)
            divergence_start = len(sub_text)
            end_text = full_text[divergence_start:]

            input_tensor = t.tensor(input_sequence)[None, :].to(self.device)
            # Generate from each model using HookedTransformer
            baseline_output = self.baseline.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False
            )
            noln_output = self.noln.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False
            )

            # Get just the new tokens
            baseline_generated = self.tokenizer.decode(baseline_output[0, len(input_sequence):])
            noln_generated = self.tokenizer.decode(noln_output[0, len(input_sequence):])

            out_dict = {'input_sequence' : sub_text,
                        'ground_truth' : end_text,
                        'baseline': baseline_generated,
                        'noln': noln_generated}
            
            examples.append(out_dict)

        self.generated_sequences = pd.DataFrame(examples)
        
    def run_analysis(self, output_path: str, max_new_tokens: int = 256):
        """Run full analysis pipeline"""
        self.generate_continuations(max_new_tokens)
        self.generated_sequences.to_csv(output_path, index=False)
        print(f"\nSaved {len(self.generated_sequences)} examples to {output_path}")


if __name__ == "__main__":
   # Paths
   examples_path = 'analysis/gpt2-small/interesting_divergences.parquet'
   raw_df_path = 'data/raw_pile10k.parquet'
   subsampled_df_path = 'data/pile_sub_l256-512_s16.parquet'
   output_path = 'experiments/model_generations.csv'
   
   # Run analysis
   analyzer = DivergenceAnalyzer(examples_path, raw_df_path, subsampled_df_path)
   analyzer.run_analysis(output_path)
