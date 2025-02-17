"""
Generate a subsampled version of the Pile dataset.
Creates strided subsequences from the original text, suitable for quick inference tasks.
"""

import argparse
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer
from tqdm import tqdm
from pathlib import Path


def main(input_path, min_length=128, max_length=512, stride=8):
    """Generate subsampled sequences from Pile dataset."""
    output_path = f'pile_sub_l{min_length}-{max_length}_s{stride}.parquet'

    # Print parameters
    print("\nParameters:")
    print(f"Min sequence length: {min_length}")
    print(f"Max sequence length: {max_length}")
    print(f"Stride length: {stride}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}\n")

    # Load data
    print("Loading data...")
    df_pile = pd.read_parquet(input_path)[['text']]

    # Initialize tokenizer
    print("Tokenizing texts...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_indices = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in tqdm(df_pile.text.tolist())
    ]
    # Add BOS and EOS tokens
    token_indices = [[50256] + seq + [50256] for seq in token_indices]
    num_tokens = [len(i) for i in token_indices]
    df_pile['token_indices'] = token_indices
    df_pile['num_tokens'] = num_tokens
    print("Tokenization complete...")

    # Remove entries outside of compatible sequence length
    df_pile = df_pile[(df_pile.num_tokens >=min_length) & (df_pile.num_tokens <= max_length)]

    # Generate subsequences
    print("Generating subsequences...")
    subsequences = []
    for idx, row in tqdm(df_pile.iterrows()):

        tokens = row['token_indices']
        seq_length = len(tokens)
        
        # Generate start positions using stride
        start_positions = range(0, seq_length - 1, stride)  # -1 to ensure room for target token
        
        for start_pos in start_positions:
            # Get entire sequence from start position to end (excluding last token)
            input_seq = tokens[start_pos:-1]
            # Get the next token after the sequence as target
            target = tokens[start_pos + 1]
            
            subsequences.append({
                'original_sequence_id': idx,
                'input_sequence': input_seq,
                'target_token': target,
                'sequence_length': len(input_seq),
                'start_position': start_pos
            })

    # Create and save output DataFrame
    print("Saving processed data...")
    out_df = pd.DataFrame(subsequences)
    out_df.to_parquet(output_path, index=False)

    # Print statistics
    print(f"Generated {len(out_df)} subsequences from {len(df_pile)} original sequences")
    print(f"Number of unique original sequences: {len(np.unique(out_df.original_sequence_id))}")
    print(f"Saved to: {output_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Subsample Pile dataset into strided subsequences.')
    
    parser.add_argument('--input_path', type=str, default='raw_pile10k.parquet',
                        help='Path to the input Parquet file containing the Pile dataset')
    parser.add_argument('--min_length', type=int, default=128,
                        help='Minimum sequence length (default: 128)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--stride', type=int, default=8,
                        help='Stride length for generating subsequences (default: 8)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Call main function with parsed arguments
    main(
        input_path=args.input_path,
        min_length=args.min_length,
        max_length=args.max_length,
        stride=args.stride
    )
