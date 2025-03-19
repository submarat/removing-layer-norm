import itertools
import pandas as pd
import multiprocessing as mp
import os
from transformers import AutoTokenizer
from functools import partial

# Set this before importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FormatInference:
    def __init__(self, output_file, save_text=False, num_workers=None):
        self.output_file = output_file
        self.save_text = save_text
        self.data = []
        # Set number of workers, default to CPU count if not specified
        self.num_workers = num_workers if num_workers else mp.cpu_count()

    def add_batch_data(self, input_seqs, ce_losses, ce_diffs, jsd_losses, topk_jsd_losses):
        """Add batch data using multiprocessing."""
        batch_tasks = []
        batch_size = input_seqs.shape[0]

        # For each sequence in the batch
        for seq_idx in range(batch_size):
            # Extract sequence-specific input IDs
            seq_input = input_seqs[seq_idx]

            # Extract sequence-specific loss values
            seq_ce_losses = {model: ce_losses[model][seq_idx] for model in ce_losses}
            seq_ce_diffs = {pair: ce_diffs[pair][seq_idx] for pair in ce_diffs}
            seq_jsd_losses = {pair: jsd_losses[pair][seq_idx] for pair in jsd_losses}
            seq_topk_jsd_losses = {pair: topk_jsd_losses[pair][seq_idx] for pair in topk_jsd_losses}

            # Create task with sequence-specific data
            task = (seq_input, seq_ce_losses, seq_ce_diffs,
                   seq_jsd_losses, seq_topk_jsd_losses)
            batch_tasks.append(task)

        # Process sequences in parallel using the static method
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(
                partial(self._process_sequence_worker, self.save_text),
                batch_tasks
            )

        # Flatten and add all results to data list
        for sequence_records in results:
            self.data.extend(sequence_records)

    @staticmethod
    def _process_sequence_worker(save_text, args):
        """Static worker function that doesn't rely on instance state"""
        seq_input, seq_ce_losses, seq_ce_diffs, seq_jsd_losses, seq_topk_jsd_losses = args

        seq_len = len(seq_input)
        sequence_records = []

        # Pre-decode tokens if we need to save text
        decoded_tokens = None
        if save_text:
            # Create tokenizer in the worker process
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            decoded_tokens = [tokenizer.decode([token.item()]) for token in seq_input]

        # Process positions where we always have valid target tokens (seq_len - 1)
        for pos in range(seq_len - 1):
            # Get context and target tokens
            context = seq_input[:pos+1].tolist()
            next_token = seq_input[pos+1].item()
            last_token = context[-1]  # Always valid since pos+1 >= 1

            # Create base record with grouped columns
            record = {
                'sequence_length': len(context),
                'full_sequence': context,
                'last_token': last_token,
                'next_token': next_token,
            }

            if save_text:
                # Use pre-decoded tokens
                context_text = ''.join(decoded_tokens[:pos+1])
                last_token_text = decoded_tokens[pos]
                next_token_text = decoded_tokens[pos+1]
                record['full_sequence_text'] = context_text
                record['last_token_text'] = last_token_text
                record['next_token_text'] = next_token_text

            # Add CE losses for each model
            for model_name in seq_ce_losses:
                record[f'ce_{model_name}'] = seq_ce_losses[model_name][pos].item()

            # Add CE differences for each model pair
            for pair_name in seq_ce_diffs:
                record[f'ce_diff_{pair_name}'] = seq_ce_diffs[pair_name][pos].item()
                record[f'jsd_{pair_name}'] = seq_jsd_losses[pair_name][pos].item()
                record[f'topk_jsd_{pair_name}'] = seq_topk_jsd_losses[pair_name][pos].item()

            sequence_records.append(record)

        return sequence_records

    def save_to_parquet(self):
        """Convert accumulated data to DataFrame and save as parquet."""
        df = pd.DataFrame(self.data)
        df.to_parquet(self.output_file, index=False)
        print(f"Saved data to {self.output_file}")
