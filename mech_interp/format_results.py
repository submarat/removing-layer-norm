import pandas as pd
import numpy as np
import os
import threading
import multiprocessing
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

class FormatInference:
    def __init__(self, output_file, save_text=False, num_threads=None):
        self.output_file = output_file
        self.save_text = save_text
        self.data = []
        self.lock = threading.Lock()  # For thread-safe appending to self.data
        
        self.num_threads = num_threads if num_threads is not None else min(32, multiprocessing.cpu_count() * 2)
        
        # Initialize tokenizer once if needed
        self.tokenizer = None
        if self.save_text:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def _process_sequence(self, args):
        """Process a single sequence and return all its subsequence records."""
        seq_idx, seq_input, ce_losses, ce_diffs, jsd_losses, topk_jsd_losses, symmetric_kl_losses, tokenizer = args
        
        seq_len = len(seq_input)
        records = []
        
        # Decode tokens if needed (only once per sequence)
        seq_decoded = None
        if self.save_text and tokenizer is not None:
            seq_decoded = [tokenizer.decode([int(token)]) for token in seq_input]
        
        # Generate sliding window subsequences
        for pos in range(seq_len - 1):
            # Get full sequence up to current position
            full_seq = seq_input[:pos+1].tolist()
            # Last token is the last element of the current subsequence
            last_token = int(seq_input[pos])
            # Next token is the token following the current subsequence
            next_token = int(seq_input[pos+1])
            
            # Create record
            record = {
                'sequence_length': pos + 1,
                'full_sequence': full_seq,
                'last_token': last_token,
                'next_token': next_token,
            }
            
            # Add text if needed
            if self.save_text and seq_decoded:
                record['full_sequence_text'] = ''.join(seq_decoded[:pos+1])
                record['last_token_text'] = seq_decoded[pos]
                record['next_token_text'] = seq_decoded[pos+1]
            
            # Add metrics directly from numpy arrays
            for model_name in ce_losses:
                record[f'ce_{model_name}'] = float(ce_losses[model_name][seq_idx][pos])
            
            for pair_name in ce_diffs:
                record[f'ce_diff_{pair_name}'] = float(ce_diffs[pair_name][seq_idx][pos])
                record[f'jsd_{pair_name}'] = float(jsd_losses[pair_name][seq_idx][pos])
                record[f'topk_jsd_{pair_name}'] = float(topk_jsd_losses[pair_name][seq_idx][pos])
                record[f'symmetric_kl_{pair_name}'] = float(symmetric_kl_losses[pair_name][seq_idx][pos])
            records.append(record)
            
        return records

    def add_batch_data(self, input_seqs, ce_losses, ce_diffs, jsd_losses, topk_jsd_losses, symmetric_kl_losses):
        """Process batch data using thread pool for parallel sequence processing."""
        batch_size = input_seqs.shape[0]
        
        # Create tasks for the thread pool - one task per sequence
        tasks = []
        for seq_idx in range(batch_size):
            # Each task is a tuple of (seq_idx, seq_input, ce_losses, ce_diffs, jsd_losses, topk_jsd_losses, symmetric_kl_losses, tokenizer)
            task = (
                seq_idx, 
                input_seqs[seq_idx], 
                ce_losses, 
                ce_diffs, 
                jsd_losses, 
                topk_jsd_losses,
                symmetric_kl_losses,
                self.tokenizer
            )
            tasks.append(task)
        
        # Use a ThreadPoolExecutor to process sequences in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks and get results
            results = list(executor.map(self._process_sequence, tasks))
            
            # Collect all records from all sequences
            with self.lock:
                for sequence_records in results:
                    self.data.extend(sequence_records)

    def save_to_parquet(self):
        """Save accumulated data to parquet."""
        if self.data:
            df = pd.DataFrame(self.data)
            df.to_parquet(self.output_file, index=False)
            print(f"Saved {len(self.data)} records to {self.output_file}")
            self.data = []
