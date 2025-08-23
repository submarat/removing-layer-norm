from datasets import load_dataset
import numpy as np
from collections import Counter
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from transformers import GPT2Tokenizer

pile_dataset = load_dataset('ANONYMIZED', 
                      split='train', 
                      num_proc=16,
                      )


openwebtext_dataset = load_dataset('ANONYMIZED', 
                      split='train', 
                      num_proc=16,
                      )

pile_dataset = pile_dataset.select(range(1_000_000))
openwebtext_dataset = openwebtext_dataset.select(range(1_000_000))

# Move the counting function outside
def count_batch(batch):
    vocab_size = 50257
    counts = np.zeros(vocab_size, dtype=np.int64)
    for seq in batch['input_ids']:
        np.add.at(counts, seq, 1)
    return counts

def extract_vocab_and_counts(dataset, batch_size=1000):
    vocab_size = 50257
    final_counts = np.zeros(vocab_size, dtype=np.int64)
    
    # Process dataset in chunks to avoid memory issues
    chunk_size = 1_000_000  # Process 1M examples at a time
    for start_idx in range(0, len(dataset), chunk_size):
        end_idx = min(start_idx + chunk_size, len(dataset))
        chunk = dataset.select(range(start_idx, end_idx))
        
        # Create batches from this chunk
        batches = chunk.iter(batch_size=batch_size)
        
        # Process batches in parallel
        with Pool(mp.cpu_count()) as pool:
            # Process and accumulate counts immediately instead of storing all results
            for batch_count in tqdm(
                pool.imap(count_batch, batches),
                total=len(chunk) // batch_size,
                desc=f"Processing chunk {start_idx//chunk_size + 1}"
            ):
                final_counts += batch_count
    
    # Create Counter and vocab set
    nonzero_indices = np.nonzero(final_counts)[0]
    token_counter = Counter({idx: int(count) for idx, count in zip(nonzero_indices, final_counts[nonzero_indices])})
    vocab_set = set(nonzero_indices)
    
    return vocab_set, token_counter

# Get vocab and counts for both datasets
pile_vocab, pile_counter = extract_vocab_and_counts(pile_dataset, batch_size=1000)
owt_vocab, owt_counter = extract_vocab_and_counts(openwebtext_dataset, batch_size=1000)

# Find tokens unique to each dataset
pile_only = pile_vocab - owt_vocab
owt_only = owt_vocab - pile_vocab

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Save unique Pile tokens
with open('pile_unique_tokens.csv', 'w', encoding='utf-8') as f:
    f.write('token_id,token,count\n')
    for token_id in pile_only:
        token = tokenizer.decode([token_id])
        count = pile_counter[token_id]
        f.write(f'{token_id},"{token}",{count}\n')

# Save unique OpenWebText tokens  
with open('owt_unique_tokens.csv', 'w', encoding='utf-8') as f:
    f.write('token_id,token,count\n')
    for token_id in owt_only:
        token = tokenizer.decode([token_id])
        count = owt_counter[token_id]
        f.write(f'{token_id},"{token}",{count}\n')

print(f"Found {len(pile_only)} tokens unique to Pile dataset")
print(f"Found {len(owt_only)} tokens unique to OpenWebText dataset")

