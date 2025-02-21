# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt


# %%
raw_df = pd.read_parquet('../../data/raw_pile10k.parquet')
sub_df = pd.read_parquet('../../data/pile_sub_l256-512_s16.parquet')

# %%
# Initialize tokenizer
print("Tokenizing texts...")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
token_indices = [
    tokenizer.encode(text, add_special_tokens=False)
    for text in tqdm(raw_df.text.tolist())
]
# Add BOS and EOS tokens
token_indices = [[50256] + seq for seq in token_indices]
num_tokens = [len(i) for i in token_indices]

# %%
plt.style.use('seaborn-v0_8-colorblind')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(num_tokens, bins=10000,
        align='left', rwidth=0.8)

ax.set_title('Histogram of Pile-10k sequence lengths', fontsize=12, pad=10)
ax.set_xlabel('Sequence length (in tokens)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.tick_params(labelsize=8)
ax.axvline(x=1024, color='black', linestyle='--', label='GPT-2-small maximum context length')
ax.axvline(x=256, color='red', linestyle='--', label='Subsequences to consider')
ax.axvline(x=512, color='red', linestyle='--')


plt.xlim(0, 20000)
plt.yscale('log')
plt.tight_layout()
plt.legend(loc='upper right', fontsize=10)
plt.savefig('pile10k_token_count.png', dpi=200)
plt.show()

# %%
plt.style.use('seaborn-v0_8-colorblind')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(sub_df.sequence_length, bins=32,
        align='left', rwidth=0.8)

ax.set_title(f'Histogram of subsampled Pile-10k sequence lengths ({len(sub_df)} sequences))', fontsize=12, pad=10)
ax.set_xlabel('Sequence length (in tokens)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.tick_params(labelsize=8)

plt.xlim(0, 512)
plt.yscale('log')
plt.tight_layout()
plt.legend(loc='upper right', fontsize=10)
plt.savefig('subsampled_pile10k_token_count.png', dpi=200)
plt.show()
# %%
