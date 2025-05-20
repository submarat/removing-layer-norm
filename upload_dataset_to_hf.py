#%%
import datasets
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import pandas as pd
from huggingface_hub import upload_file
# %%
tokens_to_filter = pd.read_csv("pile_unique_tokens.csv")
# %%
tokens_to_filter['token_repr'] = tokens_to_filter['token'].apply(lambda x: repr(x)[1:-1])

# %%
tokens_to_filter.sort_values(by='count', ascending=False, inplace=True)
# %%
tokens_to_filter.head()
# %%
apollo_pile = datasets.load_dataset("apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2", split="train", streaming=True)

#%%
# Convert token_ids to a set for faster lookup
token_ids_to_filter = set(tokens_to_filter['token_id'].tolist())

# Function to check if sequence contains filtered tokens
def contains_filtered_tokens(example):
    return not any(token_id in token_ids_to_filter for token_id in example['input_ids'])

# Filter and take first 10k examples
filtered_apollo_pile = apollo_pile.filter(contains_filtered_tokens).take(10000)
#%%
# Convert to Dataset format
filtered_apollo_pile_list = list(filtered_apollo_pile)
dataset = datasets.Dataset.from_list(filtered_apollo_pile_list)

#%%
# Push to hub
dataset.push_to_hub("lucabaroni/apollo-pile-filtered-10k", private=True)

#%%
# Save and push the dataframe
tokens_to_filter.to_csv("pile_unique_tokens.csv", index=False)
upload_file(
    path_or_fileobj="pile_unique_tokens.csv",
    path_in_repo="pile_unique_tokens.csv",
    repo_id="lucabaroni/apollo-pile-filtered-10k",
    repo_type="dataset"
)

# Push the script
upload_file(
    path_or_fileobj="upload_dataset_to_hf.py",
    path_in_repo="upload_dataset_to_hf.py",
    repo_id="lucabaroni/apollo-pile-filtered-10k",
    repo_type="dataset"
)

# push the compare_owt_and_pile.py script
upload_file(
    path_or_fileobj="compare_owt_and_pile.py",
    path_in_repo="compare_owt_and_pile.py",
    repo_id="lucabaroni/apollo-pile-filtered-10k",
    repo_type="dataset"
)

#%%
# Create README content
readme_content = """# Filtered Apollo-Pile Dataset (10k examples)

This dataset contains 10,000 examples from the Apollo-Research version of the Pile dataset (apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2) that have been filtered to remove sequences containing tokens that are present in the Pile but not in OpenWebText.
It can be useful as a test set for models that are trained on OpenWebText.

## Dataset Creation

The dataset was created by:
1. Analyzing token distributions in both the Apollo-Pile and OpenWebText datasets
2. Identifying tokens that appear in the Apollo-Pile but not in OpenWebText
3. Filtering out sequences from the Apollo-Pile that contain any of these unique tokens
4. Taking the first 10,000 examples from the filtered dataset

## Contents

- `pile_unique_tokens.csv`: CSV file containing the tokens that were filtered out, including their IDs and frequencies
- `upload_dataset_to_hf.py`: Script used to create and upload this dataset
- `compare_owt_and_pile.py`: Script used to analyze token distributions and identify unique tokens
"""

# Upload README to hub
upload_file(
    path_or_fileobj=readme_content.encode(),
    path_in_repo="README.md",
    repo_id="lucabaroni/apollo-pile-filtered-10k",
    repo_type="dataset"
)

# %%
