from datasets import load_dataset
import torch
from torch.utils.data import DataLoader as TorchDataLoader


class DataManager:
    def __init__(self,
                 dataset_name='apollo-pile',
                 batch_size=10,
                 max_context=1024,
                 num_samples=5000):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.max_context = max_context


    def get_dataset(self):
        # We currently only support the pre-tokenized, pre-batched Apollo datasets, can extend to other datasets later.
        if self.dataset_name == 'apollo-pile':
            dataset = load_dataset('apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2', 
                                 streaming=True, split="train")
        if self.dataset_name == 'luca-pile':
            dataset = load_dataset('lucabaroni/apollo-pile-filtered-10k', 
                                 streaming=True, split="train")
        elif self.dataset_name == 'apollo-owt':
            dataset = load_dataset('apollo-research/Skylion007-openwebtext-tokenizer-gpt2', 
                                 streaming=True, split="train")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return dataset.shuffle(seed=42).take(self.num_samples)


    def collate_fn(self, examples):
        # This will truncate any sequence > max_context. Note : all apollo-pile datasets already consist of 1024 sequences
        # Also returning a single tensor here, we can change this if we decide to output more just input sequences
        return torch.tensor([ex['input_ids'][:self.max_context] for ex in examples])


    def create_dataloader(self):
        dataset = self.get_dataset()
        dataloader = TorchDataLoader(
                        dataset, 
                        batch_size=self.batch_size, 
                        shuffle=False,
                        collate_fn=self.collate_fn)  
        return dataloader
