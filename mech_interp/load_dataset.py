from datasets import load_dataset, Dataset, IterableDataset
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Union, Optional, Any, Callable

class DataLoader:
    # Dataset mapping dictionary for easier maintenance and extension
    DATASET_MAPPINGS: Dict[str, str] = {
        'apollo-pile': 'apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2',
        'apollo-owt': 'apollo-research/Skylion007-openwebtext-tokenizer-gpt2'
    }
    
    def __init__(
        self, 
        dataset_name: str = 'apollo-pile',
        batch_size: int = 10,
        max_context: int = 1024,
        num_samples: int = 5000,
        prepend_bos: bool = False
    ) -> None:
        """
        Initialize the DataLoader with configuration parameters.
        
        Args:
            dataset_name: Name of the dataset to load (must be in DATASET_MAPPINGS)
            batch_size: Number of examples per batch
            max_context: Maximum sequence length to include
            num_samples: Number of examples to take from the dataset
            prepend_bos: Whether to prepend the BOS token to each sequence
        """
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.max_context = max_context
        self.prepend_bos = prepend_bos
        
        # If BOS should be prepended, load the GPT2 tokenizer to get the BOS token ID
        self.bos_token_id = None
        if self.prepend_bos:
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.bos_token_id = tokenizer.eos_token_id  # Use EOS token (50256) as BOS
        
    def get_dataset(self) -> IterableDataset:
        """
        Load and prepare the dataset based on the dataset name.
        
        Returns:
            An iterable dataset with the selected examples
            
        Raises:
            ValueError: If the dataset name is not recognized
        """
        if self.dataset_name not in self.DATASET_MAPPINGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        dataset_path = self.DATASET_MAPPINGS[self.dataset_name]
        dataset = load_dataset(dataset_path, streaming=True, split="train")
        return dataset.shuffle(seed=42).take(self.num_samples)
        
    def collate_fn(self, examples: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Process examples by optionally prepending BOS token and truncating to max_context.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            A tensor containing the processed input_ids from all examples
        """
        sequences = []
        for ex in examples:
            if self.prepend_bos and self.bos_token_id is not None:
                # Prepend BOS token and then truncate to max_context
                sequence = [self.bos_token_id] + ex['input_ids']
                sequences.append(sequence[:self.max_context])
            else:
                # Just truncate to max_context
                sequences.append(ex['input_ids'][:self.max_context])
                
        return torch.tensor(sequences)
        
    def create_dataloader(self) -> TorchDataLoader:
        """
        Create and return a PyTorch DataLoader with the configured dataset.
        
        Returns:
            A configured PyTorch DataLoader
        """
        dataset = self.get_dataset()
        return TorchDataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=self.collate_fn
        )
