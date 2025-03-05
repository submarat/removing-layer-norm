from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Literal

from models.load_models import load_baseline, load_finetuned_model, load_nln_model

import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

@dataclass
class ProcessingConfig:
    """Configuration for model processing pipeline."""
    input_file: Path
    output_dir: Path
    batch_size: int = 32
    model_type: Literal["baseline", "finetuned", "nln"] = "baseline"
    use_memmap: bool = True  # Use memory-mapped array
    chunk_size: int = 1000  # Process in chunks to save memory
    precision: str = "float32"  # Reduced precision to save memory
    device: str = "cuda" if t.cuda.is_available() else "cpu"
    shuffle: bool = False
    
    @property
    def output_file(self) -> Path:
        """Full path to output file including directory and model type name."""
        return self.output_dir / f"softmax_probabilities_{self.model_type}.npy"
    

class LengthSortedDataset(Dataset):
    """Dataset that groups sequences of similar lengths together."""
    
    def __init__(self, df: pd.DataFrame, batch_size: int):
        self.df = df
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self) -> List[List[int]]:
        """Pre-compute batches of same-length sequences."""
        batches = []
        for _, group in self.df.groupby('sequence_length'):
            indices = group.index.tolist()
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        return batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict:
        batch_indices = self.batches[idx]
        batch_data = self.df.iloc[batch_indices]
        
        return {
            'input_sequences': batch_data['input_sequence'].tolist(),
            'indices': batch_indices
        }


class ModelProcessor:
    """Handles model processing pipeline."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model = self._load_model()
        self.device = t.device(config.device)
        self.model.to(self.device)
        # Enable torch.cuda.amp for mixed precision if on GPU
        self.use_amp = self.config.device == "cuda" and self.config.precision == "float16"

        # Clean up GPU memory if possible
        if self.config.device == "cuda":
            t.cuda.empty_cache()
        
    def _load_model(self):
        """Load the appropriate model based on configuration."""
        model_loaders = {
            "baseline": load_baseline,
            "finetuned": load_finetuned_model,
            "nln": load_nln_model
        }
        
        if self.config.model_type not in model_loaders:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
            
        return model_loaders[self.config.model_type]()
    
    def _create_dataloader(self, df: pd.DataFrame) -> DataLoader:
        """Create dataloader from input dataframe."""
        dataset = LengthSortedDataset(df, self.config.batch_size)
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=self.config.shuffle
        )
    
    def process_batch(self, batch: dict) -> tuple:
        """Process a single batch of data."""
        # Create list of tensors (they will have variable sequence length)
        input_sequences = [
            seq.clone().detach().to(device=self.device, dtype=t.long)
            for seq in batch['input_sequences']
        ]
        
        # Stack sequences
        input_tensor = t.stack(input_sequences)
        indices = batch['indices']
        
        # Use autocast for mixed precision if on GPU
        if self.use_amp:
            with t.cuda.amp.autocast():
                with t.no_grad():
                    logits = self.model(input_tensor)
                    probs = F.softmax(logits, dim=-1)[:, -1, :]
        else:
            with t.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=-1)[:, -1, :]

        # Move to CPU and convert to numpy immediately to free GPU memory
        probs_np = probs.cpu().to(dtype=getattr(t, self.config.precision)).numpy()

        # Clear cache after each batch if using GPU
        if self.config.device == "cuda":
            t.cuda.empty_cache()

        return indices, probs_np

    def run_processing(self) -> None:
        """Run the complete processing pipeline."""
        # Load data
        df = pd.read_parquet(self.config.input_file)

        # Create memory-mapped array or regular array based on config
        vocab_size = self.model.cfg.d_vocab
        dtype = np.float16 if self.config.precision == "float16" else np.float32

        if self.config.use_memmap:
            # Create memory-mapped array
            self.config.output_dir.mkdir(exist_ok=True, parents=True)
            softmax_probs = np.memmap(
                self.config.output_file,
                dtype=dtype,
                mode='w+',
                shape=(len(df), vocab_size)
            )
        else:
            # Regular numpy array
            softmax_probs = np.zeros((len(df), vocab_size), dtype=dtype)

        # Process in chunks to reduce memory usage
        total_batches = len(df) // self.config.batch_size + (1 if len(df) % self.config.batch_size > 0 else 0)

        # Process dataloader in chunks
        dataloader = self._create_dataloader(df)

        for chunk_idx in range(0, len(dataloader), self.config.chunk_size):
            chunk_end = min(chunk_idx + self.config.chunk_size, len(dataloader))
            chunk_loader = list(dataloader)[chunk_idx:chunk_end]

            for batch in tqdm(chunk_loader,
                             desc=f"Processing Chunk {chunk_idx//self.config.chunk_size + 1}/{(len(dataloader)-1)//self.config.chunk_size + 1}"):
                indices, probs = self.process_batch(batch)
                softmax_probs[indices] = probs

            # Flush memory-mapped array after each chunk
            if self.config.use_memmap:
                softmax_probs.flush()

        # If using memmap, we need to explicitly close it
        if self.config.use_memmap:
            # The flush should have already saved it
            del softmax_probs
        else:
            # Save results for regular array
            np.save(self.config.output_file, softmax_probs)

        print(f"Saved softmax probabilities to: {self.config.output_file}")
        print("Inference complete...")

def main():
    # Define configuration
    config = ProcessingConfig(
        input_file=Path('data/pile_sub_l256-512_s16.parquet'),
        output_dir=Path('experiments'),
        model_type='baseline',
        batch_size=8,  # Increased batch size
        chunk_size=1000,  # Process 1000 batches at a time
        use_memmap=True,  # Use memory mapping
        precision="float32"  # Use lower precision
    )
    
    # Run processing
    processor = ModelProcessor(config)
    processor.run_processing()


if __name__ == "__main__":
    main()
