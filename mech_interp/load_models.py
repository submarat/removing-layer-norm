import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Literal, Tuple
import itertools

import torch
from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from huggingface_hub import snapshot_download


# Type aliases for clarity
ModelName = Literal['baseline', 'finetuned', 'noLN']
ModelSize = Literal['small', 'medium']
DeviceType = Union[str, torch.device]

# GPT-2 model size to HuggingFace repo mapping
GPT2_SIZE_TO_REPO = {
    'small': 'gpt2',            # 124M parameters
    'medium': 'gpt2-medium',    # 355M parameters
}


class ModelLoader(ABC):
    """Abstract base class for loading models with consistent interface."""
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[DeviceType] = None,
        repo_id: str = "gpt2",
        revision: str = "main",
    ):
        """Initialize model loader with model path."""
        # Set up device
        if device is not None:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # Only keep what's needed
        self.repo_id = repo_id  # For downloading
        self.revision = revision
        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)

    def download_if_needed(self):
        """Download the model if it doesn't exist locally."""
        if not os.listdir(self.model_path):  # Check if directory is empty
            print(f"Downloading {self.repo_id} ({self.revision}) to {self.model_path}...")
            try:
                snapshot_download(
                    repo_id=self.repo_id,
                    revision=self.revision,
                    local_dir=self.model_path
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
        else:
            print(f"Model already exists at {self.model_path}")

    @abstractmethod
    def load(self, fold_ln: bool = True, center_unembed: bool = False, eval_mode: bool = True) -> HookedTransformer:
        """Load model with specified parameters."""
        pass


class StandardModelLoader(ModelLoader):
    """Loader for standard GPT-2 models (baseline and finetuned)."""
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str,
        device: Optional[DeviceType] = None,
        repo_id: str = "gpt2",
        revision: str = "main",
    ):
        super().__init__(model_path, device, repo_id, revision)
        self.base_model_name = base_model_name
    
    def load(self,
            fold_ln: bool = True,
            center_writing_weights: bool = False,
            center_unembed: bool = False,
            eval_mode: bool = True) -> HookedTransformer:
        """Load a standard GPT-2 model."""
        self.download_if_needed()
        
        # Load the HuggingFace model first
        hf_model = GPT2LMHeadModel.from_pretrained(self.model_path)
        
        # Use the base_model_name directly
        model = HookedTransformer.from_pretrained(
            self.base_model_name,
            hf_model=hf_model,
            fold_ln=fold_ln, 
            center_unembed=center_unembed,
            center_writing_weights=center_writing_weights,
            device=self.device,
        )
        
        if eval_mode:
            model.eval()
            
        return model


class NoLNModelLoader(ModelLoader):
    """Loader for the no-layer-norm GPT-2 model."""
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str,
        device: Optional[DeviceType] = None,
        repo_id: str = "gpt2",
        revision: str = "main",
    ):
        super().__init__(model_path, device, repo_id, revision)
        self.base_model_name = base_model_name
    
    def load(self,
            fold_ln: bool = True,
            center_writing_weights: bool = False,
            center_unembed: bool = False,
            eval_mode: bool = True) -> HookedTransformer:
        """Load the no-layer-norm GPT-2 model."""
        self.download_if_needed()
        
        model = GPT2LMHeadModel.from_pretrained(self.model_path)
        
        # Scale down LayerNorm weights
        for block in model.transformer.h:
            block.ln_1.weight.data = block.ln_1.weight.data / 1e6
            block.ln_1.eps = 1e-5
            block.ln_2.weight.data = block.ln_2.weight.data / 1e6
            block.ln_2.eps = 1e-5
        model.transformer.ln_f.weight.data = model.transformer.ln_f.weight.data / 1e6
        model.transformer.ln_f.eps = 1e-5
        
        # Create custom HookedTransformer that removes LayerNorms
        class HookedTransformerNoLN(HookedTransformer):
            def removeLN(self):
                for i in range(len(self.blocks)):
                    self.blocks[i].ln1 = torch.nn.Identity()
                    self.blocks[i].ln2 = torch.nn.Identity()
                self.ln_final = torch.nn.Identity()
        
        # Use the base_model_name directly
        hooked_model = HookedTransformerNoLN.from_pretrained(
            self.base_model_name,  # 'gpt2' or 'gpt2-medium'
            hf_model=model, 
            fold_ln=fold_ln, 
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            device=self.device,
        )
        
        hooked_model.removeLN()
        hooked_model.cfg.normalization_type = None
        
        if eval_mode:
            hooked_model.eval()
            
        return hooked_model


class ModelFactory:
    """Factory for creating and accessing model variants."""
    
    def __init__(
        self, 
        model_names: List[ModelName], 
        model_dir: str,
        device: Optional[DeviceType] = None,
        fold_ln: bool = True,
        center_unembed: bool = True,
        center_writing_weights: bool = True,
        eval_mode: bool = True,
        model_size: ModelSize = "small"
    ):
        """
        Initialize with specified models and load them.
        
        Args:
            model_names: List of model variants to load
            model_dir: Directory to store/load models
            device: Device to load models on
            fold_ln: Whether to fold layer norm
            center_unembed: Whether to center unembed
            center_writing_weights: Whether to center writing weights
            eval_mode: Whether to put models in eval mode
            model_size: Size of GPT-2 model to use ('small' or 'medium')
        """
        self.model_dir = model_dir
        self.device = device
        self.fold_ln = fold_ln
        self.center_writing_weights = center_writing_weights
        self.center_unembed = center_unembed
        self.eval_mode = eval_mode
        self.model_size = model_size
        
        # Load requested models
        self.models: Dict[str, HookedTransformer] = self._load_models(model_names)
        self.model_pairs: List[Tuple[str, str]] = list(itertools.combinations(model_names, 2))

    def _load_models(self, model_names: List[ModelName]) -> Dict[str, HookedTransformer]:
        """Load multiple models by name."""
        models_dict = {}
        
        # Define repositories for each model type and size
        model_repos = {
            'small': {
                'baseline': "gpt2",
                'finetuned': "schaeff/gpt2-small_vanilla300",
                'noLN': "schaeff/gpt2-small_LNFree300"
            },
            'medium': {
                'baseline': "gpt2-medium",
                'finetuned': "schaeff/gpt2-medium_vanilla500",
                'noLN': "schaeff/gpt2-medium_LNFree500"
            }
        }
        
        # Define loader classes for each model type
        loader_classes = {
            'baseline': StandardModelLoader,
            'finetuned': StandardModelLoader,
            'noLN': NoLNModelLoader
        }
        
        # Get the base model name for the current size
        base_model_name = GPT2_SIZE_TO_REPO[self.model_size]
        
        # Create and use loaders
        for name in model_names:
            # Check if the model type is supported
            if name not in model_repos[self.model_size]:
                raise ValueError(f"Model '{name}' not supported for size '{self.model_size}'")
            
            # Get the repository for this model type and size
            repo_id = model_repos[self.model_size][name]
            
            # Create model path
            model_path = os.path.join(self.model_dir, name, self.model_size)
            
            # Create the loader with the base model name
            loader = loader_classes[name](
                model_path=model_path,
                base_model_name=base_model_name,
                device=self.device,
                repo_id=repo_id,
                revision="main"
            )
            
            # Load the model
            models_dict[name] = loader.load(
                fold_ln=self.fold_ln,
                center_unembed=self.center_unembed,
                center_writing_weights=self.center_writing_weights,
                eval_mode=self.eval_mode
            )
            
        return models_dict


# Example usage
if __name__ == '__main__':
    # Example usage with model size parameter
    factory = ModelFactory(
        ['baseline', 'finetuned', 'noLN'],
        model_dir="models",
        model_size="small"
    )
    
    baseline_model = factory.models['baseline']
    finetuned_model = factory.models['finetuned']
    noLN_model = factory.models['noLN']
    
    text = "Hello, my name is"
    tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_SIZE_TO_REPO[factory.model_size])
    tokens = tokenizer.encode(text, return_tensors="pt")
    
    with torch.no_grad():
        baseline_logit, baseline_token = torch.max(baseline_model(tokens)[:, -1, :], dim=-1)
        finetuned_logit, finetuned_token = torch.max(finetuned_model(tokens)[:, -1, :], dim=-1)
        noLN_logit, noLN_token = torch.max(noLN_model(tokens)[:, -1, :], dim=-1)
    
    print(f"Model size: {factory.model_size}")
    print(f"Input: '{text}'")
    print(f"Baseline: '{tokenizer.decode([baseline_token.item()])}', logit = {baseline_logit.item():.2f}")
    print(f"Finetuned: '{tokenizer.decode([finetuned_token.item()])}', logit = {finetuned_logit.item():.2f}")
    print(f"NoLN: '{tokenizer.decode([noLN_token.item()])}', logit = {noLN_logit.item():.2f}")
