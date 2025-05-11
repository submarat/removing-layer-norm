#%%
import sys
import datasets
sys.path.append("/workspace/removing-layer-norm/")
from mech_interp.load_models import load_finetuned_model
model_finetuned = load_finetuned_model()
from mech_interp.load_dataset import DataManager
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from copy import deepcopy
import copy  
import torch.optim as optim
from torch.nn import MSELoss
from tqdm import tqdm

def extract_attention_block(model, block_idx=0):
    """
    Extract just the attention component (with all heads) from a transformer block.
    
    Args:
        model: The transformer model to extract from
        block_idx: Index of the block containing the attention (default: 0)
    
    Returns:
        A standalone attention module with all heads
    """
    # Verify block_idx is valid
    if block_idx < 0 or block_idx >= len(model.blocks):
        raise ValueError(f"Block index {block_idx} is out of range (0-{len(model.blocks)-1})")
    
    # Extract the specified attention module
    attn = deepcopy(model.blocks[block_idx].attn)
    
    # Create a wrapper module that handles the full attention component
    class AttentionWrapper(nn.Module):
        def __init__(self, attn, model):
            super().__init__()
            self.attn = attn
            self.embed = model.embed  # Add the embedding layer
            
        def forward(self, input_ids):
            # First embed the tokens
            x = self.embed(input_ids)
            # Then pass through attention
            out = self.attn(x, x, x)
            return out
    
    # Create the wrapper with the extracted attention and embedding layer
    attn_block = AttentionWrapper(attn, model)
    
    return attn_block

dm = DataManager(dataset_name="apollo-owt", batch_size=10, max_context=1024, num_samples=1000)
dataloader = dm.create_dataloader()
# Extract the attention component from a block
attention_block = extract_attention_block(model_finetuned, block_idx=0)
attention_block_pre_training = copy.deepcopy(attention_block)
# Test the extracted attention with a sample from the dataloader
batch = next(iter(dataloader))
with torch.no_grad():
    output = attention_block(batch)
    
print(f"Attention block output shape: {output.shape}")



# Create data manager for OpenWebText dataset
data_manager = DataManager(
    dataset_name='apollo-owt',
    batch_size=16,
    max_context=128,
    num_samples=5000
)
train_dataloader = data_manager.create_dataloader()

# Define optimizer and loss function
optimizer = optim.Adam(attention_block.attn.parameters(), lr=1e-4)  # Only optimize the attention parameters
criterion = MSELoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Create two copies of model_finetuned
model_to_mlp_in = copy.deepcopy(model_finetuned)
model_to_attn_out = copy.deepcopy(model_finetuned)

# Turn off all gradients for teacher model
for param in model_to_mlp_in.parameters():
    param.requires_grad = False

# For student model, only enable gradients for attention after LN1
for param in model_to_attn_out.parameters():
    param.requires_grad = False
    
# Enable gradients only for attention parameters after LN1 in block 0
for name, param in model_to_attn_out.blocks[0].attn.named_parameters():
    param.requires_grad = True

# Move models to device
model_to_mlp_in.to(device)
model_to_attn_out.to(device)

# Training settings
num_epochs = 10
optimizer = optim.Adam(
    [p for p in model_to_attn_out.parameters() if p.requires_grad], 
    lr=1e-4
)
criterion = nn.MSELoss()
# Initialize list to track loss history
loss_history = []

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Get teacher output (up to mlp_in)
        with torch.no_grad():
            _, teacher_cache = model_to_mlp_in.run_with_cache(batch)
            teacher_output = teacher_cache['blocks.0.ln2.hook_normalized']
        teacher_output = copy.deepcopy(teacher_output).requires_grad_(True)
        # Get student output (up to attn_out)
        _, student_cache = model_to_attn_out.run_with_cache(batch)
        student_output = student_cache['blocks.0.hook_attn_out']
        
        
        # Calculate loss and update
        loss = criterion(student_output, teacher_output)    
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

# %%
