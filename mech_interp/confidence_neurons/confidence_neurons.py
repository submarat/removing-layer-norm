#%%
import sys
sys.path.append('/workspace/removing-layer-norm/mech_interp')
from load_models import load_nln_model, load_finetuned_model, load_baseline
from load_dataset import DataManager
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def entropy(x):
    x = F.softmax(x, dim=-1)
    return (-x * torch.log(x + 1e-12)).sum(dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nln_model = load_nln_model().to(device)
finetuned_model = load_finetuned_model().to(device)
baseline_model = load_baseline().to(device)
dataset = DataManager(dataset_name='luca-pile', batch_size=50, max_context=256, num_samples=1000)
dataloader = dataset.create_dataloader()

entropies_nln = []
entropies_finetuned = []
entropies_baseline = []
with torch.no_grad():
    for batch in dataloader:
        entropy_nln = entropy(nln_model(batch.to(device))).flatten()
        entropy_finetuned = entropy(finetuned_model(batch.to(device))).flatten()
        entropy_baseline = entropy(baseline_model(batch.to(device))).flatten()
        entropies_nln.append(entropy_nln)
        entropies_finetuned.append(entropy_finetuned)
        entropies_baseline.append(entropy_baseline)
    entropies_nln = torch.cat(entropies_nln)
    entropies_finetuned = torch.cat(entropies_finetuned)
    entropies_baseline = torch.cat(entropies_baseline)


plt.hist(entropies_nln.cpu().numpy(), bins=100, alpha=0.5, label='nln', histtype='step')
plt.hist(entropies_finetuned.cpu().numpy(), bins=100, alpha=0.5, label='finetuned', histtype='step')
plt.hist(entropies_baseline.cpu().numpy(), bins=100, alpha=0.5, label='baseline', histtype='step')
# plt.yscale('log')
plt.xlabel('entropy')
plt.ylabel('frequency')
plt.legend()
plt.show()

# %%

dataset = DataManager(dataset_name='luca-pile', batch_size=50, max_context=1, num_samples=1000)
dataloader = dataset.create_dataloader()

entropies_nln = []
entropies_finetuned = []
entropies_baseline = []
with torch.no_grad():
    for batch in dataloader:
        entropy_nln = entropy(nln_model(batch.to(device))).flatten()
        entropy_finetuned = entropy(finetuned_model(batch.to(device))).flatten()
        entropy_baseline = entropy(baseline_model(batch.to(device))).flatten()
        entropies_nln.append(entropy_nln)
        entropies_finetuned.append(entropy_finetuned)
        entropies_baseline.append(entropy_baseline)
    entropies_nln = torch.cat(entropies_nln)
    entropies_finetuned = torch.cat(entropies_finetuned)
    entropies_baseline = torch.cat(entropies_baseline)


plt.hist(entropies_nln.cpu().numpy(), bins=100, alpha=0.5, label='nln', histtype='step')
plt.hist(entropies_finetuned.cpu().numpy(), bins=100, alpha=0.5, label='finetuned', histtype='step')
plt.hist(entropies_baseline.cpu().numpy(), bins=100, alpha=0.5, label='baseline', histtype='step')
# plt.yscale('log')
plt.xlabel('entropy')
plt.ylabel('frequency')
plt.legend()
plt.show()

#%%

dataset = DataManager(dataset_name='luca-pile', batch_size=50, max_context=256, num_samples=1000)
dataloader = dataset.create_dataloader()

entropies_nln = []
entropies_finetuned = []
entropies_baseline = []
with torch.no_grad():
    for batch in dataloader:
        entropy_nln = entropy(nln_model(batch.to(device)))[:, -1].flatten()
        entropy_finetuned = entropy(finetuned_model(batch.to(device)))[:, -1].flatten()
        entropy_baseline = entropy(baseline_model(batch.to(device)))[:, -1].flatten()
        entropies_nln.append(entropy_nln)
        entropies_finetuned.append(entropy_finetuned)
        entropies_baseline.append(entropy_baseline)
    entropies_nln = torch.cat(entropies_nln)
    entropies_finetuned = torch.cat(entropies_finetuned)
    entropies_baseline = torch.cat(entropies_baseline)


plt.hist(entropies_nln.cpu().numpy(), bins=100, alpha=0.5, label='nln', histtype='step')
plt.hist(entropies_finetuned.cpu().numpy(), bins=100, alpha=0.5, label='finetuned', histtype='step')
plt.hist(entropies_baseline.cpu().numpy(), bins=100, alpha=0.5, label='baseline', histtype='step')
# plt.yscale('log')
plt.xlabel('entropy')
plt.ylabel('frequency')
plt.legend()
plt.show()
# %%
