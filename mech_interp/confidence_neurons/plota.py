#%%
import sys
import torch
import matplotlib.pyplot as plt
sys.path.append('/workspace/removing-layer-norm/mech_interp')
from load_models import load_nln_model, load_finetuned_model, load_baseline, download_model_if_not_exists
from transformer_lens import HookedTransformer
from transformers import GPT2LMHeadModel

def load_baseline_no_preprocessing():
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2-small")
    return hooked_model

def load_baseline_full_preprocessing():
    hooked_model = HookedTransformer.from_pretrained("gpt2-small", fold_ln=True, center_writing_weights=True, center_unembed=True)
    return hooked_model

def load_finetuned_model_full_preprocessing():
    model_path = "/workspace/removing-layer-norm/mech_interp/models/apollo_gpt2_finetuned"
    download_model_if_not_exists(
    repo_id="apollo-research/gpt2_noLN",
    revision="vanilla_1200",
    local_dir=model_path
    )
    model = GPT2LMHeadModel.from_pretrained(model_path)
    hooked_model = HookedTransformer.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=True, center_writing_weights=True)
    return hooked_model

def load_nln_model_full_preprocessing():
    model_path = "/workspace/removing-layer-norm/mech_interp/models/apollo_gpt2_noLN" 
    download_model_if_not_exists(
    repo_id="apollo-research/gpt2_noLN",
    revision="main",
    local_dir=model_path
    )

    model = GPT2LMHeadModel.from_pretrained(model_path)
    for block in model.transformer.h:
        block.ln_1.weight.data = block.ln_1.weight.data / 1e6
        block.ln_1.eps = 1e-5
        block.ln_2.weight.data = block.ln_2.weight.data / 1e6
        block.ln_2.eps = 1e-5
    model.transformer.ln_f.weight.data = model.transformer.ln_f.weight.data / 1e6
    model.transformer.ln_f.eps = 1e-5
    
    # Properly replace LayerNorms by Identities
    class HookedTransformerNoLN(HookedTransformer):
        def removeLN(self):
            for i in range(len(self.blocks)):
                self.blocks[i].ln1 = torch.nn.Identity()
                self.blocks[i].ln2 = torch.nn.Identity()
            self.ln_final = torch.nn.Identity()
    
    hooked_model = HookedTransformerNoLN.from_pretrained("gpt2", hf_model=model, fold_ln=True, center_unembed=True, center_writing_weights=True)
    hooked_model.removeLN()
    hooked_model.cfg.normalization_type = None
    return hooked_model


def get_l2_norm_W_out_last(model):
    W_out_last = model.blocks[11].mlp.W_out
    l2_norm_W_out_last = torch.norm(W_out_last, p=2, dim=1)
    return l2_norm_W_out_last

def get_logit_var(model):
    W_out_mlp = model.blocks[11].mlp.W_out
    W_u = model.W_U
    # m: mlp hidden dim, d: model residual stream size, v: vocab size
    num = torch.einsum('md, dv -> mv', W_out_mlp, W_u) # mlp hidden dim x vocab size
    den_part1 = torch.norm(W_u, p=2, dim=0) # vocab size
    den_part2 = torch.norm(W_out_mlp, p=2, dim=1) # mlp hidden dim
    logit_var = torch.var(num / (den_part1.reshape(1, -1) * den_part2.reshape(-1, 1)), dim=1) # mlp hidden dim
    return logit_var

baseline_model = load_baseline_full_preprocessing()
finetuned_model = load_finetuned_model_full_preprocessing()
noln_model = load_nln_model_full_preprocessing()
#%%
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
for i, (model, model_name) in enumerate([(baseline_model, "Baseline"), (finetuned_model, "Finetuned"), (noln_model, "NLN")]):
    axs[i].set_title(model_name)
    l2_norm_W_out_last = get_l2_norm_W_out_last(model)
    logit_var = get_logit_var(model)
    axs[i].scatter(l2_norm_W_out_last.detach().cpu().numpy(), logit_var.detach().cpu().numpy(), alpha=0.5,) 
    
    # Calculate ratio and get top 6 indices
    ratio = l2_norm_W_out_last / logit_var
    # Entropy neurons indices 
    idxs = [584, 1611, 2378, 2123, 2870, 2910]
    x_coords = l2_norm_W_out_last.detach().cpu().numpy()[idxs]
    y_coords = logit_var.detach().cpu().numpy()[idxs]
    axs[i].scatter(x_coords, y_coords, alpha=1., label='Entropy neurons')
    
    # Add text labels for each point
    for idx, x, y in zip(idxs, x_coords, y_coords):
        axs[i].annotate(str(idx), (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Top ratio indices in red
    # axs[i].scatter(l2_norm_W_out_last.detach().cpu().numpy()[top_ratio_idxs], logit_var.detach().cpu().numpy()[top_ratio_idxs], alpha=1., color='red', label='Highest ratio neurons')
    
    axs[i].set_xlabel('L2 norm of W_out last mlp')
    axs[i].set_ylabel('Logit variance')
    axs[i].set_yscale('log')
    axs[i].legend()
plt.show()

#%%
def get_svd_W_u(model):
    W_u = model.W_U.T
    U, S, Vh = torch.svd(W_u)
    return U, S, Vh

def plot_svd_projections(model, ax, singular_values_lims, idxs=[584, 1611, 2378, 2123, 2870, 2910], legend_loc='upper left'):
    U, S, Vh = get_svd_W_u(model)
    device = model.W_out.device
    Vh = Vh.to(device)
    U = U.to(device) 
    S = S.to(device)
    S = S / S.max()

    ax.plot(torch.arange(*singular_values_lims), S.detach().cpu().numpy()[singular_values_lims[0]:singular_values_lims[1]], label='W_u singular values')
    for idx in idxs:
        W_out = model.blocks[11].mlp.W_out
        w_out = W_out[idx]
        w_out_projections = torch.matmul(w_out, Vh)
        w_out_abs_cosine_sim = torch.abs(w_out_projections) / torch.norm(w_out_projections, p=2, dim=0)
        ax.plot(torch.arange(*singular_values_lims), w_out_abs_cosine_sim.detach().cpu().numpy()[singular_values_lims[0]:singular_values_lims[1]], label=f'{idx}')
        ax.set_xlabel('Singular values idx/singular vectors idx')
        ax.set_ylabel('Normalized singular values\n or \n absolute cosine similarity \n of w_out with singular vectors')
    ax.legend(loc=legend_loc)
   
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
for i, (model, model_name) in enumerate([(baseline_model, "Baseline"), (finetuned_model, "Finetuned"), (noln_model, "noLN")]):
    axs[i].set_title(model_name)
    plot_svd_projections(model, axs[i], (0, 768), legend_loc='upper center')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
for i, (model, model_name) in enumerate([(baseline_model, "Baseline"), (finetuned_model, "Finetuned"), (noln_model, "noLN")]):
    axs[i].set_title(model_name)
    plot_svd_projections(model, axs[i], (730, 768), legend_loc='upper left')
    plt.tight_layout()
plt.show()

#%%
from datasets import load_dataset
dataset = load_dataset("lucabaroni/apollo-pile-filtered-10k", split="train")

def replace_i_th_activation_with_avg(activations, avg_activation, i):
    activations_copy = activations.clone()
    activations_copy[:, :, i] = avg_activation[:, :, i]
    return activations_copy
#%%
prompts = dataset[:2]
prompts = torch.tensor(prompts['input_ids'])


#%%
ce = torch.nn.CrossEntropyLoss( reduction='none')
for model in [baseline_model]:
    with torch.no_grad():
        targets = prompts[:, 1:].flatten()

        output, cache = model.run_with_cache(prompts)
        mlp_intermediate = cache['blocks.11.mlp.hook_post']
        

        w_out = model.blocks[11].mlp.W_out
        bias = model.blocks[11].mlp.b_out
        mlp_output = torch.matmul(mlp_intermediate, w_out) + bias

        def handcomputed_ln(x):
            x_mean = x.mean(dim=-1, keepdim=True)
            # x_std = torch.sqrt((x-x_mean).pow(2).mean(dim=-1, keepdim=True) + 1e-12)
            x_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, correction=0) + 1e-12)
            ln_x = (x-x_mean) / x_std
            return ln_x
        ln_out1 = model.ln_final(mlp_output)
        ln_out2 = handcomputed_ln(mlp_output)


        print((ln_out1 - ln_out2).abs().mean())        
        
        # logits = model.unembed(model.ln_final(mlp_output))

        # logits1 = model.unembed(handcomputed_ln(mlp_output, gamma, beta))
        # print(logits.shape)
        
        # plt.show()
        # logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        # loss = ce(logits.cuda(), targets.cuda())
        
       
        # # print(loss)
        # mlp_intermediate_neuron_avg = mlp_intermediate.mean(dim=[0, 1], keepdim=True)
        # losses_ablated = []
        # from tqdm import tqdm
        # for i in tqdm(range(mlp_intermediate.shape[2])):
        #     mlp_intermediate_ablated = replace_i_th_activation_with_avg(mlp_intermediate, mlp_intermediate_neuron_avg, i)
        #     mlp_output_ablated = torch.matmul(mlp_intermediate_ablated, w_out) + bias
        #     logits_ablated = model.unembed(model.ln_final(mlp_output_ablated))
        #     logits_ablated = logits_ablated[:, :-1, :].reshape(-1, logits_ablated.shape[-1])
        #     loss_ablated = ce(logits_ablated.cuda(), targets.cuda())
        #     losses_ablated.append(loss_ablated)
    
        
        # for i in [584]:
        #     mlp_ablated = replace_i_th_activation_with_avg(mlp_intermediate, mlp_intermediate_neuron_avg, i)
        #     mlp_output_ablated = torch.matmul(mlp_ablated, w_out) + bias
        #     logits_ablated = baseline_model.unembed(baseline_model.ln_final(mlp_output_ablated))
        #     loss_ablated = ce(logits_ablated, targets)
        
# idxs = [584, 1611, 2378, 2123, 2870, 2910]
# device = baseline_model.W_out.device
# Vh = Vh.to(device)
# U = U.to(device)
# S = S.to(device)
# S_normed = S / S.max()
# plt.plot(S_normed.detach().cpu().numpy(), label='singular values')
# for idx in idxs:
#     W_out = baseline_model.blocks[11].mlp.W_out
#     w_out = W_out[idx]
#     w_out_projections = torch.matmul(w_out, U)
#     w_out_projections_normed = w_out_projections / torch.norm(w_out_projections, p=2, dim=0)
#     plt.plot(w_out_projections_normed.detach().cpu().numpy(), label=f'projection of neuron {idx}')
# plt.ylim(0, 1)
# plt.xlim(0, 768)
# plt.legend(loc='upper center')
# plt.show()
# #%%
# idxs = [584, 1611, 2378, 2123, 2870, 2910]
# device = baseline_model.W_out.device
# Vh = Vh.to(device)
# U = U.to(device)
# S = S.to(device)
# S_normed = S / S.max()
# plt.plot(S_normed.detach().cpu().numpy(), label='singular values')
# for idx in idxs:
#     W_out = baseline_model.blocks[11].mlp.W_out
#     w_out = W_out[idx]
#     w_out_projections = torch.matmul(w_out, U)
#     w_out_projections_normed = w_out_projections / torch.norm(w_out_projections, p=2, dim=0)
#     plt.plot(w_out_projections_normed.detach().cpu().numpy(), label=f'projection of neuron {idx}')
# plt.ylim(-.05, 1)
# plt.xlim(-5, 768)
# plt.legend(loc='upper center')
# plt.show()

# %%
