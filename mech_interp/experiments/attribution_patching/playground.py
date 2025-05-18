import matplotlib.pyplot as plt
import pickle

with open("/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_0/new_resid_pre_act_patch_result_vanilla.pkl", "rb") as f:
    resid_pre_act_patch_result = pickle.load(f)

with open("/workspace/removing-layer-norm/mech_interp/experiments/attribution_patching/results/prompts_0/new_resid_pre_attr_patch_result_vanilla.pkl", "rb") as f:
    resid_pre_attr_patch_result = pickle.load(f)

vmin=-7
vmax=7

plt.imshow(resid_pre_act_patch_result.mean(-1).cpu().numpy(), vmin=vmin, vmax=vmax, cmap="RdBu")
plt.colorbar()
plt.savefig("resid_pre_act_patch_result_prompts_0.png")
plt.close()

plt.imshow(resid_pre_attr_patch_result.mean(-1).cpu().numpy(), vmin=vmin, vmax=vmax, cmap="RdBu")
plt.colorbar()
plt.savefig("resid_pre_attr_patch_result_prompts_0.png")
plt.close()






