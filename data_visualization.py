# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# %%
# Loading 3 models that were finetuned on owt for much longer!
# Locations of the models on the gpu1server:
# with_ln: removing-layer-norm/results/gpt2/2025-03-24-22-19-10/checkpoint-4000
# slow: removing-layer-norm/results/gpt2/2025-03-25-11-21-50/checkpoint-4000
# fast: removing-layer-norm/results/gpt2/2025-03-24-22-19-26/checkpoint-4000

# %%
# Load the data
base_dir = "results/small_long_runs/"
data = pd.read_csv(os.path.join(base_dir, "losses.csv"))
lr = pd.read_csv(os.path.join(base_dir, "LR_Schedules.csv"))

# read a .log file which is a text file we want a string 
# with the content of the file
with open(os.path.join(base_dir, "fast.log"), 'r') as file:
    fast_log = file.read()
with open(os.path.join(base_dir, "slow.log"), 'r') as file:
    slow_log = file.read()
with open(os.path.join(base_dir, "with_ln.log"), 'r') as file:
    with_ln_log = file.read()

def process_log(a):
    a = a.split("\n")
    a = [x for x in a if x and "eval_pile_loss" in x]
    a = [x.split(",")[0] for x in a]
    a = [x.split(": ")[1] for x in a]
    a = [float(x) for x in a]
    return a

fast_log = process_log(fast_log)
slow_log = process_log(slow_log)
with_ln_log = process_log(with_ln_log)
eval_pos = np.linspace(1, 4000, 11, dtype=int)-1
eval_pos = eval_pos[1:]

# evals pos
long_eval_pos = [399, 1599, 2799, 3999]

luca_long_eval_with_ln = [2.8108, 2.8329, 2.8283, 2.8322]
luca_long_eval_fast = [2.8668, 2.9062, 2.9114, 2.9240]
luca_long_eval_slow = [69439.0070, 2.9380, 2.9281, 2.9426]

apollo_long_eval_with_ln = [2.8515, 2.8822, 2.8786, 2.8879]
apollo_long_eval_fast = [2.9101, 2.9700, 2.9741, 3.0005]
apollo_long_eval_slow = [70072.9269, 2.9861, 2.9839, 3.0163]

# %%
colors1 = ['#E6842A', '#F6B656']
colors2 = ['#137B80', '#42A5B3']
colors3 = ['#9A3E25', '#B37055']

train_loss_with_ln = data['gpt2_standard-with_ln - train/loss'] 
train_loss_without_ln = data['gpt2_standard-without_ln - train/loss']
train_loss_slow_without_ln = data['gpt2_standard_slow-without_ln - train/loss']

lr_with_ln = lr['gpt2_standard-with_ln - train/learning_rate']
lr_without_ln = lr['gpt2_standard-without_ln - train/learning_rate']
lr_slow_without_ln = lr['gpt2_standard_slow-without_ln - train/learning_rate']

train_loss_with_ln_ema = train_loss_with_ln.ewm(alpha=0.01).mean()
train_loss_without_ln_ema = train_loss_without_ln.ewm(alpha=0.01).mean()
train_loss_slow_without_ln_ema = train_loss_slow_without_ln.ewm(alpha=0.01).mean()

fig, ax = plt.subplots(5, 1, figsize=(8, 11), height_ratios=[1, 0.5, 0.5, 0.75, 0.3], sharex=True)

ax[0].plot(train_loss_slow_without_ln, alpha=0.5, lw=0.2, color=colors3[1])
ax[0].plot(train_loss_without_ln, alpha=0.5, lw=0.2, color=colors2[1])
ax[0].plot(train_loss_with_ln, alpha=0.5, lw=0.2, color=colors1[1])

ax[0].plot(train_loss_slow_without_ln_ema, label='LN slow removal', alpha=1, color=colors3[0])
ax[0].plot(train_loss_without_ln_ema, label='LN fast removal', alpha=1, color=colors2[0])
ax[0].plot(train_loss_with_ln_ema, label='LN model', alpha=1, color=colors1[0])

# Insert a vertical line at 100 steps
# step count might be off by +-2 steps.
for i in range(4):
    if i == 0:
        ax[i].axvline(118, color=colors2[0], linestyle='--', alpha=1, label='Fast removal done')
        ax[i].axvline(1160, color=colors3[0], linestyle=':', alpha=1, label='Slow removal done')
    else:
        ax[i].axvline(118, color=colors2[0], linestyle='--', alpha=1)
        ax[i].axvline(1160, color=colors3[0], linestyle=':', alpha=1)

ax[0].set_ylim(2.9, 3.1)
ax[0].set_xlim(0, 4000)

ax[0].set_ylabel('Train Loss')
ax[0].legend()

ax[1].plot(train_loss_slow_without_ln_ema-train_loss_slow_without_ln_ema[3999], label='noLN slow rem.', alpha=1, lw=0.9, color=colors3[0])
ax[1].plot(train_loss_without_ln_ema-train_loss_without_ln_ema[3999], label='noLN fast rem.', alpha=1, lw=0.9, color=colors2[0])
ax[1].plot(train_loss_with_ln_ema-train_loss_with_ln_ema[3999], label='LN model', alpha=1, lw=0.9, color=colors1[0])

ax[1].set_ylim(0, 0.05)
ax[1].legend()
ax[1].set_title('EWA Train Losses Relative to Final Value')
ax[1].set_ylabel('Train Loss - Train Loss [4k]')

ax[2].scatter(eval_pos[2:], slow_log[2:], label='noLN slow rem.', color=colors3[0], marker='P')
ax[2].scatter(eval_pos, fast_log, label='noLN fast rem.', color=colors2[0], marker='X')
ax[2].scatter(eval_pos, with_ln_log, label='LN model', color=colors1[0], marker='s')
ax[2].set_ylabel('Pile Eval Loss')
ax[2].legend()
ax[2].set_title('1k Apollo Pile Eval Losses captured during training')

ax[3].scatter(long_eval_pos, luca_long_eval_slow, label='Luca Pile noLN slow rem.', color='k', marker='P')
ax[3].scatter(long_eval_pos, luca_long_eval_fast, label='Luca Pile noLN fast rem.', color='k', marker='X')
ax[3].scatter(long_eval_pos, luca_long_eval_with_ln, label='Luca Pile LN model', color='k', marker='s')
ax[3].scatter(long_eval_pos, apollo_long_eval_slow, label='Apollo Pile noLN slow rem.', color='red', marker='P')
ax[3].scatter(long_eval_pos, apollo_long_eval_fast, label='Apollo Pile noLN fast rem.', color='red', marker='X')
ax[3].scatter(long_eval_pos, apollo_long_eval_with_ln, label='Apollo Pile LN model', color='red', marker='s')

ax[3].set_title('Luca Pile and Apollo Pile 10k Eval Losses')
ax[3].set_ylim(2.8, 3.2)
ax[3].set_ylabel('Pile Eval Loss')
# position the legend above the plot and resize the legend font
ax[3].legend(loc='upper center', ncol=2)
# ax[3].legend(loc='upper center', ncol=6)

# plot the learning rate
ax[4].plot(lr_without_ln, label='noLN fast rem.', color=colors2[0])
ax[4].plot(lr_with_ln, label='LN model', color=colors1[0])
ax[4].plot(lr_slow_without_ln, label='noLN slow rem.', color=colors3[0])
ax[4].set_ylabel('Learning Rate')
ax[4].legend()
ax[4].set_title('Learning Rate Schedules')

ax[-1].set_xlabel('Global Step')
fig.suptitle(r'GPT2-Small Finetuning Losses using 22% of OWT2 Training Data (4k steps)')
fig.tight_layout()
# save the plot as a pdf
fig.savefig("results/loss_plot.pdf")
plt.show()

# %%



# Stefan HF noLN model
# luca-apollo-pile: 2.8452
# apollo-pile: 2.8916
# apollo-pile reported in paper: 2.900

# %% 
# Conclusions of this study:
# There seems to be some differences that remain between noLN and LN models.
# THese differences do not become smaller with more training.
# the slow removal does slighly worse (sorry for the different learning rate schedule, using the identical lr schedule, the model failed...)
# The fast removal does slightly better than the slow removal.
# Weak indication that maybe training too long between steps with LN removed is not a great idea.
# I find this somewhat unsatisfactory, because it indicates that LN models are on average better...
# and the model without LN cannot catch up with more training.
# ALso, it seems like we are not getting better at the pile eval task with more finetuning on owt, but also not worse

# Hypothesis: LN might become more important for larger models. 
# - Instabilities might have more layers to grow

