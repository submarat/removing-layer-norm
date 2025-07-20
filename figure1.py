# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style and colorblind-friendly palette
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# %%
# Load the data
df_small = pd.read_csv('data/small_losses.csv')
df_medium = pd.read_csv('data/medium_losses.csv')
df_large = pd.read_csv('data/large_losses.csv')
df_xl = pd.read_csv('data/xl_losses.csv')

# Filter data based on step limits
df_small = df_small[df_small['train/global_step'] <= 300]
df_medium = df_medium[df_medium['train/global_step'] <= 500]
df_large = df_large[df_large['train/global_step'] <= 600]
df_xl = df_xl[df_xl['train/global_step'] <= 800]

# Calculate min/max for each model (with margin)
small_y = df_small['gpt2_aux-without_ln - main_loss']
medium_y = df_medium['gpt2-medium_fasttune_aux-without_ln - main_loss']
large_y = df_large['gpt2-large_aux-without_ln - main_loss']
xl_y = df_xl['gpt2-xl_aux-without_ln - main_loss']

# Calculate the global data range for scaling
all_min = min(small_y.min(), medium_y.min(), large_y.min(), xl_y.min())
all_max = max(small_y.max(), medium_y.max(), large_y.max(), xl_y.max())
global_range = all_max - all_min

# Height ratios proportional to each subplot's data range divided by the global range
small_range = small_y.max() - small_y.min()
medium_range = medium_y.max() - medium_y.min()
large_range = large_y.max() - large_y.min()
xl_range = xl_y.max() - xl_y.min()
height_ratios = [small_range / global_range, medium_range / global_range, large_range / global_range, xl_range / global_range]

# Create 4 vertically stacked subplots with equal heights (default scaling)
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 4.5), gridspec_kw={'hspace': 0.05})

# Get the colorblind-friendly colors
colors = sns.color_palette()

# Font sizes
label_fontsize = 10
legend_fontsize = 9

def plot_model(ax, x, y, color, label, hline, vlines, ylabel, ylim):
    line, = ax.plot(x, y, label=label, color=color, alpha=0.7)
    hline_obj = ax.axhline(y=hline, color='black', linestyle='-', alpha=0.5, label='GPT-2 original OWT eval loss')
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    # Improvements: wider margin, drop bottom tick label, insert more y-ticks (max 7), always align one with hline
    if ylabel == 'Small':
        min_margin = 0.10 * (y.max() - y.min())
    else:
        min_margin = 0.05 * (y.max() - y.min())
    max_margin = 0.05 * (y.max() - y.min())
    ymin = y.min() - min_margin
    ymax = y.max() + max_margin
    max_ticks = min(7, int((y.max() - y.min()) / ((y.max() - y.min()) / 3)) + 1)
    if max_ticks < 3:
        max_ticks = 3
    ticks = np.linspace(y.min(), y.max(), max_ticks)
    # Replace the tick closest to hline with hline itself
    idx = np.argmin(np.abs(ticks - hline))
    ticks[idx] = hline
    yticks = [round(t, 2) for t in ticks]
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    # Show all y-tick labels
    labels = [f"{t:.2f}" for t in yticks]
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', labelsize=9)
    # Remove top and right spines for compactness
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Draw vertical lines spanning the full height of the subplot
    y_min, y_max = ax.get_ylim()
    vline_objs = []
    for v in vlines:
        vline_objs.append(ax.plot([v, v], [y_min, y_max], color='black', linestyle='--', alpha=0.3, label='LN removal start/end')[0])
    return line, hline_obj, vline_objs

# Store handles for legend
handles = []
labels = []

# Plot Small
l1, h1, v1 = plot_model(
    axes[0],
    df_small['train/global_step'],
    small_y,
    colors[0],
    'GPT-2 Small LN-free FT',
    3.1006,
    [20, 104],
    'Small',
    (small_y.min(), small_y.max())
)
handles.append(l1)
labels.append('GPT-2 Small LN-free FT')

# Plot Medium
l2, h2, v2 = plot_model(
    axes[1],
    df_medium['train/global_step'],
    medium_y,
    colors[1],
    'GPT-2 Medium LN-free FT',
    2.8145,
    [20, 188],
    'Medium',
    (medium_y.min(), medium_y.max())
)
handles.append(l2)
labels.append('GPT-2 Medium LN-free FT')

# Plot Large
l3, h3, v3 = plot_model(
    axes[2],
    df_large['train/global_step'],
    large_y,
    colors[2],
    'GPT-2 Large LN-free FT',
    2.6623,
    [30, 534],
    'Large',
    (large_y.min(), large_y.max())
)
handles.append(l3)
labels.append('GPT-2 Large LN-free FT')

# Plot XL
l4, h4, v4 = plot_model(
    axes[3],
    df_xl['train/global_step'],
    xl_y,
    colors[3],
    'GPT-2 XL LN-free FT',
    2.5567,
    [50, 722],
    'XL',
    (xl_y.min(), xl_y.max())
)
handles.append(l4)
labels.append('GPT-2 XL LN-free FT')

# Add a single legend for the top subplot, overlapping the chart lines, with white background
import matplotlib.lines as mlines
hline_legend = mlines.Line2D([], [], color='black', linestyle='-', alpha=0.5, label='GPT-2 original OpenWebText eval loss')
vline_legend = mlines.Line2D([], [], color='black', linestyle='--', alpha=0.3, label='LN removal start/end')
legend = axes[0].legend(
    handles=[hline_legend, vline_legend],
    labels=['GPT-2 original OpenWebText eval loss', 'LN removal start/end events'],
    fontsize=legend_fontsize,
    loc='upper right',
    frameon=True
)
legend.get_frame().set_facecolor('white')

# Only the bottom subplot gets the x-axis label and x-tick labels
axes[3].set_xlabel('Step', fontsize=label_fontsize)
axes[3].set_xlim(0, None)
axes[3].tick_params(axis='both', labelsize=9, labelbottom=True)
axes[3].locator_params(axis='x', nbins=32)

# Remove x-axis labels and x-tick labels from all but the bottom subplot for compactness
for ax in axes[:-1]:
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)

# Tight layout for compactness
plt.tight_layout(pad=0.1, rect=[0, 0, 1, 1])

# Ensure x-tick labels are visible on the bottom subplot
axes[3].tick_params(labelbottom=True)

# Save the plot as PDF
plt.savefig('figure1.pdf', bbox_inches='tight', dpi=600)

# Show the plot
plt.show()

# Close the figure to free memory
plt.close()


# %%
