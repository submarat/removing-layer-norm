# %%
import pandas as pd
import matplotlib.pyplot as plt

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

# Create the plot
plt.figure(figsize=(8, 4))

# Plot small model data (offset by 3)
plt.plot(df_small['train/global_step'], df_small['gpt2_aux-without_ln - main_loss'] + 3, 
         label='GPT-2 Small LN-free FT', color='blue', alpha=0.7)

# Plot medium model data (offset by 2)
plt.plot(df_medium['train/global_step'], df_medium['gpt2-medium_fasttune_aux-without_ln - main_loss'] + 2, 
         label='GPT-2 Medium LN-free FT', color='green', alpha=0.7)

# Plot large model data (offset by 1)
plt.plot(df_large['train/global_step'], df_large['gpt2-large_aux-without_ln - main_loss'] + 1, 
         label='GPT-2 Large LN-free FT', color='orange', alpha=0.7)

# Plot xl model data (no offset)
plt.plot(df_xl['train/global_step'], df_xl['gpt2-xl_aux-without_ln - main_loss'], 
         label='GPT-2 XL LN-free FT', color='red', alpha=0.7)

# Add horizontal lines at specific loss values
plt.axhline(y=3.1006 + 3, color='black', linestyle='-', alpha=0.5)  # Small
plt.axhline(y=2.8145 + 2, color='black', linestyle='-', alpha=0.5)  # Medium
plt.axhline(y=2.6623 + 1, color='black', linestyle='-', alpha=0.5)  # Large
plt.axhline(y=2.5567, color='black', linestyle='-', alpha=0.5)      # XL

# Add vertical lines for layer norm removal ranges
# Small model (from top of plot to Small OWT score)
plt.plot([20, 20], [7.5, 3.1006 + 3], color='blue', linestyle='--', alpha=0.3)
plt.plot([104, 104], [7.5, 3.1006 + 3], color='blue', linestyle='--', alpha=0.3)

# Medium model (from Small OWT score to Medium OWT score)
plt.plot([20, 20], [3.1006 + 3, 2.8145 + 2], color='green', linestyle='--', alpha=0.3)
plt.plot([188, 188], [3.1006 + 3, 2.8145 + 2], color='green', linestyle='--', alpha=0.3)

# Large model (from Medium OWT score to Large OWT score)
plt.plot([30, 30], [2.8145 + 2, 2.6623 + 1], color='orange', linestyle='--', alpha=0.3)
plt.plot([534, 534], [2.8145 + 2, 2.6623 + 1], color='orange', linestyle='--', alpha=0.3)

# XL model (from Large OWT score to XL OWT score)
plt.plot([50, 50], [2.6623 + 1, 2.5567], color='red', linestyle='--', alpha=0.3)
plt.plot([722, 722], [2.6623 + 1, 2.5567], color='red', linestyle='--', alpha=0.3)

# Add labels and title
plt.xlabel('Step')
plt.ylabel('Main training loss (Cross-entropy)')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust y-axis limits to better show all datasets
plt.ylim(2.5, 7.5)

# Set x-axis to start from 0
plt.xlim(0, None)

# Save the plot as PDF
plt.savefig('figure1.pdf', bbox_inches='tight', dpi=600)

# Show the plot
plt.show()

# Close the figure to free memory
plt.close()


# %%
