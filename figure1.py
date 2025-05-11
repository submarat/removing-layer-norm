# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Load the data
df_small = pd.read_csv('data/small_main_eval_loss.csv')
df_medium = pd.read_csv('data/medium_main_eval_losses.csv')
df_large = pd.read_csv('data/large_main_eval_losses.csv')

# Create the plot
plt.figure(figsize=(12, 6))

# Plot small model data
plt.plot(df_small['train/global_step'], df_small['gpt2_aux-without_ln - main_loss'], 
         label='Small Model - Training Loss', color='blue', alpha=0.7)
# Filter out NaN values for evaluation loss
small_eval = df_small.dropna(subset=['gpt2_aux-without_ln - eval/loss'])
plt.plot(small_eval['train/global_step'], small_eval['gpt2_aux-without_ln - eval/loss'], 
         label='Small Model - Evaluation Loss', color='red', linestyle='-', linewidth=2)

# Plot medium model data (offset by 1)
plt.plot(df_medium['train/global_step'], df_medium['gpt2-medium_fasttune_aux-without_ln - main_loss'] + 1, 
         label='Medium Model - Training Loss', color='green', alpha=0.7)
# Filter out NaN values for evaluation loss
medium_eval = df_medium.dropna(subset=['gpt2-medium_fasttune_aux-without_ln - eval/loss'])
plt.plot(medium_eval['train/global_step'], medium_eval['gpt2-medium_fasttune_aux-without_ln - eval/loss'] + 1, 
         label='Medium Model - Evaluation Loss', color='purple', linestyle='-', linewidth=2)

# Plot large model data (offset by 2)
plt.plot(df_large['train/global_step'], df_large['gpt2-large_aux-without_ln - main_loss'] + 2, 
         label='Large Model - Training Loss', color='orange', alpha=0.7)
# Filter out NaN values for evaluation loss
large_eval = df_large.dropna(subset=['gpt2-large_aux-without_ln - eval/loss'])
plt.plot(large_eval['train/global_step'], large_eval['gpt2-large_aux-without_ln - eval/loss'] + 2, 
         label='Large Model - Evaluation Loss', color='brown', linestyle='-', linewidth=2)

# Add labels and title
plt.xlabel('Training Steps')
plt.ylabel('Loss (Medium Model offset by +1, Large Model offset by +2)')
plt.title('Training and Evaluation Loss Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

# Adjust y-axis limits to better show all datasets
plt.ylim(2.5, 7.0)

# Set x-axis to start from 0
plt.xlim(0, None)

# Show the plot
plt.show()

