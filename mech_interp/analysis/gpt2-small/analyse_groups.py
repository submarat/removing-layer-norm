# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_parquet('metrics_comparison.parquet')
df = df[df.sequence_length >= 20]

# %% 
# Use colorblind-friendly style
plt.style.use('seaborn-v0_8-colorblind')

# Assuming df is your dataframe with all the columns mentioned
def analyze_divergences(df, low_threshold=0.02, high_threshold=0.20):
    """
    Analyze and visualize divergences between models.
    
    Parameters:
    - df: DataFrame with JS divergence columns
    - low_threshold: Threshold below which JS divergence is considered low
    - high_threshold: Threshold above which JS divergence is considered high
    """
    # Create masks for different regions based on thresholds
    high_baseline_finetuned = df['js_baseline_finetuned'] >= high_threshold
    low_baseline_finetuned = df['js_baseline_finetuned'] <= low_threshold
    
    high_baseline_noLN = df['js_baseline_noln'] >= high_threshold
    low_baseline_noLN = df['js_baseline_noln'] <= low_threshold
    
    # Define the disjoint regions with quadrant labels
    regions = {
        'A': high_baseline_noLN & low_baseline_finetuned,     # Top-Left (noLN effect)
        'C': low_baseline_noLN & low_baseline_finetuned,      # Bottom-Left (consensus)
        'B': high_baseline_noLN & high_baseline_finetuned,    # Top-Right (both effects)
        #'D': low_baseline_noLN & high_baseline_finetuned,     # Bottom-Right (finetuning effect)
    }
    
    # Region explanations for the legend
    region_explanations = {
        'A': 'noLN only',
        'B': 'finetuning or noLN',
        'C': 'no difference',
        #'D': 'finetuning only'
    }
    
    # Create a new column for region labels
    df['region'] = np.nan  # Default for points that don't fall into the extreme regions
    
    for region_name, mask in regions.items():
        df.loc[mask, 'region'] = region_name
    
    # Count examples in each region
    region_df = df.dropna(subset=['region'])  # Only keep examples in the defined regions
    region_counts = region_df['region'].value_counts().sort_index()  # Sort by region name (A, B, C, D)
    
    # Create a figure with two subplots in 2 rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Get colorblind-friendly colors from the current style
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = colors[:4]  # Get first 4 colors
    
    # Create scatter plot for each region with different colors
    for i, (region, explanation) in enumerate(region_explanations.items()):
        region_data = region_df[region_df['region'] == region]
        ax1.scatter(
            region_data['js_baseline_finetuned'], 
            region_data['js_baseline_noln'],
            c=colors[i],
            label=f"{region}: {explanation}",
            alpha=0.7,
            s=50
        )
    
    # Add region boundaries
    ax1.axhline(y=high_threshold, color='black', linestyle='--', alpha=0.5)
    ax1.axhline(y=low_threshold, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=high_threshold, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=low_threshold, color='black', linestyle='--', alpha=0.5)
    
    # Label the regions with quadrant labels, shifted slightly to avoid overlap
    ax1.text(low_threshold/2 - 0.02, high_threshold*1.2, 'A', ha='center', va='center', 
             fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    ax1.text(low_threshold/2 - 0.02, low_threshold/2 - 0.02, 'C', ha='center', va='center', 
             fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    ax1.text(high_threshold*1.2, high_threshold*1.2, 'B', ha='center', va='center', 
             fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
    ax1.text(high_threshold*1.2, low_threshold/2 - 0.02, 'D', ha='center', va='center', 
             fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
   
    # Set labels and title
    ax1.set_xlabel('JS Divergence (baseline vs finetuned)')
    ax1.set_ylabel('JS Divergence (baseline vs noLN)')
    ax1.set_title('Disjoint subgroup analysis')
    ax1.legend(loc='upper right')
    
    # Histogram of counts
    bar_labels = [f"{region}: {explanation}" for region, explanation in region_explanations.items()]
    x_positions = np.arange(len(region_counts))
    bars = ax2.bar(x_positions, region_counts.values, color=colors)
    # Set the x-tick positions and labels
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(bar_labels, rotation=45, ha='right')    
    ax2.set_yscale('log')
    ax2.set_ylabel('Counts (log scale)')
    
    # Add counts above each bar
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, region_df, region_counts

# Example usage:
fig, labeled_df, counts = analyze_divergences(df)
#fig.savefig('figures/subgroup_counts.png', dpi=200)
plt.show()

# %%
# Function to plot boxplots of sequence length by region
def plot_sequence_length_by_region(region_df):
    """
    Create boxplots showing the distribution of sequence lengths for each region.
    
    Parameters:
    - df: DataFrame with 'region' column and 'input_sequence' column
    - low_threshold, high_threshold: Thresholds for categorizing regions
    
    Returns:
    - Figure with boxplots
    """
    # Define region explanations
    region_explanations = {
        'A': 'noLN only',
        'B': 'finetuning or noLN',
        'C': 'no difference',
    }
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get colorblind-friendly colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = colors[:4]
    
    # Create a list of region labels in order A, B, C, D
    regions = ['A', 'B', 'C']
    region_df = region_df[region_df['region'].isin(regions)]  # Filter out any other regions
    
    # Prepare data for boxplot
    data = [region_df[region_df['region'] == region]['sequence_length'] for region in regions]
    
    # Create boxplot
    boxplot = ax.boxplot(data, patch_artist=True)
    
    # Color the boxes
    for i, box in enumerate(boxplot['boxes']):
        box.set(facecolor=colors[i], alpha=0.7)
    
    # Add additional details to plot
    ax.set_ylabel('Sequence Length')
    
    # Set x-tick labels with full explanations
    x_labels = [f"{region}: {region_explanations[region]}" for region in regions]
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add median values as text above each box
    for i, region in enumerate(regions):
        median = region_df[region_df['region'] == region]['sequence_length'].median()
        if not np.isnan(median):  # Check if there's data
            ax.text(i+1, median + 1, f'Median: {int(median)}', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Add counts for each region
    for i, region in enumerate(regions):
        count = len(region_df[region_df['region'] == region])
        ax.text(i+1, ax.get_ylim()[0], f'n={count}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

fig = plot_sequence_length_by_region(labeled_df)
#fig.savefig('figures/subgroup_seq_len.png', dpi=200)
plt.show()


# %%
def analyze_accuracy_by_region(region_df):
    """
    Analyze and visualize the accuracy of each model across different regions.
    
    Parameters:
    - df: DataFrame with 'region', prediction columns, and 'target_token'
    - low_threshold, high_threshold: Thresholds for categorizing regions
    
    Returns:
    - Figure with accuracy plot
    - DataFrame with accuracy statistics
    """
    # Calculate accuracy for each model
    region_df['baseline_correct'] = region_df['pred_baseline'] == region_df['target_token']
    region_df['finetuned_correct'] = region_df['pred_finetuned'] == region_df['target_token']
    region_df['noLN_correct'] = region_df['pred_noln'] == region_df['target_token']
    
    # Define region explanations
    region_explanations = {
        'A': 'noLN effect',
        'B': 'finetuning or noLN',
        'C': 'no difference',
        
    }

    
    # Calculate accuracy statistics by region
    accuracy_by_region = region_df.groupby('region').agg({
        'baseline_correct': 'mean',
        'finetuned_correct': 'mean',
        'noLN_correct': 'mean',
    }).reset_index()
    
    # Add sample counts
    region_counts = region_df['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']
    accuracy_by_region = accuracy_by_region.merge(region_counts, on='region')
    
    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart of accuracy by region
    regions = ['A', 'B', 'C']
    accuracy_by_region = accuracy_by_region[accuracy_by_region['region'].isin(regions)]
    accuracy_by_region = accuracy_by_region.set_index('region').loc[regions].reset_index()
    
    # Get colorblind-friendly colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create data for plot
    x_positions = np.arange(len(regions))
    bar_width = 0.25
    opacity = 0.8
    
    # Plot bars for each model
    bars1 = ax.bar(x_positions - bar_width, accuracy_by_region['baseline_correct'], 
                   bar_width, alpha=opacity, label='Baseline Model')
    bars2 = ax.bar(x_positions, accuracy_by_region['finetuned_correct'], 
                   bar_width, alpha=opacity, label='Finetuned Model')
    bars3 = ax.bar(x_positions + bar_width, accuracy_by_region['noLN_correct'], 
                   bar_width, alpha=opacity, label='noLN Model')
    
    # Add details to the plot
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    
    # Label the x-axis with region explanations
    ax.set_xticks(x_positions)
    x_labels = [f"{region}: {region_explanations[region]}" for region in regions]
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add accuracy values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.0,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Add sample counts for each region at the bottom
    for i, count in enumerate(accuracy_by_region['count']):
        ax.text(x_positions[i], -0.2, f'n={count}', ha='center', va='top', fontweight='bold')
    
    plt.tight_layout()
    return fig, accuracy_by_region


fig, _ = analyze_accuracy_by_region(labeled_df)
#fig.savefig('figures/subgroup_accuracies.png', dpi=200)
plt.show()

# %%
def analyze_ce_by_region(region_df):
    """
    Analyze and visualize the accuracy of each model across different regions.
    
    Parameters:
    - df: DataFrame with 'region', prediction columns, and 'target_token'
    - low_threshold, high_threshold: Thresholds for categorizing regions
    
    Returns:
    - Figure with accuracy plot
    - DataFrame with accuracy statistics
    """
    # Define region explanations
    region_explanations = {
        'A': 'noLN effect',
        'B': 'finetuning or noLN',
        'C': 'no difference',
        
    }
    
    # Calculate accuracy statistics by region
    ce_by_region = region_df.groupby('region').agg({
        'ce_baseline': 'mean',
        'ce_finetuned': 'mean',
        'ce_noln': 'mean',
    }).reset_index()
    
    # Add sample counts
    region_counts = region_df['region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']
    ce_by_region = ce_by_region.merge(region_counts, on='region')
    
    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart of accuracy by region
    regions = ['A', 'B', 'C']
    ce_by_region = ce_by_region[ce_by_region['region'].isin(regions)]
    ce_by_region = ce_by_region.set_index('region').loc[regions].reset_index()
    
    # Get colorblind-friendly colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Create data for plot
    x_positions = np.arange(len(regions))
    bar_width = 0.25
    opacity = 0.8
    
    # Plot bars for each model
    bars1 = ax.bar(x_positions - bar_width, ce_by_region['ce_baseline'], 
                   bar_width, alpha=opacity, label='Baseline Model')
    bars2 = ax.bar(x_positions, ce_by_region['ce_finetuned'], 
                   bar_width, alpha=opacity, label='Finetuned Model')
    bars3 = ax.bar(x_positions + bar_width, ce_by_region['ce_noln'], 
                   bar_width, alpha=opacity, label='noLN Model')
    
    # Add details to the plot
    ax.set_ylabel('CE Loss')
    
    # Label the x-axis with region explanations
    ax.set_xticks(x_positions)
    x_labels = [f"{region}: {region_explanations[region]}" for region in regions]
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add accuracy values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.0,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    return fig


fig = analyze_ce_by_region(labeled_df)
fig.savefig('figures/subgroup_ce_losses.png', dpi=200)
plt.show()

# %%
labeled_df.region.value_counts()

# %%
a = labeled_df[labeled_df.region == 'A'][['input_sequence']]
c = labeled_df[labeled_df.region == 'C'][['input_sequence']].sample(1000, replace=False)
print(len(a), len(c))
# %%
