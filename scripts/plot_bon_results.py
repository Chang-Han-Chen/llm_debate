import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the results
df = pd.read_csv("./exp/bon_comparison_gpt4o/results.csv")

# Create the plot
plt.figure(figsize=(8, 6))
color_palette = sns.color_palette("colorblind")

# Convert accuracy to percentage
df['accuracy'] = df['accuracy'] * 100

# Calculate mean and standard error if there are multiple runs per BoN value
results = df.groupby('bon_value')['accuracy'].agg(['mean', 'std']).reset_index()
results['std_err'] = results['std'] / np.sqrt(len(df['judge_model'].unique()))

# Plot with error bars
plt.errorbar(
    results['bon_value'],
    results['mean'],
    yerr=results['std_err'],
    fmt='o-',
    capsize=5,
    capthick=2,
    linewidth=2,
    markersize=8,
    color=color_palette[0],
    label='GPT-4-Turbo'
)

# Customize the plot
plt.xlabel('Best of N', fontsize=14)
plt.ylabel('Judge Accuracy (%)', fontsize=14)
plt.grid(True, which="both", linewidth=0.5, axis="y")
plt.gca().set_axisbelow(True)

# Set y-axis limits and ticks
plt.ylim(0, 100)
plt.yticks(np.arange(0, 110, 25))

# Add legend
plt.legend(fontsize=12)

# Ensure x-axis shows all BoN values
plt.xticks(results['bon_value'], fontsize=12)

# Save the plot
plt.savefig("./exp/bon_comparison_gpt4o/bon_results.png", dpi=300, bbox_inches="tight")
plt.show()