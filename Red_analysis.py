import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
csv_file = '/Users/astr/Documents/Pigments/ScanQuantif/0617_Red_12_Plate1_R.csv'

Red_ctrl_file = '/Users/astr/Documents/Pigments/ScanQuantif/RedScotchCtrl.csv'
Rctrl_data = pd.read_csv(Red_ctrl_file)
red_ctrl = Rctrl_data.iloc[:, 2].mean()


# --- Load Data ---
# header=None: Your CSVs have no header row
# sep=',': Standard CSV separator
df = pd.read_csv(csv_file, header=None)

# Flatten the 16x24 matrix into a single list of 384 values
values = df.to_numpy().flatten()

## Add tags
#tags = np.r_[np.tile([1, 0], 12), np.zeros(360, dtype=int)]
tags = np.r_[np.zeros(384, dtype=int)]
#tags[0:48] = 1
#tags = np.r_[np.ones(24, dtype=int), np.tile([1, 0], 12), np.zeros(336, dtype=int)] # plate 6


f = pd.DataFrame({"value": values, "tag": tags})


# Map tags to readable labels for the legend
f['Group'] = f['tag'].map({1: 'Control', 0: 'Sample'})


# --- Plot Horizontal Histograms Side-by-Side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True) # sharey ensures Y scales match

# 1. Control Histogram (Horizontal)
sns.histplot(data=f[f['Group'] == 'Control'], 
             y='value',  # 'y' makes it horizontal
             ax=axes[0], 
             color='darkblue', 
             bins=20, 
             edgecolor='black')
axes[0].axhline(red_ctrl, color='red', linestyle='--', linewidth=2, label='Red')
axes[0].legend()
axes[0].set_title('Control Wells', fontsize=14)
axes[0].set_xlabel('Count (Frequency)', fontsize=12)
axes[0].set_ylabel('Red Intensity', fontsize=12)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# 2. Sample Histogram (Horizontal)
sns.histplot(data=f[f['Group'] == 'Sample'], 
             y='value', 
             ax=axes[1], 
             color='darkgreen', 
             bins=20, 
             edgecolor='black')
axes[1].axhline(red_ctrl, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Sample Wells', fontsize=14)
axes[1].set_xlabel('Count (Frequency)', fontsize=12)
axes[1].set_ylabel('') # Remove label since it's shared
axes[1].grid(axis='x', linestyle='--', alpha=0.5)

plt.suptitle(f'Red Intensity Distribution (Horizontal)\nFile: {csv_file}', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# --- Optional: Overlaid Horizontal Histogram ---
# Useful to see exactly where the distributions differ on the same scale
plt.figure(figsize=(8, 8))
sns.histplot(data=f, y='value', hue='Group', bins=20, alpha=0.5, element='step')
plt.axhline(red_ctrl, color='red', linestyle='--', linewidth=2, label='Control Red')
plt.title('Overlay: Control vs Sample (Horizontal)', fontsize=14)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Red Intensity', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.legend(title='Group')
plt.tight_layout()
plt.show()