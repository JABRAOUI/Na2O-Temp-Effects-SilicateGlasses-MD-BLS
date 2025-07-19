import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import math

def read_data_from_file(filename):
    """Read data from file in value-count format."""
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.strip() == '' or line.startswith('#'):
                continue
            try:
                value, count = map(float, line.strip().split())
                data[value] = int(count)
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
    return data

# Composition data: {Na2O%: number_of_silicon_atoms}
composition_data = {
    0: 25600, 5: 24320, 10: 23040, 15: 21760,
    20: 20480, 25: 19200, 30: 17920, 35: 16640
        }

i_values = sorted(composition_data.keys())

# Create figure with subplots
num_files = len(i_values)
num_cols = 2
num_rows = math.ceil(num_files / num_cols)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 3 * num_rows))
fig.tight_layout(pad=4.0)
axes = axes.flatten()

# Determine global x-axis range
all_values = []
for i in i_values:
    data = read_data_from_file(f"Volume_Si_X_{i}")  # Updated file name pattern
    values = np.array([val for val, count in data.items() for _ in range(int(count))])
    all_values.extend(values)
global_min, global_max = min(all_values), max(all_values)
x_range = np.linspace(global_min, global_max, 1000)

for idx, i in enumerate(i_values):
    data = read_data_from_file(f"Volume_Si_X_{i}")  # Updated file name pattern
    values = np.array([val for val, count in data.items() for _ in range(int(count))])
    total_silicon = composition_data[i]  # Now using silicon atom count
    
    # Plot histogram showing percentage of silicon atoms
    counts, bins, _ = axes[idx].hist(values, bins=50, alpha=0.6, color='blue',
                                   weights=np.ones_like(values)/total_silicon*100,
                                   range=(global_min, global_max))
    
    # Get bin centers for KDE fitting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Create weighted values for KDE fitting (repeating based on counts)
    weighted_values = np.repeat(bin_centers, counts.astype(int))
    
    # Fit KDE to the binned data
    if len(weighted_values) > 1:  # Need at least 2 points for KDE
        kde = gaussian_kde(weighted_values)
        kde_y = kde(x_range)
        
        # Scale KDE to match histogram maximum
        hist_max = counts.max()
        kde_max = kde_y.max()
        if kde_max > 0:
            scaling_factor = hist_max / kde_max
            kde_y_scaled = kde_y * scaling_factor
            
            # Ensure KDE doesn't exceed histogram maximum
            kde_y_scaled = np.clip(kde_y_scaled, 0, hist_max)
            
            axes[idx].plot(x_range, kde_y_scaled, color='green', label=f'NS{i} KDE')
            axes[idx].fill_between(x_range, kde_y_scaled, color='green', alpha=0.3)
    
    axes[idx].set_xlabel(r'Voronoi volume [Å$^3$]', fontsize=10)  # Corrected units
    axes[idx].set_ylabel('% of Si atoms', fontsize=10)
    axes[idx].set_xlim(global_min, global_max)
    axes[idx].legend(prop={'size': 8}, loc='upper right')

# Hide unused subplots
for idx in range(num_files, num_rows * num_cols):
    fig.delaxes(axes[idx])

plt.savefig("Si_Voronoi_volume_percentage.png", dpi=300, bbox_inches='tight')
plt.show()
