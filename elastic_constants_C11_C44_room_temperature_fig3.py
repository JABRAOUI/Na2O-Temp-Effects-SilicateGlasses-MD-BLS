import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Enable LaTeX rendering in Matplotlib and load the amsmath package
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Increase font sizes globally
sns.set(font_scale=1.5)  # Increase font size for seaborn
plt.rcParams['axes.labelsize'] = 16  # Axis label font size
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick font size
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick font size
plt.rcParams['legend.fontsize'] = 14  # Legend font size

# Simulation data
Na2O_percent = [0, 5, 10, 15, 20, 25, 30, 35] * 6  # 0-35% for 6 temperatures
temperature = [300]*8 + [400]*8 + [500]*8 + [600]*8 + [700]*8 + [800]*8  # temperature for each data point

# Updated simulation data for C11 and C44 (300K-800K)
C11_sim = [
    # 300K
    74.68, 69.92, 67.13, 66.25, 67.19, 70.12, 74.22, 77.91,
    # 400K
    74.76, 69.51, 66.08, 64.31, 65.56, 67.15, 70.57, 74.11,
    # 500K
    74.84, 69.02, 65.65, 62.95, 63.06, 64.08, 66.73, 69.39,
    # 600K
    74.99, 68.43, 64.76, 61.73, 60.91, 61.57, 62.86, 64.83,
    # 700K
    74.96, 67.96, 63.57, 59.27, 58.76, 57.41, 58.29, 59.76,
    # 800K
    73.83, 67.87, 63.51, 58.71, 56.71, 54.82, 54.55, 54.50
]

C44_sim = [
    # 300K
    26.93, 26.43, 24.80, 24.22, 23.50, 23.81, 24.37, 24.68,
    # 400K
    26.87, 26.25, 24.50, 23.49, 23.00, 23.24, 23.58, 23.91,
    # 500K
    26.91, 25.92, 24.10, 23.02, 22.42, 22.30, 22.65, 22.55,
    # 600K
    26.94, 25.66, 23.79, 22.70, 21.66, 21.59, 21.24, 21.47,
    # 700K
    26.89, 25.45, 23.57, 21.84, 20.97, 20.35, 20.16, 20.13,
    # 800K
    26.48, 25.40, 23.31, 21.56, 20.28, 19.71, 19.06, 18.40
]

# Experimental data
Na2O_C11_exp = np.array([0, 5.2, 6.3, 6.9, 7.6, 8.5, 9, 10.1, 10.7, 12, 13.7, 15.7, 18.5, 20.1, 22.8, 24.4, 26.9, 32.2, 36.5])
C11_exp = np.array([79.5, 74.78, 73.46, 73.45, 74.14, 73.13, 73.69, 72.89, 72.74, 72.55, 72.19, 71.42, 71.73, 71.63, 71.59, 72.51, 73.49, 74.12, 75.56])

Na2O_C44_exp = np.array([0, 5.2, 6.3, 6.9, 7.6, 8.5, 9, 10.1, 10.7, 12, 13.7, 15.7, 18.5, 20.1, 22.8, 24.4, 26.9, 32.2, 36.5, 39.9])
C44_exp = np.array([32.48, 30.48, 29.55, 29.32, 29.69, 28.99, 29.13, 29.37, 28.27, 28.25, 27.96, 27.23, 27.04, 26.38, 26.01, 25.65, 25.37, 24.96, 24.73, 24.91])

# Filter simulation data to include only Na2O percentages from 0 to 35 and only 300 K
mask = (np.array(Na2O_percent) <= 35) & (np.array(temperature) == 300)
Na2O_percent_filtered = np.array(Na2O_percent)[mask]
C11_sim_filtered = np.array(C11_sim)[mask]
C44_sim_filtered = np.array(C44_sim)[mask]

# Calculate percentage differences using absolute values
def calculate_diff(sim_values, exp_values, exp_x_values):
    diff = []
    for i, percent in enumerate(Na2O_percent_filtered):
        # Find the closest experimental Na2O percentage
        idx = np.argmin(np.abs(exp_x_values - percent))
        diff.append(np.abs((sim_values[i] - exp_values[idx]) / exp_values[idx]) * 100)
    return diff

C11_diff = calculate_diff(C11_sim_filtered, C11_exp, Na2O_C11_exp)
C44_diff = calculate_diff(C44_sim_filtered, C44_exp, Na2O_C44_exp)

# Print the calculated differences for verification
print("Na2O %:", Na2O_percent_filtered)
print("C11 differences (%):", C11_diff)
print("C44 differences (%):", C44_diff)

# Set the figure size explicitly
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid
axes = axes.flatten()  # Flatten the 2x2 grid into a 1D array for easier iteration

# Create a color palette for the different temperatures
palette = sns.color_palette("tab10", n_colors=4)

# Function to plot data with consistent style
def plot_data(ax, x, y_sim, y_exp, x_exp, ylabel, title, yrange=None):
    ax.plot(x, y_sim, label=f'Simulated 300 K', color=palette[0], linewidth=2.5, marker="o", markersize=8)
    ax.plot(x_exp, y_exp, label='Exp (Room Temp)', marker='*', markersize=10, color='black', linestyle='-')
    ax.set_xlabel(r"$\text{Na}_2\text{O}~(\%)$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    if yrange:  # Set y-axis limits if provided
        ax.set_ylim(yrange)

# Plot C11 with controlled yrange
plot_data(axes[0], Na2O_percent_filtered, C11_sim_filtered, C11_exp, Na2O_C11_exp,
          r"$C_{11}$ (GPa)", r"(a)", yrange=(60, 90))

# Plot C44 with controlled yrange
plot_data(axes[1], Na2O_percent_filtered, C44_sim_filtered, C44_exp, Na2O_C44_exp,
          r"$C_{44}$ (GPa)", r"$(b)$", yrange=(20, 40))

# Plot percentage differences with dual y-axes
ax_diff = axes[2]
ax_diff.plot(Na2O_percent_filtered, C11_diff, label=r'$C_{11}$ Difference (\%)', color='blue', linewidth=2.5, marker="o", markersize=8)
ax_diff.set_xlabel(r"$\text{Na}_2\text{O}~(\%)$", fontsize=16)
ax_diff.set_ylabel(r"$C_{11}$ Difference (\%)", fontsize=16, color='blue')
ax_diff.tick_params(axis='y', labelcolor='blue')
ax_diff.grid(True)
ax_diff.set_ylim(-1, 20)  # Adjusted for differences

# Add a second y-axis for C44 differences
ax_diff2 = ax_diff.twinx()
ax_diff2.plot(Na2O_percent_filtered, C44_diff, label=r'$C_{44}$ Difference (\%)', color='red', linewidth=2.5, marker="s", markersize=8)
ax_diff2.set_ylabel(r"$C_{44}$ Difference (\%)", fontsize=16, color='red')
ax_diff2.tick_params(axis='y', labelcolor='red')
ax_diff2.set_ylim(-1, 20)

# Set title for the third subplot
ax_diff.set_title(r"$(c)$", fontsize=16)

# Collect all legend handles and labels
handles_a, labels_a = axes[0].get_legend_handles_labels()
handles_c1, labels_c1 = ax_diff.get_legend_handles_labels()
handles_c2, labels_c2 = ax_diff2.get_legend_handles_labels()

# Combine all handles and labels (avoid duplicates for (a) and (b))
all_handles = handles_a[:2] + handles_c1 + handles_c2
all_labels = labels_a[:2] + labels_c1 + labels_c2

# Create a dummy subplot in the fourth position for the legend
axes[3].axis('off')
axes[3].legend(all_handles, all_labels, title="", loc='center', fontsize=14)

# Adjust layout
plt.tight_layout()

# Save the plot as an image
plt.savefig("elastic_constants_C11_C44_room_temp.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
