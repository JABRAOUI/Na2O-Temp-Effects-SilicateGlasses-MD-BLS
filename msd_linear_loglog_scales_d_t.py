import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import linregress
from scipy.optimize import curve_fit
from tabulate import tabulate

# Configure LaTeX and style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,siunitx}'
sns.set_style("whitegrid")
plt.rcParams['axes.grid'] = True

# Set global font sizes
plt.rcParams.update({
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Unit conversions
fs_to_s = 1e-15
A2_to_cm2 = 1e-16
kB = 8.617333262145e-5  # Boltzmann constant in eV/K

# Define base directory
base_dir = os.path.expanduser("~/Desktop/Glass_projet/Na2O_Silica_Glass/New_simulations/MSD_calculations")

# Setup figure with 1x3 layout
fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.3})
fig.suptitle(r'X=05\% Na$_2$O', y=1.05)

# Color palette
temperatures = range(300, 801, 100)
colors = sns.color_palette("plasma", n_colors=len(temperatures))

# Dictionary to store diffusion data
diffusion_data = {
    'temps': [],
    'D_values': [],
    'D_errors': [],
    'inv_T': [],
    'log_D': [],
    'log_D_errors': []
}

# ===== LEFT PANEL: Linear MSD =====
ax1 = axs[0]
for temp, color in zip(temperatures, colors):
    dir_name = f"Diffusion_X_05_{temp}K"
    file_name = f"Diff_X_05_{temp}_K.txt"
    file_path = os.path.join(base_dir, dir_name, file_name)
    
    try:
        data = np.loadtxt(file_path)
        time_s = data[:, 0] * fs_to_s
        msd_cm2 = data[:, 2] * A2_to_cm2
        
        ax1.plot(time_s, msd_cm2, 
                color=color, linewidth=2.5,
                marker='', linestyle='-',
                label=f'\SI{{{temp}}}{{\kelvin}}')
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")

# Configure linear plot
ax1.set_xlabel('Time (\\si{\\second})')
ax1.set_ylabel('MSD (\\si{\\centi\\meter\\squared})')
ax1.set_title('(a) Linear Scale')
ax1.set_xlim(0, 8e-9)
ax1.set_ylim(0, 14e-15)
ax1.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
ax1.legend(frameon=True, loc='upper left')

# ===== MIDDLE PANEL: Log-Log MSD =====
ax2 = axs[1]
for temp, color in zip(temperatures, colors):
    dir_name = f"Diffusion_X_05_{temp}K"
    file_name = f"Diff_X_05_{temp}_K.txt"
    file_path = os.path.join(base_dir, dir_name, file_name)
    
    try:
        data = np.loadtxt(file_path)
        time_s = data[:, 0] * fs_to_s
        msd_cm2 = data[:, 2] * A2_to_cm2
        
        mask = (time_s > 0) & (msd_cm2 > 0)
        ax2.loglog(time_s[mask], msd_cm2[mask],
                  color=color, linewidth=2.5,
                  marker='', linestyle='-',
                  alpha=0.9,
                  label=f'\SI{{{temp}}}{{\kelvin}}')
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")

# Configure log-log plot
ax2.set_xlabel('Time (\\si{\\second})')
ax2.set_ylabel('MSD (\\si{\\centi\\meter\\squared})')
ax2.set_title('(b) Log-Log Scale')
ax2.xaxis.set_major_locator(plt.LogLocator(numticks=6))
ax2.yaxis.set_major_locator(plt.LogLocator(numticks=6))
ax2.grid(True, which="both", ls="-", alpha=0.2)
ax2.legend(frameon=True, loc='upper left')

# ===== RIGHT PANEL: Diffusion Coefficients vs Temperature =====
ax3 = axs[2]
for temp, color in zip(temperatures, colors):
    dir_name = f"Diffusion_X_05_{temp}K"
    file_name = f"Diff_X_05_{temp}_K.txt"
    file_path = os.path.join(base_dir, dir_name, file_name)
    
    try:
        data = np.loadtxt(file_path)
        time_s = data[:, 0] * fs_to_s
        msd_cm2 = data[:, 2] * A2_to_cm2
        
        # Calculate diffusion coefficient (D = MSD/(6t) in cm²/s)
        # Using the last 20% of the data for linear fit
        n_points = len(time_s)
        start_idx = int(0.90 * n_points)
        slope, intercept, r_value, p_value, std_err = linregress(time_s[start_idx:], msd_cm2[start_idx:])
        D = slope / 6  # 3D diffusion
        D_error = std_err / 6  # Error propagation
        
        # Store data
        diffusion_data['temps'].append(temp)
        diffusion_data['D_values'].append(D)
        diffusion_data['D_errors'].append(D_error)
        diffusion_data['inv_T'].append(1/temp)
        diffusion_data['log_D'].append(np.log(D))
        # Error propagation for log(D): δlnD ≈ δD/D
        diffusion_data['log_D_errors'].append(D_error/D if D > 0 else 0)
        
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")

# Plot D vs T with error bars
ax3.errorbar(diffusion_data['temps'], diffusion_data['D_values'], 
             yerr=diffusion_data['D_errors'],
             fmt='o-', color='darkblue', markersize=8, linewidth=2,
             capsize=5, capthick=2, elinewidth=2)

ax3.set_xlabel('Temperature (\\si{\\kelvin})')
ax3.set_ylabel('D (\\si{\\centi\\meter\\squared\\per\\second})')
ax3.set_title('(c) Diffusion Coefficient vs Temperature')
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig('MSD_and_Diffusion_Analysis_X_05.png', dpi=300, bbox_inches='tight')

# ===== PRINT RESULTS TABLE =====
print("\nDiffusion Parameters Analysis with Errors")
print("="*70)

# Prepare data for table
table_data = []
for i in range(len(diffusion_data['temps'])):
    table_data.append([
        f"{diffusion_data['temps'][i]} K",
        f"{diffusion_data['inv_T'][i]:.5f} K⁻¹",
        f"({diffusion_data['D_values'][i]:.3e} ± {diffusion_data['D_errors'][i]:.3e}) cm²/s",
        f"({diffusion_data['log_D'][i]:.4f} ± {diffusion_data['log_D_errors'][i]:.4f})"
    ])

# Print the table
headers = ["Temperature", "1/T (K⁻¹)", "D (cm²/s)", "ln(D)"]
print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

plt.show()
