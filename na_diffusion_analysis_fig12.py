import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.gridspec import GridSpec

# Configure LaTeX and style
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,siunitx}'
sns.set(font_scale=1.5)
plt.rcParams.update({
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':'
})

# Updated data from the table
Na2O_percent = [5, 10, 15, 20, 25, 30, 35]
temperatures = [300, 400, 500, 600, 700, 800]

# Diffusion coefficients (converted to cm²/s)
D_values = [
    [-0.1614e-8, 0.0007534e-8, 0.09062e-8, 0.09611e-8, 0.03390e-8, 0.03159e-8, 0.03544e-8],  # 300K
    [0.09061e-8, 0.2773e-8, 0.2476e-8, 0.1693e-8, 0.2218e-8, 0.2720e-8, 0.1694e-8],          # 400K
    [0.02076e-8, 1.265e-8, 1.618e-8, 1.779e-8, 1.195e-8, 1.173e-8, 0.9650e-8],                # 500K
    [0.1873e-7, 0.2958e-7, 0.5880e-7, 0.8158e-7, 0.7165e-7, 0.6389e-7, 0.5789e-7],           # 600K
    [0.2134e-7, 1.360e-7, 1.919e-7, 2.244e-7, 2.516e-7, 2.536e-7, 2.957e-7],                  # 700K
    [0.7710e-7, 2.775e-7, 5.116e-7, 6.678e-7, 7.707e-7, 8.049e-7, 9.297e-7]                   # 800K
]

# Error values (converted to cm²/s)
D_errors = [
    [0.0748e-8, 0.0003537e-8, 0.002329e-8, 0.001817e-8, 0.001527e-8, 0.001228e-8, 0.001036e-8],  # 300K
    [0.1302e-8, 0.0005861e-8, 0.007132e-8, 0.004305e-8, 0.003038e-8, 0.002897e-8, 0.002041e-8],  # 400K
    [0.1680e-8, 0.001622e-8, 0.001253e-8, 0.0009271e-8, 0.0008384e-8, 0.0006446e-8, 0.0004954e-8], # 500K
    [0.03507e-7, 0.03415e-7, 0.02901e-7, 0.02732e-7, 0.02220e-7, 0.01714e-7, 0.01268e-7],         # 600K
    [0.07195e-7, 0.05925e-7, 0.08648e-7, 0.05650e-7, 0.03574e-7, 0.04658e-7, 0.04625e-7],         # 700K
    [0.1164e-7, 0.1017e-7, 0.1230e-7, 0.1260e-7, 0.1133e-7, 0.09485e-7, 0.1171e-7]               # 800K
]

# Activation energies from table (converted from kJ/mol to eV)
Ea_values = [23.86/96.485, 25.88/96.485, 40.16/96.485, 39.03/96.485, 
             45.40/96.485, 46.27/96.485, 50.12/96.485]  # Convert kJ/mol to eV
Ea_errors = [0.03/96.485, 0.04/96.485, 0.08/96.485, 0.02/96.485,
             0.02/96.485, 0.01/96.485, 0.01/96.485]     # Convert kJ/mol to eV

# Create figure
fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1])

# 3D Surface Plot
ax1 = fig.add_subplot(gs[0], projection='3d')
X, Y = np.meshgrid(Na2O_percent, temperatures)
Z = np.array(D_values)

# Convert negative values to small positive for visualization
Z_vis = np.where(Z > 0, Z, 1e-12)  # Replace negatives with small positive value

surf = ax1.plot_surface(X, Y, Z_vis, cmap='viridis', edgecolor='k', alpha=0.8, linewidth=0.5)
ax1.view_init(elev=25, azim=60)
ax1.set_xlabel(r'Na$_2$O (\%)', labelpad=10)
ax1.set_ylabel('Temperature (K)', labelpad=10)
ax1.set_zlabel(r'$D$ (cm²/s)', labelpad=10)
ax1.set_title('(a) Diffusion Coefficient Surface', y=1.0)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='both', which='major', labelsize=12)

# Add colorbar with scientific notation
cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, pad=0.1)
cbar.set_label(r'$D$ (cm²/s)', fontsize=14)
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()

# Activation Energy Plot
ax2 = fig.add_subplot(gs[1])
ax2.errorbar(Na2O_percent, Ea_values, yerr=Ea_errors, 
             fmt='o-', markersize=8, capsize=5, capthick=2, elinewidth=2)
ax2.set_xlabel(r'Na$_2$O (\%)')
ax2.set_ylabel(r'Activation Energy ($E_a$) (eV)')
ax2.set_title('(b) Activation Energy vs Composition', y=1.0)
ax2.grid(True, linestyle='--', alpha=0.7)

# Add text annotation with conversion note
#ax2.text(0.05, 0.95, r'1 eV = 96.485 kJ/mol', 
#         transform=ax2.transAxes, fontsize=10, 
#         bbox=dict(facecolor='white', alpha=0.8))

# Save data to CSV
data = {
    'Na2O (%)': Na2O_percent,
    'Ea (eV)': Ea_values,
    'Ea_error (eV)': Ea_errors
}
df = pd.DataFrame(data)
df.to_csv('Na2O_Ea_data.csv', index=False)

plt.tight_layout()
plt.savefig("Na_diffusion_analysis.png", dpi=300, bbox_inches="tight")
plt.show()
