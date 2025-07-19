import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from tabulate import tabulate
from matplotlib.gridspec import GridSpec

# Configure LaTeX and style to match elastic constants plot
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

# Complete data for all Na2O percentages (5-35%)
data = {
    5: {'T': [300, 400, 500, 600, 700, 800],
        'D': [0.0007534, 0.2773, 1.265, 2.958, 1.360, 2.775],
        'D_err': [0.0003537, 0.0005861, 0.001622, 0.03415, 0.05925, 0.1017]},
    10: {'T': [300, 400, 500, 600, 700, 800],
         'D': [0.0007534, 0.2773, 1.265, 2.958, 13.60, 27.75],
         'D_err': [0.0003537, 0.0005861, 0.001622, 0.03415, 0.5925, 1.017]},
    15: {'T': [300, 400, 500, 600, 700, 800],
         'D': [0.9062, 0.2476, 1.618, 5.880, 19.19, 51.16],
         'D_err': [0.002329, 0.007132, 0.01253, 0.02901, 0.08648, 0.1230]},
    20: {'T': [300, 400, 500, 600, 700, 800],
         'D': [0.9611, 0.1693, 1.779, 8.158, 22.44, 66.78],
         'D_err': [0.001817, 0.004305, 0.0009271, 0.02732, 0.05650, 0.1260]},
    25: {'T': [300, 400, 500, 600, 700, 800],
         'D': [0.3390, 0.2218, 1.195, 7.165, 25.16, 77.07],
         'D_err': [0.001527, 0.003038, 0.0008384, 0.02220, 0.03574, 0.1133]},
    30: {'T': [300, 400, 500, 600, 700, 800],
         'D': [0.3159, 0.2720, 1.173, 6.389, 25.36, 80.49],
         'D_err': [0.001228, 0.002897, 0.0006446, 0.01714, 0.04658, 0.09485]},
    35: {'T': [300, 400, 500, 600, 700, 800],
         'D': [0.3544, 0.1694, 0.9650, 5.789, 29.57, 92.97],
         'D_err': [0.001036, 0.002041, 0.0004954, 0.01268, 0.04625, 0.1171]}
}

# Create figure with 2x1 grid layout
fig = plt.figure(figsize=(8, 10))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax = fig.add_subplot(gs[0])
legend_ax = fig.add_subplot(gs[1])
legend_ax.axis('off')

# Use the same color palette as elastic constants plot
palette = sns.color_palette("tab10", n_colors=len(data))

# Dictionary to store fitted parameters
fitted_params = {}

# Conversion factor from kJ/mol to eV
kJmol_to_eV = 0.010364

# Define Arrhenius function with overflow protection
def arrhenius(T, D0, Ea):
    R = 8.314  # J/(mol·K)
    exponent = -Ea / (R * T)
    exponent = np.clip(exponent, -700, 700)  # Prevent overflow
    return D0 * np.exp(exponent)

# Main plotting loop for all compositions
for (conc, values), color in zip(sorted(data.items()), palette):
    try:
        D_corrected = np.array(values['D']) * 1e-8  # Convert to cm²/s
        D_err_corrected = np.array(values['D_err']) * 1e-8
        
        # Perform weighted nonlinear fit
        popt, pcov = curve_fit(arrhenius, 
                              np.array(values['T']), 
                              D_corrected,
                              sigma=D_err_corrected,
                              p0=[1e-5, 50000],
                              maxfev=10000)
        
        D0, Ea = popt
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate R-squared
        residuals = D_corrected - arrhenius(np.array(values['T']), *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((D_corrected - np.mean(D_corrected))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Store fitted parameters (keeping kJ/mol for calculations)
        fitted_params[conc] = {
            'D0': D0,
            'D0_err': perr[0],
            'E0': Ea,
            'E0_err': perr[1],
            'R2': r_squared
        }
        
        # Calculate normalized values with proper error propagation
        norm_D = D_corrected / D0
        norm_D_err = norm_D * np.sqrt((D_err_corrected/D_corrected)**2 + (perr[0]/D0)**2)
        
        # Plot data points with error bars
        line = ax.errorbar(1/np.array(values['T']), np.log(norm_D),
                         yerr=norm_D_err/norm_D,
                         fmt='o', color=color,
                         capsize=4, capthick=1.5,
                         elinewidth=1.5, markersize=8)
        
        # Plot theoretical Arrhenius line
        T_fit = np.linspace(min(values['T']), max(values['T']), 100)
        ax.plot(1/T_fit, -Ea/(8.314*T_fit), '--', color=color, alpha=0.7)
        
        # Create legend entry with Ea in eV
        Ea_eV = Ea * 0.010364  # Convert kJ/mol to eV
        Ea_err_eV = perr[1] * 0.010364
        legend_text = fr'{conc}\% Na$_2$O: $E_a$={Ea_eV:.2f} eV, $D_0$={D0:.1e} cm²/s'
        legend_ax.plot([], [], 'o', color=color, label=legend_text)
        
    except Exception as e:
        print(f"Error processing {conc}% Na2O: {str(e)}")
        continue

# Format plot axes
ax.invert_xaxis()
ax.set_xlabel(r'Inverse Temperature (1/T) [K$^{-1}$]')
ax.set_ylabel(r'$\ln(D/D_0)$')
ax.grid(True, alpha=0.3, linestyle=':')

# Create legend in bottom panel
legend_ax.legend(loc='center', ncol=2, frameon=True, framealpha=0.9)

# Generate and print parameter table (still in kJ/mol)
if fitted_params:
    table_data = []
    for conc in sorted(fitted_params.keys()):
        params = fitted_params[conc]
        # Convert to eV for the table output
        Ea_eV = params['E0'] * kJmol_to_eV
        Ea_err_eV = params['E0_err'] * kJmol_to_eV
        table_data.append([
            f"{conc}",
            f"{params['D0']:.3e} ± {params['D0_err']:.1e}",
            f"{Ea_eV:.3f} ± {Ea_err_eV:.3f}",
            f"{params['R2']:.4f}"
        ])

    print("\nFitted Arrhenius Parameters with Uncertainties:")
    print(tabulate(table_data, 
                 headers=["Na$_2$O (%)", "D$_0$ (cm²/s)", "E$_a$ (eV)", "R²"],
                 tablefmt="grid", stralign="center"))
else:
    print("Warning: No successful fits were completed")

plt.savefig('Normalized_arrhenius_plot_eV.png', bbox_inches='tight', dpi=300)
plt.show()
