import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Enable LaTeX rendering in Matplotlib and load the amsmath package
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Increase font sizes globally
sns.set(font_scale=1.5)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Convert lists to NumPy arrays
Na2O_percent = np.array([0, 5, 10, 15, 20, 25, 30, 35])
C11_sim = np.array([74.68, 69.92, 67.13, 66.25, 67.19, 70.12, 74.22, 77.91])
C44_sim = np.array([26.93, 26.43, 24.80, 24.22, 23.50, 23.81, 24.37, 24.68])
C12_sim = np.array([21.10, 17.56, 17.33, 18.24, 20.26, 23.23, 26.40, 29.94])

# Experimental data (from table at ~300K)
Na2O_C11_exp = np.array([0, 6.3, 8.3, 9.5, 10, 12, 13.7, 18.5, 22.8, 36.5, 40])
C11_exp = np.array([79.5, 73.78, 73.31, 73.54, 72.15, 71.93, 71.42, 70.74, 70.7, 74.35, 76.34])
C44_exp = np.array([32.48, 29.82, 29.69, 29.04, 28.25, 28.01, 27.62, 26.65, 25.45, 24.64, 24.74])

# Compute mechanical properties
def compute_E(C11, C44):
    return C44 * (3 * C11 - 4 * C44) / (C11 - C44)

def compute_B(C11, C12):
    return (C11 + 2 * C12) / 3

def compute_nu(C11, C12, C44):
    return C12 / (C11 + C12)

# Simulation properties
E_sim = compute_E(C11_sim, C44_sim)
B_sim = compute_B(C11_sim, C12_sim)
nu_sim = compute_nu(C11_sim, C12_sim, C44_sim)

# Experimental properties
C12_exp = C11_exp - 2 * C44_exp[:len(C11_exp)]
E_exp = compute_E(C11_exp, C44_exp[:len(C11_exp)])
B_exp = compute_B(C11_exp, C12_exp)
nu_exp = compute_nu(C11_exp, C12_exp, C44_exp[:len(C11_exp)])

# Compute percentage differences (absolute values)
def compute_abs_diff(sim, exp, x_sim, x_exp):
    return np.abs((sim - np.interp(x_sim, x_exp, exp))) / np.interp(x_sim, x_exp, exp) * 100

E_diff = compute_abs_diff(E_sim, E_exp, Na2O_percent, Na2O_C11_exp)
B_diff = compute_abs_diff(B_sim, B_exp, Na2O_percent, Na2O_C11_exp)
nu_diff = compute_abs_diff(nu_sim, nu_exp, Na2O_percent, Na2O_C11_exp)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
palette = sns.color_palette("tab10")

def plot_property(ax, x, y_sim, y_exp, x_exp, ylabel, title, yrange):
    ax.plot(x, y_sim, 'o-', color=palette[0], linewidth=2, markersize=8, label='Simulation (300K)')
    ax.plot(x_exp, y_exp, '*-', color='black', linewidth=2, markersize=10, label='Experiment')
    ax.set_xlabel(r"Na$_2$O (\%)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    ax.set_ylim(yrange)
    return ax.lines[0], ax.lines[1]

# Young's Modulus
line_sim_E, line_exp_E = plot_property(axes[0], Na2O_percent, E_sim, E_exp, Na2O_C11_exp, 
                                      r"$E$ (GPa)", "(a) Young's Modulus", (40, 90))

# Bulk Modulus
line_sim_B, line_exp_B = plot_property(axes[1], Na2O_percent, B_sim, B_exp, Na2O_C11_exp,
                                     r"$B$ (GPa)", "(b) Bulk Modulus", (30, 60))

# Poisson's Ratio
line_sim_nu, line_exp_nu = plot_property(axes[2], Na2O_percent, nu_sim, nu_exp, Na2O_C11_exp,
                                        r"$\nu$", "(c) Poisson's Ratio", (0.05, 0.35))

# Differences plot
axes[3].plot(Na2O_percent, E_diff, 'o-', color='blue', label=r'$E$ Difference')
axes[3].plot(Na2O_percent, B_diff, 's-', color='red', label=r'$B$ Difference')
axes[3].plot(Na2O_percent, nu_diff, '^-', color='green', label=r'$\nu$ Difference')
axes[3].set_xlabel(r"Na$_2$O (\%)", fontsize=16)
axes[3].set_ylabel("Absolute Difference (\%)", fontsize=16)
axes[3].set_title("(d) Simulation-Experiment Differences", fontsize=16)
axes[3].grid(True)
axes[3].set_ylim(-1, 50)

# Legend
handles = [line_sim_E, line_exp_E, 
           axes[3].lines[0], axes[3].lines[1], axes[3].lines[2]]
labels = ["Simulation (300K)", "Experiment", 
          r"$E$ Difference", r"$B$ Difference", r"$\nu$ Difference"]
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("mechanical_properties_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
