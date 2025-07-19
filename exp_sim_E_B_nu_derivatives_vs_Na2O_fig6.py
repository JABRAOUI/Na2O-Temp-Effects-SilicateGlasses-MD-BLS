import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# Enable LaTeX rendering in Matplotlib (requires a LaTeX installation)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Load amsmath for LaTeX equations

# Increase font sizes globally for readability
sns.set(font_scale=1.5)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Simulation data: Na2O percentage (mol %) and temperatures (K)
Na2O_percent = [0, 5, 10, 15, 20, 25, 30, 35]
temperature = [300, 400, 500, 600]

# C11 and C44 simulation data (GPa)
C11_sim = [
    [74.68, 69.92, 67.13, 66.25, 67.19, 70.12, 74.22, 77.91],  # 300 K
    [74.76, 69.51, 66.08, 64.31, 65.56, 67.15, 70.57, 74.11],  # 400 K
    [74.84, 69.02, 65.65, 62.95, 63.06, 64.08, 66.73, 69.39],  # 500 K
    [74.99, 68.43, 64.76, 61.73, 60.91, 61.57, 62.86, 64.83]   # 600 K
]
C44_sim = [
    [26.93, 26.43, 24.80, 24.22, 23.50, 23.81, 24.37, 24.68],  # 300 K
    [26.87, 26.25, 24.50, 23.49, 23.00, 23.24, 23.58, 23.91],  # 400 K
    [26.91, 25.92, 24.10, 23.02, 22.42, 22.30, 22.65, 22.55],  # 500 K
    [26.94, 25.66, 23.79, 22.70, 21.66, 21.59, 21.24, 21.47]   # 600 K
]
C11_sim = np.array(C11_sim)
C44_sim = np.array(C44_sim)

# Elastic property formulas (cubic, isotropic assumptions)
def compute_C12(C11, C44):
    # Approximate C12 from C11 and C44 when direct data unavailable
    return C11 - 2 * C44

def compute_E(C11, C44):
    # Young's modulus E (GPa)
    return C44 * (3 * C11 - 4 * C44) / (C11 - C44)

def compute_B(C11, C44):
    # Bulk modulus B (GPa) 
    return C11 -  4 * C44/3

def compute_nu(C11, C44):
    # Poisson's ratio nu (dimensionless)
    return (C11 - 2 * C44) / (2 * (C11 - C44))

E_sim = compute_E(C11_sim, C44_sim)
B_sim = compute_B(C11_sim, C44_sim)
nu_sim = compute_nu(C11_sim, C44_sim)

# Interpolate simulation data for finer temperature grid (300K to 600K, step 10K)
fine_temps = np.arange(300, 610, 10)
E_interp = np.zeros((len(fine_temps), len(Na2O_percent)))
B_interp = np.zeros((len(fine_temps), len(Na2O_percent)))
nu_interp = np.zeros((len(fine_temps), len(Na2O_percent)))

for i in range(len(Na2O_percent)):
    # Linear interpolation to fill finer temperature steps
    E_interp[:, i] = interp1d(temperature, E_sim[:, i], kind='linear')(fine_temps)
    B_interp[:, i] = interp1d(temperature, B_sim[:, i], kind='linear')(fine_temps)
    nu_interp[:, i] = interp1d(temperature, nu_sim[:, i], kind='linear')(fine_temps)

# Compute temperature derivatives (central differences)
dE_dT = np.gradient(E_interp, fine_temps, axis=0)
dB_dT = np.gradient(B_interp, fine_temps, axis=0)
dnu_dT = np.gradient(nu_interp, fine_temps, axis=0)

# Average derivatives over temperature for each Na2O concentration
dE_dT_avg = np.mean(dE_dT, axis=0)
dB_dT_avg = np.mean(dB_dT, axis=0)
dnu_dT_avg = np.mean(dnu_dT, axis=0)

# Experimental derivatives dC11/dT and dC44/dT (with NaNs)
Na2O_exp = [0, 5.2, 6.3, 6.9, 7.6, 8.5, 9.0, 10.1, 10.7, 12.0, 13.7, 15.7, 18.5, 20.1, 22.8, 26.9, 32.2, 36.5, 39.9]
dC11_dT_exp = [0.01785, 0.01368, 0.01122, 0.00990, 0.01124, 0.00987, 0.00814, 0.00696, 0.00700, 0.00548, 0.00346,
               0.00054, -0.00390, -0.00592, -0.00603, -0.01314, -0.01679, -0.01867, -0.02197]
dC44_dT_exp = [0.004733, np.nan, 0.001561, 0.001508, 0.001987, np.nan, 0.001232, 0.000382, np.nan, np.nan, np.nan, np.nan,
               -0.002899, np.nan, -0.003151, np.nan, np.nan, -0.006012, -0.007728]

# Fill missing dC44/dT values by linear interpolation (including extrapolation)
valid_indices = np.where(~np.isnan(dC44_dT_exp))[0]
interp_func = interp1d(np.array(Na2O_exp)[valid_indices], np.array(dC44_dT_exp)[valid_indices],
                       kind='linear', fill_value='extrapolate')
dC44_dT_exp_interp = interp_func(Na2O_exp)

# Experimental data from the Excel file (updated with table values at corresponding temperatures)
data = {
    "0% Na2O": {
        "T_C": [19.5, 52, 101.3, 152, 200.3, 225.2, 250, 300.5, 326, 350.2, 400],
        "C11": [79.5, 79.92, 81.03, 81.9, 82.87, 83.49, 83.77, 84.76, 85.09, 85.39, 86.12],
        "C44": [32.48, np.nan, np.nan, np.nan, 33.31, np.nan, np.nan, np.nan, np.nan, np.nan, 34.28]
    },
    "6.3% Na2O": {
        "T_C": [24, 101, 198.5, 198.5, 275.7, 350.5, 400.3, 425.2, 452.5],
        "C11": [73.78, 74.37, 75.54, 75.54, 76.42, 77.23, 77.92, 78.13, 78.47],
        "C44": [29.82, 30.01, 30.13, 30.13, 29.98, 30.06, 30.36, 30.6, 30.61]
    },
    "8.3% Na2O": {
        "T_C": [23.2, 23.2, 98.5, 146.2, 224.8, 303, 348, 397.6, 423.6, 447],
        "C11": [73.31, 73.31, 74.25, 74.63, 75.36, 76.16, 76.54, 76.43, 77.58, 77.83],
        "C44": [29.46, 29.46, 29.71, 29.13, 29.75, 29.96, 29.68, 29.86, 30.04, 30.22]
    },
    "10% Na2O": {
        "T_C": [26.1, 99, 175.3, 243.9, 303.7, 392.1, 421.4, 461, 439, 341],
        "C11": [72.15, 72.62, 73.22, 73.57, 74.13, 74.74, 75.19, 75.55, 75.63, 74.97],
        "C44": [28.25, np.nan, 28.32, 28.36, 28.35, np.nan, 28.62, 28.62, 28.82, 28.7]
    },
    "12% Na2O": {
        "T_C": [19, 100.8, 198, 300.2, 404.2],
        "C11": [71.93, 72.29, 72.64, 73.45, 74.0],
        "C44": [28.01, np.nan, 27.99, np.nan, 27.75]
    },
    "13.7% Na2O": {
        "T_C": [23, 125, 194.6, 336.6, 429.6],
        "C11": [71.42, 71.68, 71.76, 72.52, 73.3],
        "C44": [27.62, np.nan, 27.36, np.nan, 27.74]
    },
    "18.5% Na2O": {
        "T_C": [19.2, 19.2, 54, 98.3, 154, 201, 258, 298.5, 344.2, 376.5, 398, 409.5, 424, 438.5, 453.4, 468],
        "C11": [70.74, 71.06, 70.61, 70.52, 70.33, 70.08, 69.98, 70.23, 70.44, 70.26, 70.36, 70.61, 70.63, 70.74, 70.89, 70.78],
        "C44": [27.03, 26.65, 26.59, 26.48, 26.37, 26.32, 26.09, 26.11, 26.42, 25.81, 25.92, 26.02, 26.04, 26.0, 26.05, 25.93]
    },
    "22.8% Na2O": {
        "T_C": [19, 19, 76.5, 98.2, 156, 192.8, 219.8, 297.5, 386.7, 396.5, 412.2, 416, 422, 428, 440, 451.5],
        "C11": [70.7, 70.73, 70.62, 70.21, 69.78, 69.38, 69.38, 68.73, 68.59, 68.64, 68.64, 69.0, 68.74, 68.73, 68.64, 68.77],
        "C44": [25.45, 25.57, 25.75, 25.72, 25.2, 25.07, 25.24, 24.6, 24.56, 24.51, 24.35, 24.67, 24.54, 24.39, 24.39, 24.62]
    },
    "36.5% Na2O": {
        "T_C": [19, 19, 101, 179, 256.2, 313.7, 381.2, 401, 423.6, 449.3, 458.8, 249.5],
        "C11": [74.35, 74.94, 72.95, 71.66, 70.26, 69.02, 67.88, 67.84, 67.7, 66.97, 66.62, 71.19],
        "C44": [24.64, 24.43, 23.49, 23.51, 22.89, 23.28, 21.87, 21.71, 21.7, 21.5, 21.22, 22.9]
    },
    "40% Na2O": {
        "T_C": [23, 121.5, 237.4, 339.2, 434.3],
        "C11": [76.34, 74.17, 71.37, 69.21, 67.41],
        "C44": [24.74, np.nan, np.nan, 22.23, 21.59]
    }
}

# Convert temperatures from Celsius to Kelvin
for key in data:
    data[key]["T_K"] = [temp + 273.15 for temp in data[key]["T_C"]]

# Sort datasets by Na2O content ascending
sorted_keys = sorted(data.keys(), key=lambda x: float(x.split("%")[0]))
sorted_data = {key: data[key] for key in sorted_keys}

# Color palette for plotting different temperatures
palette = sns.color_palette("tab10", n_colors=4)

# Helper function to interpolate missing data and ensure connectivity at zero Na2O
def interpolate_and_fill(x_vals, y_vals, x_fine):
    y_vals = np.array(y_vals, dtype=np.float64)
    nan_mask = np.isnan(y_vals)
    # Ensure value at zero Na2O exists by copying nearest available or interpolating
    if x_vals[0] == 0 and np.isnan(y_vals[0]):
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 0:
            y_vals[0] = y_vals[valid_indices[0]]
    # Interpolate missing values linearly with extrapolation
    interp_func = interp1d(np.array(x_vals)[~nan_mask], y_vals[~nan_mask], kind='linear', fill_value='extrapolate')
    y_vals[nan_mask] = interp_func(np.array(x_vals)[nan_mask])
    # Interpolate to fine grid
    return interp_func(x_fine)

# Filter function used in experimental plotting sections
def filter_data(na2o_percent):
    return na2o_percent < 8.0 or na2o_percent > 10.0

# Compute Poisson's ratio for experimental data where possible
for key in sorted_data:
    C11_exp = sorted_data[key]["C11"]
    C44_exp = sorted_data[key]["C44"]
    nu_exp = []
    for i in range(len(C11_exp)):
        if C44_exp[i] is not None and C11_exp[i] is not None:
            nu = (C11_exp[i] - 2 * C44_exp[i]) / (2 * (C11_exp[i] - C44_exp[i]))
            nu_exp.append(nu)
        else:
            nu_exp.append(None)
    sorted_data[key]["nu"] = nu_exp

# Prepare figure with a 3x3 grid (last cell for combined legend)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

# Plot simulation data (E, B, nu)
for i, temp in enumerate(temperature):
    axes[0].plot(Na2O_percent, E_sim[i], label=f'{temp} K', color=palette[i], linewidth=2.5, marker="o", markersize=8)
    axes[1].plot(Na2O_percent, B_sim[i], color=palette[i], linewidth=2.5, marker="o", markersize=8)
    axes[2].plot(Na2O_percent, nu_sim[i], color=palette[i], linewidth=2.5, marker="o", markersize=8)

axes[0].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$")
axes[0].set_ylabel(r"$E$ (GPa)")
axes[0].text(0.2, 0.95, r"(a: Sim)", transform=axes[0].transAxes, fontsize=16, va='top')
axes[0].grid(True)

axes[1].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$")
axes[1].set_ylabel(r"$B$ (GPa)")
axes[1].text(0.2, 0.95, r"(b: Sim)", transform=axes[1].transAxes, fontsize=16, va='top')
axes[1].grid(True)

axes[2].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$")
axes[2].set_ylabel(r"$\nu$")
axes[2].text(0.2, 0.95, r"(c: Sim)", transform=axes[2].transAxes, fontsize=16, va='top')
axes[2].grid(True)

# Plot experimental data interpolated for specific temperatures
target_temps = [300, 400, 500, 600]

for i, temp in enumerate(target_temps):
    E_data, B_data, nu_data, na2o_percents = [], [], [], []
    for key, values in sorted_data.items():
        na2o_percent = float(key.split("%")[0])
        if filter_data(na2o_percent):
            # Interpolate C11 with missing values handled
            C11_interp = interpolate_and_fill(values["T_K"], values["C11"], [temp])[0]
            # Interpolate C44 similarly, only if available and enough points
            if values["C44"]:
                valid_indices = [idx for idx, val in enumerate(values["C44"]) if val is not None]
                if len(valid_indices) >= 2:
                    T_K_C44 = [values["T_K"][idx] for idx in valid_indices]
                    C44_vals = [values["C44"][idx] for idx in valid_indices]
                    C44_interp = interpolate_and_fill(T_K_C44, C44_vals, [temp])[0]
                    # Calculate properties with isotropic approximations (state assumptions)
                    E_val = compute_E(C11_interp, C44_interp)
                    B_val = compute_B(C11_interp, C44_interp)
                    nu_val = compute_nu(C11_interp, C44_interp)
                    E_data.append(E_val)
                    B_data.append(B_val)
                    nu_data.append(nu_val)
                    na2o_percents.append(na2o_percent)
    axes[3].plot(na2o_percents, E_data, color=palette[i], linewidth=2.5, marker="o", markersize=8)
    axes[4].plot(na2o_percents, B_data, color=palette[i], linewidth=2.5, marker="o", markersize=8)
    axes[5].plot(na2o_percents, nu_data, color=palette[i], linewidth=2.5, marker="o", markersize=8)

axes[3].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$")
axes[3].set_ylabel(r"$E$ (GPa)")
axes[3].text(0.2, 0.95, r"(d: Exp)", transform=axes[3].transAxes, fontsize=16, va='top')
axes[3].grid(True)

axes[4].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$")
axes[4].set_ylabel(r"$B$ (GPa)")
axes[4].text(0.2, 0.95, r"(e: Exp)", transform=axes[4].transAxes, fontsize=16, va='top')
axes[4].grid(True)

axes[5].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$")
axes[5].set_ylabel(r"$\nu$")
axes[5].text(0.2, 0.95, r"(f: Exp)", transform=axes[5].transAxes, fontsize=16, va='top')
axes[5].grid(True)


# Plot simulation derivatives dE/dT, dB/dT, dnu/dT vs Na2O%
axes[6].plot(Na2O_percent, dE_dT_avg, label=r'$\frac{dE}{dT}$', color='blue', linewidth=2.5, marker="o", markersize=8)
axes[6].plot(Na2O_percent, dB_dT_avg, label=r'$\frac{dB}{dT}$', color='red', linewidth=2.5, marker="s", markersize=8)
axes[6].plot(Na2O_percent, dnu_dT_avg, label=r'$\frac{d\nu}{dT}$', color='green', linewidth=2.5, marker="^", markersize=8)
axes[6].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$")
axes[6].set_ylabel(r"$\frac{d}{dT}$ (GPa/K)")
axes[6].text(0.2, 0.65, r"(g: Sim)", transform=axes[6].transAxes, fontsize=16, va='top')
axes[6].grid(True)

# Calculate experimental derivatives of E, B, and nu from C11 and C44 derivatives
# Using chain rule on the formulas for E, B, and nu
def compute_dE_dT(dC11_dT, dC44_dT, C11, C44):
    numerator = (3*C11 - 4*C44)*dC44_dT + C44*(3*dC11_dT - 4*dC44_dT)
    denominator = C11 - C44
    return (numerator*(C11 - C44) - C44*(3*C11 - 4*C44)*(dC11_dT - dC44_dT)) / (denominator**2)

def compute_dB_dT(dC11_dT, dC44_dT):
    return dC11_dT - (4/3)*dC44_dT

def compute_dnu_dT(dC11_dT, dC44_dT, C11, C44):
    numerator = (C11 - 2*C44)*(dC11_dT - dC44_dT) - (dC11_dT - 2*dC44_dT)*(C11 - C44)
    denominator = 2*(C11 - C44)**2
    return numerator / denominator

# Calculate experimental C11 and C44 values at reference temperature (300K)
# We'll use the simulation data at 300K as reference for the experimental derivatives
C11_ref = C11_sim[0]  # 300K simulation data
C44_ref = C44_sim[0]  # 300K simulation data

# Interpolate experimental C11 and C44 derivatives to match Na2O_percent grid
interp_dC11_dT = interp1d(Na2O_exp, dC11_dT_exp, kind='linear', fill_value='extrapolate')
interp_dC44_dT = interp1d(Na2O_exp, dC44_dT_exp_interp, kind='linear', fill_value='extrapolate')

dC11_dT_exp_interp = interp_dC11_dT(Na2O_percent)
dC44_dT_exp_interp = interp_dC44_dT(Na2O_percent)

# Calculate experimental derivatives of E, B, and nu
dE_dT_exp = compute_dE_dT(dC11_dT_exp_interp, dC44_dT_exp_interp, C11_ref, C44_ref)
dB_dT_exp = compute_dB_dT(dC11_dT_exp_interp, dC44_dT_exp_interp)
dnu_dT_exp = compute_dnu_dT(dC11_dT_exp_interp, dC44_dT_exp_interp, C11_ref, C44_ref)

# Plot experimental derivatives dE/dT, dB/dT, dnu/dT vs Na2O%
axes[7].plot(Na2O_percent, dE_dT_exp, label=r'$\frac{dE}{dT}$ (Exp)', color='blue', linestyle='--', linewidth=2.5, marker="o", markersize=8)
axes[7].plot(Na2O_percent, dB_dT_exp, label=r'$\frac{dB}{dT}$ (Exp)', color='red', linestyle='--', linewidth=2.5, marker="s", markersize=8)
axes[7].plot(Na2O_percent, dnu_dT_exp, label=r'$\frac{d\nu}{dT}$ (Exp)', color='green', linestyle='--', linewidth=2.5, marker="^", markersize=8)
axes[7].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$")
axes[7].set_ylabel(r"$\frac{d}{dT}$ (GPa/K)")
axes[7].text(0.2, 0.95, r"(h: Exp)", transform=axes[7].transAxes, fontsize=16, va='top')
axes[7].grid(True)

# Legend subplot (9th cell)
axes[8].axis('off')
handles_sim, labels_sim = axes[0].get_legend_handles_labels()
handles_deriv, labels_deriv = axes[6].get_legend_handles_labels()
handles_expderiv, labels_expderiv = axes[7].get_legend_handles_labels()
all_handles = handles_sim + handles_deriv + handles_expderiv
all_labels = labels_sim + labels_deriv + labels_expderiv
axes[8].legend(all_handles, all_labels, loc='center', fontsize=14)

plt.tight_layout()
plt.savefig("Exp_Sim_E_B_nu_and_derivatives_vs_Na2O.png", dpi=300, bbox_inches="tight")
plt.show()
