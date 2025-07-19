import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

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
Na2O_percent = [0, 5, 10, 15, 20, 25, 30, 35]  # Na2O percentages from 0% to 35% in steps of 5%
temperature = [300, 400, 500, 600]  # Temperature range from 300K to 600K
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

# Convert to numpy arrays
C11_sim = np.array(C11_sim)
C44_sim = np.array(C44_sim)

# Interpolate data to finer temperature grid
fine_temps = np.arange(300, 610, 10)  # Temperature range from 300 K to 600 K in steps of 10 K
C11_interp = np.zeros((len(fine_temps), len(Na2O_percent)))
C44_interp = np.zeros((len(fine_temps), len(Na2O_percent)))

for i in range(len(Na2O_percent)):
    C11_interp[:, i] = interp1d(temperature, C11_sim[:, i], kind='linear')(fine_temps)
    C44_interp[:, i] = interp1d(temperature, C44_sim[:, i], kind='linear')(fine_temps)

# Compute derivatives dC11/dT and dC44/dT using central differences
dC11_dT = np.gradient(C11_interp, fine_temps, axis=0)
dC44_dT = np.gradient(C44_interp, fine_temps, axis=0)

# Average derivatives over temperature for each Na2O concentration
dC11_dT_avg = np.mean(dC11_dT, axis=0)
dC44_dT_avg = np.mean(dC44_dT, axis=0)

# Experimental data for dC11/dT and dC44/dT
Na2O_exp = [0, 5.2, 6.3, 6.9, 7.6, 8.5, 9.0, 10.1, 10.7, 12.0, 13.7, 15.7, 18.5, 20.1, 22.8, 26.9, 32.2, 36.5, 39.9]
dC11_dT_exp = [0.01785, 0.01368, 0.01122, 0.00990, 0.01124, 0.00987, 0.00814, 0.00696, 0.00700, 0.00548, 0.00346, 0.00054, -0.00390, -0.00592, -0.00603, -0.01314, -0.01679, -0.01867, -0.02197]
dC44_dT_exp = [0.004733, np.nan, 0.001561, 0.001508, 0.001987, np.nan, 0.001232, 0.000382, np.nan, np.nan, np.nan, np.nan, -0.002899, np.nan, -0.003151, np.nan, np.nan, -0.006012, -0.007728]

# Interpolate/extrapolate missing values in dC44_dT_exp
valid_indices = np.where(~np.isnan(dC44_dT_exp))[0]  # Indices of non-NaN values
interp_func = interp1d(np.array(Na2O_exp)[valid_indices], np.array(dC44_dT_exp)[valid_indices], kind='linear', fill_value="extrapolate")
dC44_dT_exp_interp = interp_func(Na2O_exp)  # Replace NaNs with interpolated/extrapolated values

# Experimental data from the Excel file
data = {
    "6.9% Na2O": {
        "T_C": [23.2, 23.2, 98.5, 146.2, 224.8, 303, 348, 397.6, 423.6, 447],
        "C11": [73.31, 73.31, 74.25, 74.63, 75.36, 76.16, 76.54, 76.43, 77.58, 77.83],
        "C44": [29.46, 29.46, 29.71, 29.13, 29.75, 29.96, 29.68, 29.86, 30.04, 30.22]
    },
    "6.3% Na2O": {
        "T_C": [24, 101, 198.5, 198.5, 275.7, 350.5, 400.3, 425.2, 452.5],
        "C11": [73.78, 74.37, 75.54, 75.54, 76.42, 77.23, 77.92, 78.13, 78.47],
        "C44": [29.82, 30.01, 30.13, 30.13, 29.98, 30.06, 30.36, 30.6, 30.61]
    },
    "12% Na2O": {
        "T_C": [19, 100.8, 198, 300.2, 404.2],
        "C11": [71.93, 72.29, 72.64, 73.45, 74],
        "C44": [28.01, None, 27.99, None, 27.75]
    },
    "10% Na2O": {
        "T_C": [26.1, 99, 175.3, 243.9, 303.7, 392.1, 421.4, 461, 439, 341],
        "C11": [72.15, 72.62, 73.22, 73.57, 74.13, 74.74, 75.19, 75.55, 75.63, 74.97],
        "C44": [28.25, None, 28.32, 28.36, 28.35, None, 28.62, 28.62, 28.82, 28.7]
    },
    "13.7% Na2O": {
        "T_C": [23, 125, 194.6, 336.6, 429.6],
        "C11": [71.42, 71.68, 71.76, 72.52, 73.3],
        "C44": [27.62, None, 27.36, None, 27.74]
    },
    "40% Na2O": {
        "T_C": [23, 121.5, 237.4, 339.2, 434.3],
        "C11": [76.34, 74.17, 71.37, 69.21, 67.41],
        "C44": [24.74, None, None, 22.23, 21.59]
    },
    "36.5% Na2O": {
        "T_C": [19, 19, 101, 179, 256.2, 313.7, 381.2, 401, 423.6, 449.3, 458.8, 249.5],
        "C11": [74.35, 74.94, 72.95, 71.66, 70.26, 69.02, 67.88, 67.84, 67.7, 66.97, 66.62, 71.19],
        "C44": [24.64, 24.43, 23.49, 23.51, 22.89, 23.28, 21.87, 21.71, 21.7, 21.5, 21.22, 22.9]
    },
    "22.8% Na2O": {
        "T_C": [19, 19, 76.5, 98.2, 156, 192.8, 219.8, 297.5, 386.7, 396.5, 412.2, 416, 422, 428, 440, 451.5],
        "C11": [70.7, 70.73, 70.62, 70.21, 69.78, 69.38, 69.38, 68.73, 68.59, 68.64, 68.64, 69, 68.74, 68.73, 68.64, 68.77],
        "C44": [25.45, 25.57, 25.75, 25.72, 25.2, 25.07, 25.24, 24.6, 24.56, 24.51, 24.35, 24.67, 24.54, 24.39, 24.39, 24.62]
    },
    "18.5% Na2O": {
        "T_C": [19.2, 19.2, 54, 98.3, 154, 201, 258, 298.5, 344.2, 376.5, 398, 409.5, 424, 438.5, 453.4, 468],
        "C11": [70.74, 71.06, 70.61, 70.52, 70.33, 70.08, 69.98, 70.23, 70.44, 70.26, 70.36, 70.61, 70.63, 70.74, 70.89, 70.78],
        "C44": [27.03, 26.65, 26.59, 26.48, 26.37, 26.32, 26.09, 26.11, 26.42, 25.81, 25.92, 26.02, 26.04, 26, 26.05, 25.93]
    },
    "9.5% Na2O": {
        "T_C": [20, 47.5, 94, 130, 190, 233, 276, 330, 360.8, 397, 406, 415, 426, 491.5, 497],
        "C11": [73.54, 73.58, 74.16, 74.41, 74.76, 75.28, 75.58, 75.94, 76.14, 76.76, 76.6, 76.78, 76.8, 77.22, 77.37],
        "C44": [29.04, 29.11, 29.31, 29.24, 29.48, 29.54, 29.61, 29.54, 29.35, 29.57, 29.6, 29.7, 29.54, 29.71, 29.8]
    },
    "0% Na2O": {
        "T_C": [19.5, 52, 101.3, 152, 200.3, 225.2, 250, 300.5, 326, 350.2, 400],
        "C11": [79.5, 79.92, 81.03, 81.9, 82.87, 83.49, 83.77, 84.76, 85.09, 85.39, 86.12],
        "C44": [32.48, None, None, None, 33.31, None, None, None, None, None, 34.28]
    }
}

# Convert temperatures from Celsius to Kelvin
for key in data:
    data[key]["T_K"] = [temp + 273.15 for temp in data[key]["T_C"]]

# Sort the data by Na2O concentration (from 0% to higher)
sorted_keys = sorted(data.keys(), key=lambda x: float(x.split("%")[0]))
sorted_data = {key: data[key] for key in sorted_keys}

# Define a color palette for the different temperatures (now 4 colors for 300K-600K)
palette = sns.color_palette("tab10", n_colors=4)

def filter_data(na2o_percent):
    return na2o_percent < 8.0 or na2o_percent > 10.0

def plot_c11_c44_vs_na2o(ax1, ax2):
    for i, temp in enumerate([300, 400, 500, 600]):
        C11_data, C44_data, na2o_percents = [], [], []

        for key, values in sorted_data.items():
            na2o_percent = float(key.split("%")[0])
            
            if filter_data(na2o_percent):
                interp_func = interp1d(values["T_K"], values["C11"], kind='linear', fill_value="extrapolate")
                C11 = interp_func(temp)
                C11_data.append(C11)
                na2o_percents.append(na2o_percent)

                if values["C44"]:
                    valid_indices = [i for i, c44 in enumerate(values["C44"]) if c44 is not None]
                    T_K_C44 = [values["T_K"][i] for i in valid_indices]
                    C44 = [values["C44"][i] for i in valid_indices]

                    if len(T_K_C44) >= 2:
                        interp_func = interp1d(T_K_C44, C44, kind='linear', fill_value="extrapolate")
                        C44_data.append(interp_func(temp))

        # Plot C11 vs Na2O
        ax1.plot(na2o_percents, C11_data, label=f'', color=palette[i], linewidth=2.5, marker="o", markersize=8)
        
        # Plot C44 vs Na2O
        ax2.plot(na2o_percents, C44_data, label=f'', color=palette[i], linewidth=2.5, marker="o", markersize=8)
    
#    ax1.set_ylim(65, 85)  # Adjust range for C11
#    ax2.set_ylim(20, 35)  # Adjust range for C44


# Create a 3x2 grid layout
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
axes = axes.flatten()  # Flatten the 3x2 grid into a 1D array for easier iteration

# Plot simulation results for C11 and C44 vs Na2O
for i, temp in enumerate(temperature):
    axes[0].plot(Na2O_percent, C11_sim[i], label=f'{temp} K', color=palette[i], linewidth=2.5, marker="o", markersize=8)
    axes[1].plot(Na2O_percent, C44_sim[i], color=palette[i], linewidth=2.5, marker="o", markersize=8)

#axes[0].set_ylim(55, 90)  # Adjust range for C11
#axes[1].set_ylim(15, 35)  # Adjust range for C44

axes[0].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$", fontsize=16)
axes[0].set_ylabel(r"$C_{11}$ (GPa)", fontsize=16)
axes[0].text(0.05, 0.95, r"(a)", transform=axes[0].transAxes, fontsize=16, va='top')
axes[0].grid(True)

axes[1].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$", fontsize=16)
axes[1].set_ylabel(r"$C_{44}$ (GPa)", fontsize=16)
axes[1].text(0.2, 0.95, r"(b)", transform=axes[1].transAxes, fontsize=16, va='top')
axes[1].grid(True)


# Plot experimental results for C11 and C44 vs Na2O
plot_c11_c44_vs_na2o(axes[2], axes[3])

axes[2].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$", fontsize=16)
axes[2].set_ylabel(r"$C_{11}$ (GPa)", fontsize=16)
axes[2].text(0.2, 0.95, r"(c)", transform=axes[2].transAxes, fontsize=16, va='top')
axes[2].grid(True)

axes[3].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$", fontsize=16)
axes[3].set_ylabel(r"$C_{44}$ (GPa)", fontsize=16)
axes[3].text(0.2, 0.95, r"(d)", transform=axes[3].transAxes, fontsize=16, va='top')
axes[3].grid(True)

# Plot simulation derivatives dC11/dT and dC44/dT vs Na2O
axes[4].plot(Na2O_percent, dC11_dT_avg, label=r'$\frac{dC_{11}}{dT}$ (Sim)', color='blue', linewidth=2.5, marker="o", markersize=8)
axes[4].plot(Na2O_percent, dC44_dT_avg, label=r'$\frac{dC_{44}}{dT}$ (Sim)', color='red', linewidth=2.5, marker="s", markersize=8)

# Plot experimental derivatives dC11/dT and dC44/dT vs Na2O
axes[4].plot(Na2O_exp, dC11_dT_exp, label=r'$\frac{dC_{11}}{dT}$ (Exp)', color='blue', linestyle='--', linewidth=2.5, marker="^", markersize=8)
axes[4].plot(Na2O_exp, dC44_dT_exp_interp, label=r'$\frac{dC_{44}}{dT}$ (Exp)', color='red', linestyle='--', linewidth=2.5, marker="v", markersize=8)

#axes[4].set_ylim(-0.025, 0.025)  # Adjust range for derivatives


axes[4].set_xlabel(r"$\text{Na}_2\text{O}~(\%)$", fontsize=16)
axes[4].set_ylabel(r"$\frac{dC}{dT}$ (GPa/K)", fontsize=16)
axes[4].text(0.2, 0.95, r"(e)", transform=axes[4].transAxes, fontsize=16, va='top')
axes[4].grid(True)

# Collect all legend handles and labels
handles_a, labels_a = axes[0].get_legend_handles_labels()  # Handles and labels for C11 simulation
handles_c, labels_c = axes[2].get_legend_handles_labels()  # Handles and labels for C11 experimental
handles_e, labels_e = axes[4].get_legend_handles_labels()  # Handles and labels for derivatives

# Combine all handles and labels
all_handles = handles_a + handles_c + handles_e
all_labels = labels_a + labels_c + labels_e

# Create a dummy subplot in the sixth position for the legend
axes[5].axis('off')  # Turn off the axis for the sixth subplot
axes[5].legend(all_handles, all_labels, title="", loc='center', fontsize=14)  # Place the legend in the sixth subplot

# Adjust layout
plt.tight_layout()

# Save the plot as an image
plt.savefig("Exp_Sim_C11_C44_and_derivatives_vs_Na2O.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
