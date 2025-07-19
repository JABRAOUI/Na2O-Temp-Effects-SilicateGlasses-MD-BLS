import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Configure LaTeX rendering and fonts to match the second code
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,siunitx}'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 18
sns.set_style("whitegrid")
sns.set_palette("deep")

# Simulation data
Na2O_percent = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] * 4
temperature = [300] * 11 + [400] * 11 + [500] * 11 + [600] * 11
density = [
    2.2, 2.26535113, 2.315809397, 2.368980141, 2.422525175, 2.46865288,
    2.522157348, 2.558586574, 2.588381225, 2.593071909, 2.60231169,
    2.196321408, 2.262397621, 2.307285586, 2.355017488, 2.407773754, 2.461152806,
    2.500700194, 2.537613299, 2.56720841, 2.57242494, 2.578717969,
    2.199411701, 2.257502136, 2.298162798, 2.341960231, 2.395447879, 2.440546752,
    2.490719877, 2.524597955, 2.54637273, 2.552643082, 2.556246482,
    2.202110679, 2.258484276, 2.294258321, 2.338158389, 2.376834053, 2.425981473,
    2.475447064, 2.500616656, 2.517742854, 2.533236497, 2.534397419
]

# Experimental density data (room temperature)
Na2O_density_exp = np.array([0, 5, 6, 7, 9, 12, 16, 20, 24, 29, 33])
density_exp = np.array([2.203, 2.2247, 2.258, 2.271, 2.274, 2.300, 2.346, 2.393, 2.428, 2.468, 2.494])

# Create DataFrames
df_simulated = pd.DataFrame({
    "Percentage_Na2O": Na2O_percent,
    "Temperature": temperature,
    "Density": density
})

# Filter to include only Na₂O percentages from 0% to 35%
df_simulated = df_simulated[df_simulated["Percentage_Na2O"] <= 35]

df_experimental = pd.DataFrame({
    "Percentage_Na2O": Na2O_density_exp,
    "Density": density_exp
})

# Create figure with layout matching the second code
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(r'Density Comparison', fontsize=18, y=1.02)

# Plot simulated data with styling from second code
colors = sns.color_palette("deep")
for i, temp in enumerate(sorted(df_simulated["Temperature"].unique())):
    temp_data = df_simulated[df_simulated["Temperature"] == temp]
    ax.plot(temp_data["Percentage_Na2O"], temp_data["Density"], 
            label=f"Simulated {temp} K", color=colors[i], 
            linewidth=2.5, linestyle=['-', '--', ':', '-.'][i % 4])

# Plot experimental data with distinct style
ax.plot(df_experimental["Percentage_Na2O"], df_experimental["Density"], 
        label="Experimental (Room Temp)", color='k', 
        linewidth=2.5, linestyle='-', marker='o', markersize=8)

# Axis labels matching second code style
ax.set_xlabel(r'Na$_2$O (\%)', fontsize=16)
ax.set_ylabel(r'Density (g/cm$^3$)', fontsize=16)

# Grid and limits
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_ylim(2.15, 2.65)

# Legend matching second code style
legend = ax.legend(title="Temperature", frameon=True, framealpha=0.9, 
                  shadow=True, bbox_to_anchor=(1.02, 1), loc='upper left')
legend.get_title().set_fontsize('14')

plt.tight_layout()
plt.savefig("density_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
