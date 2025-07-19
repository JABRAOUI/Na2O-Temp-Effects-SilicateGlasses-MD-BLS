import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Enable LaTeX rendering in Matplotlib
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, chemformula}'

# Increase font sizes globally
sns.set(font_scale=1.3)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Create figure with 3x3 grid
fig = plt.figure(figsize=(10, 15))

# Create a 3x3 grid
gs = fig.add_gridspec(3, 3, width_ratios=[3, 1, 0.1], wspace=0.05)

# --- Qi Species Data ---
ax1 = fig.add_subplot(gs[0, 0])
ax1_leg = fig.add_subplot(gs[0, 1])  # For legend only

qi_data = {
    "Percentage_Na2O": [0, 5, 10, 15, 20, 25, 30, 35],
    "Q0": [0, 0, 0, 0, 0, 0.041666667, 0.133928571, 0.288461538],
    "Q1": [0, 0, 0, 0.220588235, 0.1953125, 0.916666667, 2.455357143, 4.182692308],
    "Q2": [0, 0.263157895, 1.25, 2.941176471, 6.5625, 10.83333333, 16.47321429, 22.35576923],
    "Q3": [1.595744681, 10.19736842, 19.82638889, 28.78676471, 35.7421875, 41.5, 43.83928571, 46.10576923],
    "Q4": [93.99249061, 88.84868421, 78.61111111, 67.86764706, 57.421875, 46.54166667, 37.05357143, 26.875],
    "Q5": [4.349186483, 0.657894737, 0.3125, 0.183823529, 0.078125, 0.166666667, 0.044642857, 0.192307692]
}

qi_df = pd.DataFrame(qi_data)
qi_melted = qi_df.melt(id_vars=["Percentage_Na2O"], 
                       value_vars=["Q0", "Q1", "Q2", "Q3", "Q4", "Q5"],
                       var_name="Qi_species", value_name="Percentage")

qi_melted["Qi_species"] = qi_melted["Qi_species"].replace({
    "Q0": r"$Q_0$", "Q1": r"$Q_1$", "Q2": r"$Q_2$",
    "Q3": r"$Q_3$", "Q4": r"$Q_4$", "Q5": r"$Q_5$"
})

palette = sns.color_palette("tab10", n_colors=6)
lines = []
labels = []
for qi, qi_data in qi_melted.groupby("Qi_species"):
    line = sns.lineplot(data=qi_data, x="Percentage_Na2O", y="Percentage", 
                        color=palette[qi_melted["Qi_species"].unique().tolist().index(qi)], 
                        linewidth=2.5, marker="o", markersize=8, ax=ax1)
    lines.extend(line.get_lines())
    labels.append(qi)

# Place title inside the panel at the top
ax1.text(0.5, 0.95, '(a) Qi Species Distribution', transform=ax1.transAxes,
         fontsize=16, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
ax1.set_xlabel(r'$\ch{Na2O}~(\%)$')
ax1.set_ylabel(r'Percentage (\%)')
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3)

# Create legend in the right panel
ax1_leg.legend(lines, labels, title=r'$Q_i$ Species', loc='center', framealpha=1)
ax1_leg.axis('off')

# --- Oxygen Types Data ---
ax2 = fig.add_subplot(gs[1, 0])
ax2_leg = fig.add_subplot(gs[1, 1])  # For legend only

oxy_data = {
    "Percentage_Na2O": [0, 5, 10, 15, 20, 25, 30, 35],
    "BO": [100.00, 94.76, 89.46, 83.77, 77.88, 71.54, 64.93, 58.11],
    "NBO": [0.00, 5.24, 10.51, 16.22, 21.99, 28.34, 34.87, 41.31],
    "FO": [0.00, 0.00, 0.03, 0.02, 0.12, 0.13, 0.20, 0.59]
}

oxy_df = pd.DataFrame(oxy_data)
oxy_melted = oxy_df.melt(id_vars=["Percentage_Na2O"], 
                        value_vars=["BO", "NBO", "FO"],
                        var_name="Oxygen_Type", value_name="Percentage")

oxy_melted["Oxygen_Type"] = oxy_melted["Oxygen_Type"].replace({
    "BO": r"$\text{BO}$", "NBO": r"$\text{NBO}$", "FO": r"$\text{FO}$"
})

palette = sns.color_palette("tab10", n_colors=3)
lines2 = []
labels2 = []
for oxy, oxy_data in oxy_melted.groupby("Oxygen_Type"):
    line = sns.lineplot(data=oxy_data, x="Percentage_Na2O", y="Percentage", 
                        color=palette[oxy_melted["Oxygen_Type"].unique().tolist().index(oxy)], 
                        linewidth=2.5, marker="o", markersize=8, ax=ax2)
    lines2.extend(line.get_lines())
    labels2.append(oxy)

# Place title inside the panel at the top
ax2.text(0.5, 0.95, '(b) Oxygen Type Distribution', transform=ax2.transAxes,
         fontsize=16, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
ax2.set_xlabel(r'$\ch{Na2O}~(\%)$')
ax2.set_ylabel(r'Percentage of O (\%)')
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.3)

# Create legend in the right panel
ax2_leg.legend(lines2, labels2, title='Oxygen Types', loc='center', framealpha=1)
ax2_leg.axis('off')

# --- Na Coordination Number Distribution Data ---
ax3 = fig.add_subplot(gs[2, 0])
ax3_leg = fig.add_subplot(gs[2, 1])  # For legend only

na_cn_data = {
    "Percentage_Na2O": [5, 10, 15, 20, 25, 30, 35],
    "CN3": [6.875, 5.15625, 4.6875, 4.140625, 2.5625, 1.979166667, 1.116071429],
    "CN4": [28.4375, 27.5, 26.04166667, 21.015625, 19.125, 15.57291667, 12.36607143],
    "CN5": [39.375, 38.90625, 35.9375, 38.984375, 38.1875, 33.07291667, 32.1875],
    "CN6": [20.9375, 23.75, 25.3125, 25.859375, 26.9375, 31.30208333, 30.58035714],
    "CN7": [3.4375, 4.375, 6.666666667, 8.671875, 10.75, 14.0625, 16.91964286],
    "CN8": [0.3125, 0.3125, 1.354166667, 1.09375, 2.375, 3.4375, 5.625],
    "CN9": [0, 0, 0, 0.078125, 0.0625, 0.572916667, 1.026785714]
}

na_cn_df = pd.DataFrame(na_cn_data)
na_cn_melted = na_cn_df.melt(id_vars=["Percentage_Na2O"], 
                            value_vars=["CN3", "CN4", "CN5", "CN6", "CN7", "CN8", "CN9"],
                            var_name="Coordination", value_name="Percentage")

na_cn_melted["Coordination"] = na_cn_melted["Coordination"].replace({
    "CN3": r"$\text{CN=3}$", "CN4": r"$\text{CN=4}$", "CN5": r"$\text{CN=5}$",
    "CN6": r"$\text{CN=6}$", "CN7": r"$\text{CN=7}$", "CN8": r"$\text{CN=8}$",
    "CN9": r"$\text{CN=9}$"
})

palette = sns.color_palette("viridis", n_colors=7)
lines3 = []
labels3 = []
for cn, cn_data in na_cn_melted.groupby("Coordination"):
    line = sns.lineplot(data=cn_data, x="Percentage_Na2O", y="Percentage", 
                        color=palette[na_cn_melted["Coordination"].unique().tolist().index(cn)], 
                        linewidth=2.5, marker="o", markersize=8, ax=ax3)
    lines3.extend(line.get_lines())
    labels3.append(cn)

# Place title inside the panel at the top
ax3.text(0.5, 0.95, '(c) Na Coordination Number Distribution', transform=ax3.transAxes,
         fontsize=16, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
ax3.set_xlabel(r'$\ch{Na2O}~(\%)$')
ax3.set_ylabel(r'Percentage of Na (\%)')
ax3.set_ylim(0, 45)
ax3.grid(True, alpha=0.3)

# Create legend in the right panel
ax3_leg.legend(lines3, labels3, title='Na Coordination', loc='center', framealpha=1)
ax3_leg.axis('off')

# Adjust layout with more padding between subplots
plt.tight_layout(h_pad=3.0)

# Save the plot
plt.savefig("structural_properties_Qi_Oxygens_CN.png", dpi=300, bbox_inches="tight")
plt.show()
