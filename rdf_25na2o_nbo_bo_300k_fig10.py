import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

# Configure LaTeX rendering and fonts
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

# Custom rounding function with proper decimal handling
def custom_round(value, decimals=2):
    multiplier = 10 ** decimals
    return math.floor(value * multiplier + 0.5) / multiplier

def read_data_from_file(file_path):
    """Improved data reader with error handling and validation"""
    distances = []
    si_x = []
    na_x = []
    x_x = []

    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) != 7:
                        print(f"Warning: Line {line_num} in {file_path} has {len(parts)} columns (expected 7)")
                        continue
                    try:
                        distances.append(float(parts[0]))
                        si_x.append(float(parts[2]))
                        na_x.append(float(parts[5]))
                        x_x.append(float(parts[4]))
                    except ValueError as e:
                        print(f"Error parsing line {line_num} in {file_path}: {e}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None, None, None, None
    
    return distances, si_x, na_x, x_x

def find_first_peak(distances, rdf, peak_range=(0, 10)):
    """Enhanced peak finder with smoothing and validation"""
    distances = np.array(distances)
    rdf = np.array(rdf)
    
    if len(distances) != len(rdf):
        print("Error: distances and RDF arrays must have same length")
        return None, None
    
    mask = (distances >= peak_range[0]) & (distances <= peak_range[1])
    if not np.any(mask):
        print(f"Warning: No data points in range {peak_range}")
        return None, None
        
    # Apply simple smoothing
    window_size = 3
    if len(rdf[mask]) > window_size:
        kernel = np.ones(window_size)/window_size
        smoothed_rdf = np.convolve(rdf[mask], kernel, mode='valid')
        peak_index = np.argmax(smoothed_rdf)
        return (distances[mask][peak_index + window_size//2], 
                smoothed_rdf[peak_index])
    else:
        peak_index = np.argmax(rdf[mask])
        return distances[mask][peak_index], rdf[mask][peak_index]

def plot_rdf_comparison(ax, distances1, y1, distances2, y2, 
                       label1, label2, ylabel, peak_info):
    """Enhanced plotting function with consistent styling"""
    ax.plot(distances1, y1, label=label1, linewidth=2.5, 
           linestyle='-', color=sns.color_palette()[0])
    ax.plot(distances2, y2, label=label2, linewidth=2.5, 
           linestyle='--', color=sns.color_palette()[1])
    
    ax.set_xlabel(r'Distance (\si{\angstrom})', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(frameon=True, framealpha=0.9, shadow=True)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_ylim(bottom=0)
    
    # Add peak information with proper LaTeX formatting
    text = (r"$\mathbf{" + label1.split('-')[0] + "}$-" + label1.split('-')[1] + ": " +
            f"{peak_info[0]:.2f}" + r"\,\si{\angstrom}" + "\n" +
            r"$\mathbf{" + label2.split('-')[0] + "}$-" + label2.split('-')[1] + ": " +
            f"{peak_info[1]:.2f}" + r"\,\si{\angstrom}")
            
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    # Get file paths with validation
    while True:
        file_path_1 = input("Path to BO data file (txt/data/dat): ")
        distances_bo, si_bo, na_bo, bo_bo = read_data_from_file(file_path_1)
        if distances_bo is not None:
            break
            
    while True:
        file_path_2 = input("Path to NBO data file (txt/data/dat): ")
        distances_nbo, si_nbo, na_nbo, nbo_nbo = read_data_from_file(file_path_2)
        if distances_nbo is not None:
            break

    # Find peaks with error handling
    si_bo_peak = find_first_peak(distances_bo, si_bo) or (0, 0)
    si_nbo_peak = find_first_peak(distances_nbo, si_nbo) or (0, 0)
    na_bo_peak = find_first_peak(distances_bo, na_bo) or (0, 0)
    na_nbo_peak = find_first_peak(distances_nbo, na_nbo) or (0, 0)
    bo_bo_peak = find_first_peak(distances_bo, bo_bo) or (0, 0)
    nbo_nbo_peak = find_first_peak(distances_nbo, nbo_nbo, (2.0, 3.0)) or (0, 0)

    # Create figure with constrained layout
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), 
                            constrained_layout=True)
    fig.suptitle(r'Radial Distribution Function Analysis', 
                fontsize=18, y=1.05)

    # Plot all comparisons
    plot_rdf_comparison(axes[0], distances_bo, si_bo, distances_nbo, si_nbo,
                       'Si-BO', 'Si-NBO', r'RDF$_{\mathrm{Si-O}}$',
                       (custom_round(si_bo_peak[0], 2), 
                       custom_round(si_nbo_peak[0], 2)))

    plot_rdf_comparison(axes[1], distances_bo, na_bo, distances_nbo, na_nbo,
                       'Na-BO', 'Na-NBO', r'RDF$_{\mathrm{Na-O}}$',
                       (custom_round(na_bo_peak[0], 2), 
                       custom_round(na_nbo_peak[0], 2)))

    plot_rdf_comparison(axes[2], distances_bo, bo_bo, distances_nbo, nbo_nbo,
                       'BO-BO', 'NBO-NBO', r'RDF$_{\mathrm{O-O}}$',
                       (custom_round(bo_bo_peak[0], 2), 
                       custom_round(nbo_nbo_peak[0], 2)))

    # Save figure with the requested name
    output_file = "RDF_25Na2O_NBO_BO_300K.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_file}")

    plt.show()

if __name__ == "__main__":
    main()

