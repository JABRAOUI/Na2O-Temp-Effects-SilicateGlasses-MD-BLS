import os
import matplotlib.pyplot as plt

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        if line.strip().startswith('#'):  # Skip lines starting with '#'
            continue
        row = line.split()
        try:
            data.append([float(val) for val in row])
        except ValueError:
            continue
    return data

def subsample_data(x, y, num_points=40):
    # Subsample data to keep only 'num_points' evenly spaced points
    step = max(1, len(x) // num_points)
    x_subsampled = x[::step]
    y_subsampled = y[::step]
    return x_subsampled, y_subsampled

def main():
    # Prompt the user to enter file paths (skip 0% Na₂O)
    file_paths = [
        input(f"Enter the path for File {i+1} ({(i+1)*5}% Na₂O): ")
        for i in range(7)  # Only 7 files for 5% to 35% Na₂O
    ]

    # Read the data from the files
    data = [read_data(file_path) for file_path in file_paths]

    # Extract relevant columns (assuming columns are 0-indexed)
    column_y_idx = 12 
    column_x_idx = 1 

    # Extract data for each file
    x_data = []
    y_data = []
    for dataset in data:
        x_temp = []
        y_temp = []
        for row in dataset:
            if len(row) > max(column_x_idx, column_y_idx):  # Ensure row has enough columns
                x_temp.append(float(row[column_x_idx]))
                y_temp.append(float(row[column_y_idx]))
            else:
                print(f"Skipping row with insufficient columns: {row}")
        x_data.append(x_temp)
        y_data.append(y_temp)

    # Filter data within the x range [0, 6]
    x_filtered = []
    y_filtered = []
    for x, y in zip(x_data, y_data):
        x_f, y_f = zip(*[(xi, yi) for xi, yi in zip(x, y) if 0 <= xi <= 6])
        x_filtered.append(x_f)
        y_filtered.append(y_f)

    # Subsample the data to keep only a few points with markers
    num_points_to_plot = 40
    x_sub = []
    y_sub = []
    for x_f, y_f in zip(x_filtered, y_filtered):
        x_s, y_s = subsample_data(x_f, y_f, num_points_to_plot)
        x_sub.append(x_s)
        y_sub.append(y_s)

    # Create the plot
    plt.figure(figsize=(6, 6))
    colors = ['black', 'orange', 'red', 'purple', 'pink', 'brown', 'green']  # Colors for 5% to 35%
    labels = [f'{(i+1)*5}% Na₂O 300K' for i in range(7)]  # Labels for 5% to 35%

    # Add an offset to each curve to make differences visible
    offset = 3  # Adjust this value as needed
    for i in range(len(file_paths)):
        y_offset = [y_val + i * offset for y_val in y_filtered[i]]  # Add offset to y-values
        plt.plot(x_filtered[i], y_offset, '-', label=labels[i], color=colors[i], lw=2)  # Plot with line

    plt.xlabel(r'r [$\AA$]', fontsize=14)  # Fixed escape sequence
    plt.ylabel('RDF ($r_{Na-O}$)', fontsize=14)  # Update ylabel to indicate offset

    # Increase tick label size for both x and y axes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Increase legend font size and position
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot as a PNG image in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'plot_RDF_NaO_curves_300K_5_to_35.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')  # dpi=300 for higher resolution

    plt.show()

if __name__ == "__main__":
    main()
