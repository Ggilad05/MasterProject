import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Base path where all folders are located
base_path = "/home/mansour/ML3300-24a/shreibshtein/scripts/" # Make sure this path is correct

# Plotting styling
plot_linewidth = 1.8

# Define the exact plots, their styles, and their order
# Note: '6d' implies folder_id '6', '12d' implies '12', etc.
# '334 persistance' is interpreted as '34 persistance'
plot_definitions = [
    # Group 1: Blue
    {'label': '31', 'folder_id': '31', 'type': 'dl', 'color': 'blue', 'linestyle': 'solid'},
    {'label': '6d', 'folder_id': '6', 'type': 'dl', 'color': 'blue', 'linestyle': 'dotted'},
    {'label': '31 persistance', 'folder_id': '31', 'type': 'persistence', 'color': 'blue', 'linestyle': 'dashed'},

    # Group 2: Red
    {'label': '32', 'folder_id': '32', 'type': 'dl', 'color': 'red', 'linestyle': 'solid'},
    {'label': '12d', 'folder_id': '12', 'type': 'dl', 'color': 'red', 'linestyle': 'dotted'},
    {'label': '32 persistance', 'folder_id': '32', 'type': 'persistence', 'color': 'red', 'linestyle': 'dashed'},

    # Group 3: Green
    {'label': '33', 'folder_id': '33', 'type': 'dl', 'color': 'green', 'linestyle': 'solid'},
    {'label': '18d', 'folder_id': '18', 'type': 'dl', 'color': 'green', 'linestyle': 'dotted'},
    {'label': '33 persistance', 'folder_id': '33', 'type': 'persistence', 'color': 'green', 'linestyle': 'dashed'},

    # Group 4: Yellow (Gold)
    {'label': '34', 'folder_id': '34', 'type': 'dl', 'color': 'gold', 'linestyle': 'solid'},
    {'label': '24d', 'folder_id': '24', 'type': 'dl', 'color': 'gold', 'linestyle': 'dotted'},
    {'label': '34 persistance', 'folder_id': '34', 'type': 'persistence', 'color': 'gold', 'linestyle': 'dashed'},
]

# Prepare the plot
plt.figure(figsize=(14, 8))
plotted_any = False
plotted_handles = []
plotted_labels = []

# Loop through the defined plot configurations
for pdef in plot_definitions:
    folder_name = f"saved_graph1_{pdef['folder_id']}"
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.isdir(folder_path):
        print(f"Warning: Folder '{folder_path}' for label '{pdef['label']}' not found. Skipping.")
        continue

    data_filename = ""
    if pdef['type'] == 'dl':
        data_filename = 'all_raw_dl.npy'
    elif pdef['type'] == 'persistence':
        data_filename = 'all_raw_persistence.npy'
    else:
        print(f"Warning: Unknown plot type '{pdef['type']}' for label '{pdef['label']}'. Skipping.")
        continue

    data_path = os.path.join(folder_path, data_filename)

    if os.path.isfile(data_path):
        try:
            errors = np.load(data_path)
            if errors.size > 1 and np.std(errors) > 0:  # Check for variability
                print(f"Processing: Label='{pdef['label']}', Folder='{folder_name}', File='{data_filename}', Shape='{errors.shape}'")
                kde = gaussian_kde(errors)
                x_range = np.linspace(min(errors), max(errors), 500)
                line, = plt.plot(x_range, kde(x_range),
                                 label=pdef['label'],
                                 color=pdef['color'],
                                 linestyle=pdef['linestyle'],
                                 linewidth=plot_linewidth)
                plotted_handles.append(line)
                plotted_labels.append(pdef['label'])
                plotted_any = True
            else:
                print(f"Warning: Data in '{data_path}' for label '{pdef['label']}' has no variability or insufficient data. Skipping KDE.")
        except Exception as e:
            print(f"Error loading or processing file '{data_path}' for label '{pdef['label']}': {e}. Skipping.")
    else:
        print(f"Warning: Missing data file '{data_filename}' in '{folder_path}' for label '{pdef['label']}'. Skipping.")


plt.title("Error Distributions for different models ")
plt.xlabel("Error (mbar)")
plt.ylabel("Density")
plt.grid(True)
plt.xlim(-10, 10)  # Consistent X-axis limit

if plotted_any:
    # Create legend with handles and labels in the desired order
    plt.legend(handles=plotted_handles, labels=plotted_labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside
    output_path = os.path.join(base_path, "custom_ordered_kde_error_distributions.png")
    plt.savefig(output_path)
    plt.show()
    print(f"Plot saved to: {output_path}")
else:
    print("No data was plotted. Ensure your data files and folder structure match the 'plot_definitions'.")
