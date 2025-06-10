import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
# ▼▼▼ V V V SET YOUR FILE PATH AND SETTINGS HERE V V V ▼▼▼

# 1. SET THE DIRECTORY WHERE YOUR CSV FILES ARE LOCATED
# ---
DATA_DIRECTORY = r'C:\Users\shrei\PycharmProjects\MasterProject\final scripts'

# 2. SET THE CATEGORIES TO PLOT (AS A LIST)
# ---
# All categories in this list will be drawn on the same figure.
CATEGORIES_TO_PLOT = ['NH_DJF', 'NH_MAM', 'NH_JJA', 'NH_SON']

# 3. SET THE FILE SEARCH PATTERN
# ---
CSV_PATTERN = 'mae_report_analysis_*.csv'


# ▲▲▲ A A A YOUR SETTINGS END HERE A A A ▲▲▲
# =============================================================================


def plot_mae_multi_category(plot_data_dict):
    """
    Generates a single plot of MAE vs. Offset for multiple categories.
    Each category is given a unique color, and each model a unique line style.
    """
    print("\n📈 Generating single plot for all specified categories...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 10))  # Made figure larger for readability

    # Define line styles for the different models
    model_styles = {
        'DL Model': {'linestyle': '-', 'marker': 'o'},
        'Persistence Baseline': {'linestyle': '--', 'marker': '^'},
        'Linear Baseline': {'linestyle': ':', 'marker': 's'}
    }

    # Assign a unique color to each category
    # Using a colormap to get distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES_TO_PLOT)))
    category_colors = {category: color for category, color in zip(CATEGORIES_TO_PLOT, colors)}

    # Loop through each category that has data
    for category, data_list in plot_data_dict.items():
        if not data_list:
            print(f"  ⚠️ No data collected for category '{category}'. Skipping.")
            continue

        # Sort the data by offset to ensure lines connect correctly
        data_list.sort(key=lambda x: x[0])
        offsets, dl_maes, persistence_maes, linear_maes = zip(*data_list)

        data_by_model = {
            'DL Model': dl_maes,
            'Persistence Baseline': persistence_maes,
            'Linear Baseline': linear_maes
        }

        # Plot a line for each model within this category
        for model_name, maes in data_by_model.items():
            ax.plot(offsets, maes,
                    label=f'{model_name} ({category})',
                    color=category_colors[category],
                    linestyle=model_styles[model_name]['linestyle'],
                    marker=model_styles[model_name]['marker'])

    # --- Formatting ---
    ax.set_title(f'Model MAE vs. Forecast Offset', fontsize=18, weight='bold')
    ax.set_xlabel('Forecast Offset', fontsize=14)
    ax.set_ylabel('Mean Absolute Error (mbar)', fontsize=14)

    # Get all unique offsets to set the x-ticks
    all_offsets = sorted(list(set(t[0] for cat_data in plot_data_dict.values() for t in cat_data)))
    if all_offsets:
        ax.set_xticks(all_offsets)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize='medium', loc='best')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    fig.tight_layout()

    # Display the plot without saving it to a file
    print("✅ Displaying plot...")
    plt.show()


def main():
    """
    Main function to find CSVs, parse data, and generate the multi-category plot.
    """
    print("--- MAE Plotting Script ---")

    full_search_path = os.path.join(DATA_DIRECTORY, CSV_PATTERN)
    print(f"🔍 Searching for CSV files at: '{full_search_path}'")

    csv_files = glob.glob(full_search_path)

    if not csv_files:
        print(f"\n❌ ERROR: No CSV files found at the specified location.")
        return

    print(f"Found {len(csv_files)} files: {sorted(csv_files)}")

    plot_data_dict = {category: [] for category in CATEGORIES_TO_PLOT}

    for f_path in csv_files:
        print(f"\nProcessing file: {f_path}")

        match = re.search(r'\d+', os.path.basename(f_path))
        if not match:
            print(f"  ⚠️ WARNING: Could not parse offset number from filename. Skipping.")
            continue

        offset = int(match.group(0))
        print(f"  Parsed Offset: {offset}")

        try:
            df = pd.read_csv(f_path)
            df.set_index('Category', inplace=True)

            for category in CATEGORIES_TO_PLOT:
                if category not in df.index:
                    print(f"  - Category '{category}' not found in this file.")
                    continue

                dl_mae = df.loc[category, 'DL_MAE']
                persistence_mae = df.loc[category, 'PERSISTENCE_MAE']
                linear_mae = df.loc[category, 'LINEAR_MAE']

                plot_data_dict[category].append((offset, dl_mae, persistence_mae, linear_mae))
                print(f"  + Collected data for category '{category}'")

        except Exception as e:
            print(f"  ❌ ERROR: Could not process file. Reason: {e}")
            continue

    # --- Generate the single, combined plot ---
    plot_mae_multi_category(plot_data_dict)


if __name__ == '__main__':
    main()