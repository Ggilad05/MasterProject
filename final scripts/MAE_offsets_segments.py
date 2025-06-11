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
CATEGORIES_TO_PLOT = ['SH_SA','NH_NP','NH_NA', 'SH_SO', 'NH_MED']

# 3. SET THE FILE SEARCH PATTERN
# ---
CSV_PATTERN = 'mae_report_analysis_*.csv'

# 4. DEFINE THE MAPPING FROM FILENAME NUMBER TO FORECAST HOUR
# ---
OFFSET_MAPPING = {
    31: 6,
    32: 12,
    33: 18,
    34: 24
}


# ▲▲▲ A A A YOUR SETTINGS END HERE A A A ▲▲▲
# =============================================================================


def plot_mae_multi_category(plot_data_dict):
    """
    Generates a single plot of MAE vs. Offset for multiple categories.
    This version only plots the DL Model and Persistence Baseline.
    """
    print("\n📈 Generating single plot for all specified categories...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 10))

    # Define line styles for the models to be plotted
    model_styles = {
        'DL Model': {'linestyle': '-', 'marker': 'o'},
        'Persistence Baseline': {'linestyle': '--', 'marker': '^'}
    }

    # Assign a unique color to each category
    colors = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES_TO_PLOT)))
    category_colors = {category: color for category, color in zip(CATEGORIES_TO_PLOT, colors)}

    # Loop through each category that has data
    for category, data_list in plot_data_dict.items():
        if not data_list:
            print(f"  ⚠️ No data collected for category '{category}'. Skipping.")
            continue

        # Sort the data by offset to ensure lines connect correctly
        data_list.sort(key=lambda x: x[0])
        offsets, dl_maes, persistence_maes = zip(*data_list)

        data_by_model = {
            'DL Model': dl_maes,
            'Persistence Baseline': persistence_maes
        }

        # Plot a line for each model within this category
        for model_name, maes in data_by_model.items():
            ax.plot(offsets, maes,
                    label=f'{model_name} ({category})',
                    color=category_colors[category],
                    linestyle=model_styles[model_name]['linestyle'],
                    marker=model_styles[model_name]['marker'])

    # --- Formatting ---
    ax.set_title(f'Prediction Comparison Across Regions (NH)', fontsize=36)
    ax.set_xlabel('Forecast Offset (Hours)', fontsize=28)
    ax.set_ylabel('Mean Absolute Error (mbar)', fontsize=28)

    # Get all unique offsets to set the x-ticks
    all_offsets = sorted(list(set(t[0] for cat_data in plot_data_dict.values() for t in cat_data)))
    if all_offsets:
        ax.set_xticks(all_offsets)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=20, loc='best')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    fig.patch.set_facecolor('#81CFF3')
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    fig.tight_layout()

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
            print(f"  ⚠️ WARNING: Could not parse number from filename '{f_path}'. Skipping.")
            continue

        file_number = int(match.group(0))

        # Use the mapping to get the correct offset in hours
        offset_hour = OFFSET_MAPPING.get(file_number)

        if offset_hour is None:
            print(f"  ⚠️ WARNING: No mapping found for file number {file_number} in OFFSET_MAPPING. Skipping.")
            continue

        print(f"  Parsed file number {file_number} as Offset: {offset_hour} hours")

        try:
            df = pd.read_csv(f_path)
            df.set_index('Category', inplace=True)

            for category in CATEGORIES_TO_PLOT:
                if category not in df.index:
                    print(f"  - Category '{category}' not found in this file.")
                    continue

                dl_mae = df.loc[category, 'DL_MAE']
                persistence_mae = df.loc[category, 'PERSISTENCE_MAE']

                plot_data_dict[category].append((offset_hour, dl_mae, persistence_mae))
                print(f"  + Collected data for category '{category}'")

        except Exception as e:
            print(f"  ❌ ERROR: Could not process file. Reason: {e}")
            continue

    # --- Generate the single, combined plot ---
    plot_mae_multi_category(plot_data_dict)


if __name__ == '__main__':
    main()