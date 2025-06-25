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
CATEGORIES_TO_PLOT = ['NH_NP', 'NH_NA', 'SH_SO', 'NH_MED']

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


def plot_mae_multi_category(plot_data_dict, sample_counts):
    """
    Generates a final plot of MAE vs. Offset with manually placed annotations.
    """
    print("\n📈 Generating single plot for all specified categories...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 12))

    # --- Configuration Section ---
    label_mapping = {
        'NH_NA': 'N.ATL',
        'NH_NP': 'N.PAC',
        'SH_SO': 'S.O.',
        'NH_MED': 'MED'
    }

    high_contrast_colors = [
        '#4363D8',  # Blue
        '#E6194B',  # Red
        '#3CB44B',  # Green
        '#911EB4',  # Purple
    ]
    category_colors = {category: color for category, color in zip(CATEGORIES_TO_PLOT, high_contrast_colors)}

    # ▼▼▼ MANUAL LABEL POSITION CONFIGURATION ▼▼▼
    # Set the exact (x, y) coordinates for each label on the right.
    label_positions = {
        'N.ATL': (24.3, 3.7),
        'S.O.':  (24.3, 3.5),
        'N.PAC': (24.3, 3.1),
        'MED':   (24.3, 2.6)
    }
    # ▼▼▼ MANUAL PERSISTENCE ANNOTATION POSITION CONFIGURATION ▼▼▼
    # Set the exact (x, y) coordinates for each persistence annotation.
    persistence_anno_positions = {
        'NH_NA':  (7, 2.4),
        'NH_NP':  (7, 1.6),
        'NH_MED': (7, 1.4),
        'SH_SO':  (7, 2.2),
    }
    # ▲▲▲ END OF CONFIGURATION ▲▲▲

    # ▼▼▼ FONT SIZE CONFIGURATION ▼▼▼
    legend_fontsize = 25
    label_fontsize = 25
    persistence_fontsize = 25
    # ▲▲▲ END OF FONT SIZE CONFIGURATION ▲▲▲

    persistence_lookup = {}


    # Loop through each category to plot data
    for category, data_list in plot_data_dict.items():
        if not data_list: continue
        data_list.sort(key=lambda x: x[0])
        offsets, dl_maes, persistence_maes = zip(*data_list)

        # Store persistence values for later
        persistence_lookup[category] = {off: mae for off, mae in zip(offsets, persistence_maes)}
        print(persistence_lookup)

        hemisphere = category.split('_')[0]
        line_style = '-' if hemisphere == 'NH' else '--'
        color = category_colors.get(category, 'black')

        # Plot the DL Model line
        ax.plot(offsets, dl_maes, color=color, linestyle=line_style, linewidth=2.5, zorder=3)

        ax.plot()

    # --- Place Persistence Annotations Manually ---
    # for category, p_data in persistence_lookup.items():
    #     if 24 in p_data and category in persistence_anno_positions:
    #         val_24hr = p_data[24]
    #         x_pos, y_pos = persistence_anno_positions[category]
    #         ax.text(x_pos, y_pos, f"[{val_24hr:.2f}]", ha='right', va='center',
    #                 color=category_colors.get(category), fontsize=persistence_fontsize, fontweight='bold')

    # --- Place 6h Persistence Annotations Manually ---
    # for category, p_data in persistence_lookup.items():
    #     if 6 in p_data and category in persistence_anno_positions:
    #         val_6hr = p_data[6]
    #         x_pos, y_pos = persistence_anno_positions[category]
    #         ax.text(x_pos, y_pos - 0.3, f"[{val_6hr:.2f}] (6h)", ha='right', va='center',
    #                 color=category_colors.get(category), fontsize=persistence_fontsize, fontweight='normal')

    # # --- Place Region Labels Manually ---
    # for category, short_label in label_mapping.items():
    #     if category in category_colors and short_label in label_positions:
    #         color = category_colors[category]
    #         x_pos, y_pos = label_positions[short_label]
    #         ax.text(x_pos, y_pos, short_label,
    #                 color=color, fontsize=label_fontsize,
    #                 fontweight='bold', ha='left', va='center')


    # --- Formatting ---
    ax.set_title('Model MAE Comparison by Region', fontsize=36, pad=20)
    ax.set_xlabel('Forecast Offset (Hours)', fontsize=28, labelpad=15)
    ax.set_ylabel('Mean Absolute Error (mbar)', fontsize=28, labelpad=15)

    all_offsets = sorted(list(set(t[0] for cat_data in plot_data_dict.values() for t in cat_data)))
    if all_offsets:
        ax.set_xticks(all_offsets)
        ax.set_xlim(right=max(all_offsets) + 5)

    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)

    # --- Build Legend Text ---
    legend_lines = [
        'NH: Solid Line (—)', 'SH: Dashed Line (--)', '',
        '[value]: 24h Persistence MAE', ''
    ]
    if sample_counts:
        legend_lines.append('Mean Sample Counts:')
        for category in CATEGORIES_TO_PLOT:
            count = sample_counts.get(category, 'N/A')
            legend_lines.append(f"  {label_mapping.get(category, category)}: {count}")

    explanation_text = '\n'.join(legend_lines)
    ax.text(0.7, 0.45, explanation_text, transform=ax.transAxes, fontsize=legend_fontsize,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    fig.tight_layout()
    fig.patch.set_facecolor('#81CFF3')

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

    unique_categories = sorted(list(set(CATEGORIES_TO_PLOT)))

    plot_data_dict = {category: [] for category in unique_categories}
    sample_counts_raw = {category: [] for category in unique_categories}
    sample_counts_present = True

    for f_path in csv_files:
        print(f"\nProcessing file: {f_path}")
        match = re.search(r'\d+', os.path.basename(f_path))
        if not match: continue
        file_number = int(match.group(0))
        offset_hour = OFFSET_MAPPING.get(file_number)
        if offset_hour is None: continue
        print(f"  Parsed file number {file_number} as Offset: {offset_hour} hours")

        try:
            df = pd.read_csv(f_path)
            df.set_index('Category', inplace=True)
            for category in unique_categories:
                if category not in df.index: continue

                dl_mae = df.loc[category, 'DL_MAE']
                persistence_mae = df.loc[category, 'PERSISTENCE_MAE']
                plot_data_dict[category].append((offset_hour, dl_mae, persistence_mae))
                print(f"  + Collected data for category '{category}'")

                if sample_counts_present:
                    try:
                        count = df.loc[category, 'DL_N']
                        sample_counts_raw[category].append(count)
                    except KeyError:
                        print("    - NOTE: 'DL_N' column not found. Will not display sample counts.")
                        sample_counts_present = False

        except Exception as e:
            print(f"  ❌ ERROR: Could not process file. Reason: {e}")
            continue

    sample_counts_mean = {}
    if sample_counts_present:
        for category, counts_list in sample_counts_raw.items():
            if counts_list:
                sample_counts_mean[category] = int(np.mean(counts_list))
            else:
                sample_counts_mean[category] = 'N/A'

    plot_mae_multi_category(plot_data_dict, sample_counts_mean)


if __name__ == '__main__':
    main()