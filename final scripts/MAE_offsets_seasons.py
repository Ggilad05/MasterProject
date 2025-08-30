import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
from collections import defaultdict

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
# ▼▼▼ V V V SET YOUR FILE PATH AND SETTINGS HERE V V V ▼▼▼

# 1. SET THE DIRECTORY WHERE YOUR CSV FILES ARE LOCATED
# ---
DATA_DIRECTORY = r'C:\Users\shrei\PycharmProjects\MasterProject\final scripts'

# 2. SET THE CATEGORIES TO PLOT (AS A LIST)
# ---
CATEGORIES_TO_PLOT = ['NH_DJF', 'NH_MAM', 'NH_JJA', 'NH_SON', 'SH_DJF', 'SH_MAM', 'SH_JJA', 'SH_SON']

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
OFFSETS_ORDER = [6, 12, 18, 24]

def compute_skill_scores(plot_data_dict):
    """
    From plot_data_dict {category: [(offset, dl_mae, pers_mae), ...]},
    compute SS = 1 - dl_mae/pers_mae per (category, offset).
    Returns:
        ss_by_cat: dict[category] -> dict[offset] -> float
    """
    ss_by_cat = defaultdict(dict)
    for category, triples in plot_data_dict.items():
        for offset, dl_mae, pers_mae in triples:
            if pers_mae and pers_mae > 0:
                ss = 1.0 - (float(dl_mae) / float(pers_mae))
            else:
                ss = float('nan')
            ss_by_cat[category][offset] = ss
    return ss_by_cat

def compute_and_print_skill(plot_data_dict):
    """
    Compute SS tables and print them nicely.
    1) Full table: rows = categories (NH/SH × seasons), cols = offsets.
    2) Seasonal mean table: averages NH & SH for each season per offset.
    """
    ss_by_cat = compute_skill_scores(plot_data_dict)

    # Build DataFrame for all categories
    import pandas as pd
    rows = []
    for category in sorted(ss_by_cat.keys()):
        row = {"Category": category}
        for off in OFFSETS_ORDER:
            row[f"SS@{off}h"] = ss_by_cat[category].get(off, float('nan'))
        rows.append(row)
    df_all = pd.DataFrame(rows).set_index("Category")

    # Compute seasonal means (NH+SH) for DJF/MAM/JJA/SON if both exist
    season_names = ["DJF", "MAM", "JJA", "SON"]
    seasonal_rows = []
    for s in season_names:
        nh_key = f"NH_{s}"
        sh_key = f"SH_{s}"
        row = {"Season": s}
        for off in OFFSETS_ORDER:
            vals = []
            if nh_key in ss_by_cat and off in ss_by_cat[nh_key]:
                vals.append(ss_by_cat[nh_key][off])
            if sh_key in ss_by_cat and off in ss_by_cat[sh_key]:
                vals.append(ss_by_cat[sh_key][off])
            row[f"SS@{off}h"] = float(pd.Series(vals).mean()) if len(vals) else float('nan')
        seasonal_rows.append(row)
    df_season = pd.DataFrame(seasonal_rows).set_index("Season")

    # Print
    print("\n=== Skill Score (SS) vs Persistence by Category ===")
    print(df_all.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== Seasonal Mean Skill (NH & SH averaged) ===")
    print(df_season.to_string(float_format=lambda x: f"{x:.3f}"))

    # Optional: return dfs if you later want to save CSVs
    return df_all, df_season

def plot_mae_multi_category(plot_data_dict, sample_counts):
    """
    Generates a final, refined plot of MAE vs. Offset with all requested annotations.
    """
    print("\n📈 Generating single plot for all specified categories...")

    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 12))

    # --- Configuration Section ---
    season_mapping = {
        'NH_DJF': 'Winter', 'NH_MAM': 'Spring', 'NH_JJA': 'Summer', 'NH_SON': 'Autumn',
        'SH_DJF': 'Summer', 'SH_MAM': 'Autumn', 'SH_JJA': 'Winter', 'SH_SON': 'Spring'
    }
    season_colors = {
        'Winter': '#3498db', 'Spring': '#2ecc71', 'Summer': '#e74c3c', 'Autumn': '#f39c12'
    }
    season_abbr = {'Winter': '', 'Spring': '', 'Summer': '', 'Autumn': ''}

    # ▼▼▼ FONT SIZE CONFIGURATION ▼▼▼
    legend_fontsize = 22
    persistence_fontsize = 22
    season_label_fontsize = 22
    # ▲▲▲ END OF FONT SIZE CONFIGURATION ▲▲▲

    line_final_points = {}
    persistence_lookup = {}
    min_y, max_y = np.inf, -np.inf  # To track y-axis limits for label spacing

    # Loop through each category to plot data and collect info
    for category, data_list in plot_data_dict.items():
        if not data_list: continue
        data_list.sort(key=lambda x: x[0])
        offsets, dl_maes, persistence_maes = zip(*data_list)

        # Update y-axis limits based on the MAE data
        if dl_maes:
            min_y = min(min_y, min(dl_maes))
            max_y = max(max_y, max(dl_maes))

        persistence_lookup[category] = {off: mae for off, mae in zip(offsets, persistence_maes)}

        hemisphere = category.split('_')[0]
        line_style = '-' if hemisphere == 'NH' else '--'
        color = season_colors.get(season_mapping.get(category, ''), 'black')

        print(f"dl_maes: {dl_maes}")
        ax.plot(offsets, dl_maes, color=color, linestyle=line_style, marker='o', linewidth=2.5, zorder=3)

        # Add the 6h persistence baseline point
        # if offsets:
        #     ax.scatter(offsets[0], persistence_maes[0], color=color, marker='^', s=200, zorder=5, alpha=0.9)


        if len(offsets) > 1:
            line_final_points[category] = {
                'p2': (offsets[-1], dl_maes[-1]),
                'color': color,
                'season': season_mapping.get(category)
            }

    # # --- Prepare Persistence Annotations ---
    # persistence_annotations = []
    # for category, p_data in persistence_lookup.items():
    #     if 6 in p_data and 24 in p_data:
    #         season_name = season_mapping.get(category)
    #         abbr = season_abbr.get(season_name, '')
    #         hemisphere = category.split('_')[0]
    #         val_24hr = p_data[24]
    #
    #         if hemisphere == 'SH':
    #             anno_text = f"{{{val_24hr:.2f}}}{abbr}"
    #         else:
    #             anno_text = f"[{val_24hr:.2f}]{abbr}"
    #
    #         persistence_annotations.append({
    #             'text': anno_text,
    #             'color': season_colors.get(season_name),
    #             'category_key': category
    #         })

    # # --- Manual Placement for Each of the 8 Annotations ---
    # for anno in persistence_annotations:
    #     category = anno['category_key']
    #     if category == 'NH_DJF':
    #         x_pos, y_pos = 8, 2.9
    #         ax.text(x_pos, y_pos, anno['text'], ha='right', va='center', color=anno['color'],
    #                 fontsize=persistence_fontsize, fontweight='bold')
    #     elif category == 'NH_MAM':
    #         x_pos, y_pos = 9.5, 2.3
    #         ax.text(x_pos, y_pos, anno['text'], ha='right', va='center', color=anno['color'],
    #                 fontsize=persistence_fontsize, fontweight='bold')
    #     elif category == 'NH_JJA':
    #         x_pos, y_pos = 7, 1.8
    #         ax.text(x_pos, y_pos, anno['text'], ha='right', va='center', color=anno['color'],
    #                 fontsize=persistence_fontsize, fontweight='bold')
    #     elif category == 'NH_SON':
    #         x_pos, y_pos = 9.5, 2.7
    #         ax.text(x_pos, y_pos, anno['text'], ha='right', va='center', color=anno['color'],
    #                 fontsize=persistence_fontsize, fontweight='bold')
    #     elif category == 'SH_DJF':
    #         x_pos, y_pos = 8, 2.2
    #         ax.text(x_pos, y_pos, anno['text'], ha='right', va='center', color=anno['color'],
    #                 fontsize=persistence_fontsize, fontweight='bold')
    #     elif category == 'SH_MAM':
    #         x_pos, y_pos = 8, 2.4
    #         ax.text(x_pos, y_pos, anno['text'], ha='right', va='center', color=anno['color'],
    #                 fontsize=persistence_fontsize, fontweight='bold')
    #     elif category == 'SH_JJA':
    #         x_pos, y_pos = 8, 2.7
    #         ax.text(x_pos, y_pos, anno['text'], ha='right', va='center', color=anno['color'],
    #                 fontsize=persistence_fontsize, fontweight='bold')
    #     elif category == 'SH_SON':
    #         x_pos, y_pos = 9.5, 2.5
    #         ax.text(x_pos, y_pos, anno['text'], ha='right', va='center', color=anno['color'],
    #                 fontsize=persistence_fontsize, fontweight='bold')
    #
    # # --- Prepare and Place Season Labels ---
    labels_to_draw = []
    for season, color in season_colors.items():
        cats_for_season = [cat for cat, props in line_final_points.items() if props['season'] == season]
        if not cats_for_season: continue
        label_cat = max(cats_for_season, key=lambda c: line_final_points[c]['p2'][1])
        props = line_final_points[label_cat]

        hemisphere_for_label = label_cat.split('_')[0]
        # Using a simple hyphen for solid line as per previous request
        line_style_for_label = '-' if hemisphere_for_label == 'NH' else '--'
        label_text = f"{season}:  NH {line_style_for_label}, SH --" # This logic might need review if format is complex
        if hemisphere_for_label == 'NH':
            label_text = f"{season}: NH —, SH --"
        else:
            label_text = f"{season}: NH —, SH --" # A single label represents both

        labels_to_draw.append({'x': props['p2'][0], 'y': props['p2'][1], 'text': label_text, 'color': props['color']})

    # Intelligent label placement
    sorted_labels = sorted(labels_to_draw, key=lambda d: d['y'])
    last_y_plotted_s = -np.inf
    final_y_values = [d['y'] for d in labels_to_draw]
    if final_y_values:
        min_y_separation_s = (max(final_y_values) - min(final_y_values)) * 0.3
    else:
        min_y_separation_s = 0.35

    for label in sorted_labels:
        adjusted_y = max(label['y'], last_y_plotted_s + min_y_separation_s)
        ax.text(6, adjusted_y, label['text'], color=label['color'], fontsize=season_label_fontsize,
                fontweight='bold', ha='left', va='center')
        last_y_plotted_s = adjusted_y

    # ax.text(6, 0.4, '24h Persistence MAE (mbar)',
    #         color='black', fontsize=30, fontweight='bold', ha='left', va='top')

    # --- Formatting ---
    ax.set_title('Model MAE Comparison by Season and Hemisphere', fontsize=38)
    ax.set_xlabel('Forecast Offset (Hours)', fontsize=36, labelpad=15)
    ax.set_ylabel('MAE (mbar)', fontsize=28, labelpad=15)

    all_offsets = sorted(list(set(t[0] for cat_data in plot_data_dict.values() for t in cat_data)))
    if all_offsets:
        ax.set_xticks(all_offsets)
        # ax.set_xlim(right=max(all_offsets) + 5)

    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)

    # # --- Build Legend Text ---
    # legend_lines = [
    #     'o  DL Model',
    #     '^  6h Persistence',
    #     '',
    #     '24h Persistence :',
    #     '  NH: [24h MAE]id',
    #     '  SH: {24h MAE}id',
    #     ""
    # ]
    # if sample_counts:
    #     legend_lines.append('Mean Sample Counts:')
    #     for category in CATEGORIES_TO_PLOT:
    #         cat_name = f"{category.split('_')[0]} {season_mapping.get(category, '')}"
    #         count = sample_counts.get(category, 'N/A')
    #         legend_lines.append(f"  {cat_name}: {count}")
    #
    # explanation_text = '\n'.join(legend_lines)
    # ax.text(0.98, 0.02, explanation_text, transform=ax.transAxes, fontsize=legend_fontsize,
    #         verticalalignment='bottom', horizontalalignment='right',
    #         bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    # ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    fig.tight_layout()
    # fig.patch.set_facecolor('#81CFF3')
    ax.set_ylim(bottom=0)


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
    sample_counts_raw = {category: [] for category in CATEGORIES_TO_PLOT}
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
            for category in CATEGORIES_TO_PLOT:
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
    compute_and_print_skill(plot_data_dict)


if __name__ == '__main__':
    main()