# import numpy as np
# import pandas as pd
# import os
#
# # Define base path
# base_path = r'C:\Users\shrei\PycharmProjects\MasterProject\final scripts'
#
# # Load all 4 CSV files
# mae_31 = pd.read_csv(os.path.join(base_path, 'mae_report_analysis_31.csv'))
# mae_32 = pd.read_csv(os.path.join(base_path, 'mae_report_analysis_32.csv'))
# mae_33 = pd.read_csv(os.path.join(base_path, 'mae_report_analysis_33.csv'))
# mae_34 = pd.read_csv(os.path.join(base_path, 'mae_report_analysis_34.csv'))
#
# # Store them in a list for easy iteration
# mae_files = [mae_31, mae_32, mae_33, mae_34]
#
# # Dictionary to store DL_MAE values per category
# dl_mae_by_category = {}
#
# # Iterate through each unique category in the first file
# for category in mae_31['Category']:
#     dl_mae_values = []
#     for df in mae_files:
#         # Get the DL_MAE value for the current category
#         row = df[df['Category'] == category]
#         if not row.empty:
#             dl_mae_values.append(row['DL_MAE'].values[0])
#         else:
#             dl_mae_values.append(np.nan)  # In case the category is missing
#     dl_mae_by_category[category] = dl_mae_values
#
# # Print the DL_MAE values for verification
# print("DL_MAE by Category:")
# for category, mae_list in dl_mae_by_category.items():
#     print(f"{category}: {mae_list}")
#
# # Now, add the TL MAE values based on the provided information
# tl_mae_info = {
#     "DJF_NH": [1.47, np.nan, 3.15, 4.19],
#     "MAM_NH": [1.17, np.nan, 3.47, 3.36],
#     "JJA_NH": [0.95, np.nan, 2.09, 2.8],
#     "SON_NH": [1.25, np.nan, np.nan, 3.35],
#     "NA": [1.27, 2.03, 2.94, 3.47],
#     "MED": [1.12, 1.73, 2.24],
#     "SO": [1.64, 4.38, 5.69, np.nan],
#     "NP": [1.21, 3.37, 2.76, 2.09]
# }
#
# # Add TL MAE values to the dictionary
# for category, tl_values in tl_mae_info.items():
#     if category in dl_mae_by_category:
#         # Append the TL MAE value to the existing list
#         dl_mae_by_category[category].append(tl_values)
#     else:
#         # If the category is not in the dictionary, create a new entry
#         dl_mae_by_category[category] = [np.nan] * 4 + [tl_values]
#
# # Print the combined DL_MAE and TL values
# print("\nCombined DL_MAE and TL by Category:")
# for category, mae_list in dl_mae_by_category.items():
#     print(f"{category}: {mae_list}")

import numpy as np
import matplotlib.pyplot as plt

# Data for the regular deep learning model
dl_mae_by_category = {
    'BOMB': [np.float64(1.3389), np.float64(1.9395), np.float64(3.082), np.float64(4.0017)],
    'GL': [np.float64(1.1492), np.float64(1.6607), np.float64(2.538), np.float64(3.1954)],
    'GO': [np.float64(1.1396), np.float64(1.6874), np.float64(2.6552), np.float64(3.4341)],
    'MID': [np.float64(1.1449), np.float64(1.6795), np.float64(2.618), np.float64(3.3625)],
    'NBOMB': [np.float64(1.1118), np.float64(1.6361), np.float64(2.5383), np.float64(3.2432)],
    'NH': [np.float64(1.1603), np.float64(1.674), np.float64(2.5724), np.float64(3.2469)],
    'NH_BOMB': [np.float64(1.4418), np.float64(2.0501), np.float64(3.2462), np.float64(4.1764)],
    'NH_DJF': [np.float64(1.3762), np.float64(1.9668), np.float64(3.0614), np.float64(3.878)],
    'NH_GL': [np.float64(1.1566), np.float64(1.6664), np.float64(2.5174), np.float64(3.1517)],
    'NH_GO': [np.float64(1.1613), np.float64(1.6788), np.float64(2.6258), np.float64(3.3411)],
    'NH_JJA': [np.float64(0.9323), np.float64(1.3502), np.float64(2.0053), np.float64(2.5098)],
    'NH_MAM': [np.float64(1.1482), np.float64(1.6669), np.float64(2.5449), np.float64(3.1908)],
    'NH_MED': [np.float64(1.0556), np.float64(1.4732), np.float64(2.0905), np.float64(2.5291)],
    'NH_MID': [np.float64(1.1638), np.float64(1.6784), np.float64(2.5754), np.float64(3.2473)],
    'NH_NA': [np.float64(1.207), np.float64(1.7766), np.float64(2.7974), np.float64(3.5613)],
    'NH_NBOMB': [np.float64(1.1141), np.float64(1.6113), np.float64(2.4529), np.float64(3.0763)],
    'NH_NH.': [np.float64(1.1586), np.float64(1.6718), np.float64(2.5642), np.float64(3.2335)],
    'NH_NP': [np.float64(1.1533), np.float64(1.6517), np.float64(2.5681), np.float64(3.2617)],
    'NH_SA': [np.float64(np.nan), np.float64(np.nan), np.float64(np.nan), np.float64(np.nan)],
    'NH_SH': [np.float64(np.nan), np.float64(np.nan), np.float64(np.nan), np.float64(np.nan)],
    'NH_SO': [np.float64(np.nan), np.float64(np.nan), np.float64(np.nan), np.float64(np.nan)],
    'NH_SON': [np.float64(1.1767), np.float64(1.7061), np.float64(2.6627), np.float64(3.3977)],
    'NH_SUB': [np.float64(1.0519), np.float64(1.3452), np.float64(1.9361), np.float64(2.4854)],
    'SH': [np.float64(1.1278), np.float64(1.6818), np.float64(2.6577), np.float64(3.4581)],
    'SH_BOMB': [np.float64(1.2489), np.float64(1.8431), np.float64(2.9395), np.float64(3.8511)],
    'SH_DJF': [np.float64(1.0016), np.float64(1.5081), np.float64(2.3918), np.float64(3.1347)],
    'SH_GL': [np.float64(1.1216), np.float64(1.6392), np.float64(2.6138), np.float64(3.3538)],
    'SH_GO': [np.float64(1.13), np.float64(1.6911), np.float64(2.6682), np.float64(3.4753)],
    'SH_JJA': [np.float64(1.2239), np.float64(1.8121), np.float64(2.8401), np.float64(3.7093)],
    'SH_MAM': [np.float64(1.1647), np.float64(1.7442), np.float64(2.767), np.float64(3.6003)],
    'SH_MED': [np.float64(np.nan), np.float64(np.nan), np.float64(np.nan), np.float64(np.nan)],
    'SH_MID': [np.float64(1.1266), np.float64(1.6806), np.float64(2.6588), np.float64(3.4727)],
    'SH_NA': [np.float64(np.nan), np.float64(np.nan), np.float64(np.nan), np.float64(np.nan)],
    'SH_NBOMB': [np.float64(1.1097), np.float64(1.6581), np.float64(2.6142), np.float64(3.3912)],
    'SH_NH': [np.float64(np.nan), np.float64(np.nan), np.float64(np.nan), np.float64(np.nan)],
    'SH_NP': [np.float64(np.nan), np.float64(np.nan), np.float64(np.nan), np.float64(np.nan)],
    'SH_SA': [np.float64(1.1162), np.float64(1.6562), np.float64(2.6247), np.float64(3.4304)],
    'SH_SH': [np.float64(1.1289), np.float64(1.6841), np.float64(2.6608), np.float64(3.4585)],
    'SH_SO': [np.float64(1.1318), np.float64(1.6919), np.float64(2.6736), np.float64(3.496)],
    'SH_SON': [np.float64(1.1237), np.float64(1.6721), np.float64(2.6483), np.float64(3.3995)],
    'SH_SUB': [np.float64(0.9703), np.float64(1.2812), np.float64(1.9349), np.float64(2.1919)],
    'SUB': [np.float64(0.9979), np.float64(1.3028), np.float64(1.9353), np.float64(2.2897)],
    'unclassified': [np.float64(1.1235), np.float64(1.6544), np.float64(2.5745), np.float64(3.3209)]
}

# Data for the transfer learning model
tl_mae_by_category = {
    'NH_DJF': [1.47, np.nan, 3.15, 4.19],
    'NH_MAM': [1.17, np.nan, 3.47, 3.36],
    'NH_JJA': [0.95, np.nan, 2.09, 2.8],
    'NH_SON': [1.25, np.nan, np.nan, 3.35]
}

# Offsets for the x-axis
offsets = ["6", "12", "18", "24"]

# --- Plotting Configuration ---
seasons = ['DJF', 'MAM', 'JJA', 'SON']
# Define a color for each season
colors = {
    'DJF': 'blue',
    'MAM': 'green',
    'JJA': 'red',
    'SON': 'darkorange'
}

# --- Plotting ---

# Create a larger plot to accommodate all the lines and the legend
fig, ax = plt.subplots(figsize=(14, 8))

# Loop through each season to plot the data
for season in seasons:
    color = colors[season]

    # --- Plot NH (Regular Model) - Solid Line ---
    nh_key = f'NH_{season}'
    if nh_key in dl_mae_by_category:
        ax.plot(offsets, dl_mae_by_category[nh_key],
                linestyle='-', marker='o', color=color,
                label=f'NH {season} (Regular)')

    # --- Plot SH (Regular Model) - Dashed Line ---
    sh_key = f'SH_{season}'
    if sh_key in dl_mae_by_category:
        ax.plot(offsets, dl_mae_by_category[sh_key],
                linestyle='--', marker='x', color=color,
                label=f'SH {season} (Regular)')

    # --- Plot TL (NH Only) - Dotted Line ---
    tl_key = f'NH_{season}'
    if tl_key in tl_mae_by_category:
        ax.plot(offsets, tl_mae_by_category[tl_key],
                linestyle=':', marker='s', color=color,
                label=f'NH {season} (TL)')

# --- Final Touches ---

# Add labels and title
ax.set_xlabel('Offset (hours)')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('Seasonal MAE Comparison vs. Offset by Hemisphere')

# Add a grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Place the legend outside of the plot area for clarity
ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# Adjust layout to prevent the legend from being cut off
fig.tight_layout(rect=[0, 0, 0.85, 1])

# Show the plot
plt.show()

# Data for the regional transfer learning model
tl_regional_mae = {
    'NA': [1.27, 2.03, 2.94, 3.47],
    'MED': [1.12, 1.73, 2.24, np.nan],
    'SO': [1.64, 4.38, 5.69, np.nan],
    'NP': [1.21, 3.37, 2.76, 2.09]
}

# Offsets for the x-axis
offsets = ["6", "12", "18", "24"]

# --- Plotting Configuration ---
# Define the regions to plot and their corresponding keys for the regular model
regions_to_plot = {
    'NA': 'NH_NA',
    'NP': 'NH_NP',
    'MED': 'NH_MED',
    'SO': 'SH_SO'
}

# Define a color for each region
colors = {
    'NA': 'blue',
    'NP': 'green',
    'MED': 'red',
    'SO': 'purple'
}

# --- Plotting ---

# Create a larger plot to accommodate all the lines and the legend
fig, ax = plt.subplots(figsize=(12, 7))

# Loop through each region to plot its data
for region_name, data_key in regions_to_plot.items():
    color = colors[region_name]

    # Plot Regular Model data (solid line)
    if data_key in dl_mae_by_category:
        ax.plot(offsets, dl_mae_by_category[data_key],
                linestyle='-', marker='o', color=color,
                label=f'{region_name} (Regular)')

    # Plot Transfer Learning data (dotted line)
    if region_name in tl_regional_mae:
        ax.plot(offsets, tl_regional_mae[region_name],
                linestyle=':', marker='s', color=color,
                label=f'{region_name} (TL)')

# --- Final Touches ---

# Add labels and title
ax.set_xlabel('Offset (hours)')
ax.set_ylabel('Mean Absolute Error (MAE)')
ax.set_title('Regional MAE Comparison: Regular vs. Transfer Learning')

# Add a grid for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend to identify the lines
ax.legend()

# Adjust layout to ensure everything fits
fig.tight_layout()

# Show the plot
plt.show()
