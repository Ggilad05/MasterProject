
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_position, minimum_position

# --- Helper function: extract square window ---
def extract_window(field, center_y, center_x, radius):
    y0 = max(center_y - radius, 0)
    y1 = min(center_y + radius + 1, field.shape[0])
    x0 = max(center_x - radius, 0)
    x1 = min(center_x + radius + 1, field.shape[1])
    return field[y0:y1, x0:x1], (y0, x0)

# --- MSLP couplet finder using optional radius window ---

def compute_mslp_couplets(FMSLP, RMSLP, center_y=None, center_x=None, radius=None):
    """
    Compute ΔMSLP = FMSLP - RMSLP and find local extrema (+/-) within full field
    or within a window centered at (center_y, center_x) if given.
    Also plots the difference field with min/max markers and contours.
    """
    diff = FMSLP - RMSLP

    if center_y is not None and center_x is not None and radius is not None:
        diff_win, (dy0, dx0) = extract_window(diff, center_y, center_x, radius)
        era5_win, _ = extract_window(RMSLP, center_y, center_x, radius)
        forecast_win, _ = extract_window(FMSLP, center_y, center_x, radius)

        pos_idx = maximum_position(diff_win)
        neg_idx = minimum_position(diff_win)

        pos_y, pos_x = pos_idx[0] + dy0, pos_idx[1] + dx0
        neg_y, neg_x = neg_idx[0] + dy0, neg_idx[1] + dx0

        # Plot the difference window with contours and markers
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(diff_win, cmap='RdBu_r', origin='lower')
        ax.set_title("ΔMSLP window (FMSLP - RMSLP)")

        ax.plot(pos_idx[1], pos_idx[0], 'go', markersize=10, label='Max ΔMSLP (+)')
        ax.plot(neg_idx[1], neg_idx[0], 'ro', markersize=10, label='Min ΔMSLP (−)')

        ax.contour(era5_win, colors='blue', linewidths=1.2)
        ax.contour(forecast_win, colors='red', linewidths=1.2, linestyles='dashed')

        ax.legend(loc='upper right')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("ΔMSLP (hPa)")
        plt.tight_layout()
        plt.show()

    else:
        pos_y, pos_x = maximum_position(diff)
        neg_y, neg_x = minimum_position(diff)

    delta_pos = FMSLP[pos_y, pos_x] - RMSLP[pos_y, pos_x]
    delta_neg = FMSLP[neg_y, neg_x] - RMSLP[neg_y, neg_x]
    delta_dipole = delta_neg - delta_pos

    return {
        "pos_center": (pos_y, pos_x),
        "neg_center": (neg_y, neg_x),
        "delta_pos": delta_pos,
        "delta_neg": delta_neg,
        "delta_dipole": delta_dipole,
        "f_pos": FMSLP[pos_y, pos_x],
        "r_pos": RMSLP[pos_y, pos_x],
        "f_neg": FMSLP[neg_y, neg_x],
        "r_neg": RMSLP[neg_y, neg_x],
    }
# --- Main loop ---
era5_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\msl\\2010\\"
forecast_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\forecast_0\\msl\\2010\\"

radius = 8  # Radius around the cyclone center to evaluate min SLP (used if center is provided)
results = []

for file in os.listdir(forecast_folder):
    if not file.endswith(".npy"):
        continue

    era5_path = os.path.join(era5_folder, file)
    forecast_path = os.path.join(forecast_folder, file)

    if not os.path.exists(era5_path):
        print(f"Skipping {file} — ERA5 file not found.")
        continue

    era5_data = np.load(era5_path)[4:]        # shape: (T, Y, X)
    forecast_data = np.load(forecast_path)[4:]

    n_frames, ny, nx = era5_data.shape

    for t in range(n_frames):
        FMSLP = forecast_data[t]
        RMSLP = era5_data[t]

        # ⚠️ Optional: Replace with real cyclone center if available
        cyclone_center_y = ny // 2
        cyclone_center_x = nx // 2

        result = compute_mslp_couplets(
            FMSLP, RMSLP,
            center_y=cyclone_center_y,
            center_x=cyclone_center_x,
            radius=radius
        )

        print(
            f"{file} | t={t} | "
            f"Δ+ = {result['delta_pos']:.2f} at {result['pos_center']} "
            f"| Δ− = {result['delta_neg']:.2f} at {result['neg_center']}"
        )

        results.append({
            "file": file,
            "time": t,
            "delta_pos": result["delta_pos"],
            "delta_neg": result["delta_neg"],
            "delta_dipole": result["delta_dipole"],
            "f_pos": result["f_pos"],
            "r_pos": result["r_pos"],
            "f_neg": result["f_neg"],
            "r_neg": result["r_neg"],
            "pos_center": result["pos_center"],
            "neg_center": result["neg_center"],
        })
# %%

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_position, minimum_position
import pandas as pd

def extract_window(field, center_y, center_x, radius):
    y0 = max(center_y - radius, 0)
    y1 = min(center_y + radius + 1, field.shape[0])
    x0 = max(center_x - radius, 0)
    x1 = min(center_x + radius + 1, field.shape[1])
    return field[y0:y1, x0:x1], (y0, x0)

def plot_minima_and_contours(FMSLP, RMSLP, center_y, center_x, radius=8):
    """
    Plot ERA5 and Forecast MSLP contours in a window around a cyclone center,
    mark their local minima, and print ΔMSLP between those minima.
    """
    # Extract local windows
    forecast_win, (fy0, fx0) = extract_window(FMSLP, center_y, center_x, radius)
    era5_win, (ry0, rx0) = extract_window(RMSLP, center_y, center_x, radius)

    # Find local minima
    f_idx = np.unravel_index(np.argmin(forecast_win), forecast_win.shape)
    r_idx = np.unravel_index(np.argmin(era5_win), era5_win.shape)

    f_val = forecast_win[f_idx]
    r_val = era5_win[r_idx]

    f_abs = (f_idx[0] + fy0, f_idx[1] + fx0)
    r_abs = (r_idx[0] + ry0, r_idx[1] + rx0)

    delta_mslp = f_val - r_val

    # Plot contours and minima
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow((forecast_win + era5_win) / 2, cmap='gray', origin='lower', alpha=0.2)
    ax.contour(era5_win, colors='blue', linewidths=1.2, label='ERA5 Contour')
    ax.contour(forecast_win, colors='red', linewidths=1.2, linestyles='dashed', label='Forecast Contour')

    ax.plot(f_idx[1], f_idx[0], 'gx', markersize=10, markeredgewidth=2, label='Forecast Min')
    ax.plot(r_idx[1], r_idx[0], 'kx', markersize=10, markeredgewidth=2, label='ERA5 Min')

    ax.set_title(f"MSLP Minima Comparison (ΔMSLP = {delta_mslp:.2f} hPa)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    print(f"Forecast Min: {f_val:.2f} hPa at {f_abs}")
    print(f"ERA5 Min:     {r_val:.2f} hPa at {r_abs}")
    print(f"ΔMSLP = F - R = {delta_mslp:.2f} hPa")

# --- Main loop ---
era5_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\msl\\2010\\"
forecast_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\forecast_0\\msl\\2010\\"

# Build the table
error_table = pd.DataFrame([
    {
        "Name": entry["file"],
        "Step": entry["time"],
        "Error (Pa)": entry["f_neg"] - entry["r_neg"]  # Forecast min - ERA5 min
    }
    for entry in results
])

radius = 8  # Radius around the cyclone center to evaluate min SLP
for file in os.listdir(forecast_folder):
    if not file.endswith(".npy"):
        continue

    era5_path = os.path.join(era5_folder, file)
    forecast_path = os.path.join(forecast_folder, file)

    if not os.path.exists(era5_path):
        print(f"Skipping {file} — ERA5 file not found.")
        continue

    era5_data = np.load(era5_path)[4:]        # shape: (T, Y, X)
    forecast_data = np.load(forecast_path)[4:]

    n_frames, ny, nx = era5_data.shape

    for t in range(n_frames):
        FMSLP = forecast_data[t]
        RMSLP = era5_data[t]

        # ⚠️ Replace with real cyclone center if available
        cyclone_center_y = ny // 2
        cyclone_center_x = nx // 2

        print(f"{file} | t={t}")
        plot_minima_and_contours(FMSLP, RMSLP, cyclone_center_y, cyclone_center_x, radius=radius)

# Plot histogram of the pressure errors
plt.figure(figsize=(8, 5))
plt.hist(error_table["Error (Pa)"], bins=40, color='skyblue', edgecolor='black')
plt.title("Histogram of Forecast - ERA5 Min MSLP Differences (Pa)")
plt.xlabel("ΔMSLP (Pa)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Helper function: extract a square window from a 2D field ---
def extract_window(field, center_y, center_x, radius):
    """
    Extracts a square window of a given radius from a 2D numpy array.
    """
    y0 = max(center_y - radius, 0)
    y1 = min(center_y + radius + 1, field.shape[0])
    x0 = max(center_x - radius, 0)
    x1 = min(center_x + radius + 1, field.shape[1])
    return field[y0:y1, x0:x1]

# --- Main analysis loop ---

# Define paths to the data folders
era5_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\msl\\2010\\"
forecast_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\forecast_0\\msl\\2010\\"
hour_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\hour\\2010\\"

# This list will store the results for each file and time step
results_list = []

# Radius around the cyclone center to find the minimum MSLP
radius = 8

# Loop through each file in the forecast directory
print("Processing files...")
for file in os.listdir(forecast_folder):
    if not file.endswith(".npy"):
        continue

    # Construct the full paths for both forecast and ERA5 files
    forecast_path = os.path.join(forecast_folder, file)
    era5_path = os.path.join(era5_folder, file)
    hour_path = os.path.join(hour_folder, file)

    if not os.path.exists(era5_path):
        print(f"Skipping {file} — Corresponding ERA5 file not found.")
        continue

    # Load the data arrays (assuming MSLP is in hPa)
    era5_data = np.load(era5_path)[4:]
    forecast_data = np.load(forecast_path)[4:]
    hour_data = np.load(hour_path)[4:,0,0]


    n_frames, ny, nx = era5_data.shape

    # Process each time step (frame) in the data
    for t in range(n_frames):
        FMSLP = forecast_data[t]
        RMSLP = era5_data[t]

        # ⚠️ IMPORTANT: Define the approximate center of the cyclone.
        # This is a placeholder for the cyclone's approximate location.
        cyclone_center_y = ny // 2
        cyclone_center_x = nx // 2

        # Extract the relevant window around the cyclone center
        forecast_window = extract_window(FMSLP, cyclone_center_y, cyclone_center_x, radius)
        era5_window = extract_window(RMSLP, cyclone_center_y, cyclone_center_x, radius)

        # Find the minimum pressure value within each window (in hPa)
        min_forecast_hpa = np.min(forecast_window)
        min_era5_hpa = np.min(era5_window)

        # Calculate the error (Forecast - ERA5) in hPa
        error_hpa = min_forecast_hpa - min_era5_hpa

        # Append the results, including the absolute error for MAE calculation
        results_list.append({
            "Name": file,
            "Step": t,
            "Error (hPa)": error_hpa,
            "Absolute Error (hPa)": abs(error_hpa)
        })

print("✅ Processing complete.")

# --- Tabulate and Plot Results ---

# 1. Add all the results to a table (Pandas DataFrame)
error_table = pd.DataFrame(results_list)

print("\n--- Error Table ---")
print("Showing the first 5 rows of the results:")
print(error_table.head())

# --- Calculate and Print Mean Absolute Error ---

# 2. Calculate the mean of the 'Absolute Error (hPa)' column
mean_mae = error_table["Absolute Error (hPa)"].mean()

print("\n--- Overall Performance Metric ---")
print(f"📊 Mean Absolute Error (MAE): {mean_mae:.2f} hPa")


# --- Visualize the Error Distribution ---

# 3. Make a histogram of all errors
print("\nGenerating histogram...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

plt.hist(error_table["Error (hPa)"], bins=50, color='dodgerblue', edgecolor='black', alpha=0.7)

plt.title("Histogram of Forecast Errors (Forecast MSLP - ERA5 MSLP)", fontsize=16)
plt.xlabel("Error in Minimum Sea Level Pressure (hPa)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No Error (Forecast = ERA5)')
plt.legend()
plt.tight_layout()
plt.show()
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Helper function: extract a square window from a 2D field ---
def extract_window(field, center_y, center_x, radius):
    """
    Extracts a square window of a given radius from a 2D numpy array.
    """
    y0 = max(center_y - radius, 0)
    y1 = min(center_y + radius + 1, field.shape[0])
    x0 = max(center_x - radius, 0)
    x1 = min(center_x + radius + 1, field.shape[1])
    return field[y0:y1, x0:x1]


# --- Main analysis loop ---

# Define paths to the data folders
era5_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\msl\\2010\\"
forecast_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\forecast_0\\msl\\2010\\"
hour_folder = r"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\hour\\2010\\"

# This list will store the results for each file and time step
results_list = []

# Radius around the cyclone center to find the minimum MSLP
radius = 8

# Loop through each file in the forecast directory
print("Processing files...")
for file in os.listdir(forecast_folder):
    if not file.endswith(".npy"):
        continue

    forecast_path = os.path.join(forecast_folder, file)
    era5_path = os.path.join(era5_folder, file)
    hour_path = os.path.join(hour_folder, file)

    if not os.path.exists(era5_path):
        print(f"Skipping {file} — Corresponding ERA5 file not found.")
        continue

    if not os.path.exists(hour_path):
        print(f"Skipping {file} — Corresponding hour file not found at {hour_path}.")
        continue

    era5_data = np.load(era5_path)[4:]
    forecast_data = np.load(forecast_path)[4:]
    try:
        hour_data = np.load(hour_path)[4:, 0, 0]
    except Exception as e:
        print(f"Error loading hour data for {file}: {e}. Skipping file.")
        continue

    if not (era5_data.shape[0] == forecast_data.shape[0] == hour_data.shape[0]):
        print(f"Skipping {file} — Mismatch in number of frames after slicing between MSLP and hour data.")
        print(
            f"ERA5 frames: {era5_data.shape[0]}, Forecast frames: {forecast_data.shape[0]}, Hour frames: {hour_data.shape[0]}")
        continue

    n_frames, ny, nx = era5_data.shape

    for t in range(n_frames):
        FMSLP = forecast_data[t]
        RMSLP = era5_data[t]
        current_forecast_hour = hour_data[t]

        cyclone_center_y = ny // 2
        cyclone_center_x = nx // 2

        forecast_window = extract_window(FMSLP, cyclone_center_y, cyclone_center_x, radius)
        era5_window = extract_window(RMSLP, cyclone_center_y, cyclone_center_x, radius)

        min_forecast_hpa = np.min(forecast_window)
        min_era5_hpa = np.min(era5_window)
        error_hpa = (min_forecast_hpa - min_era5_hpa)/ 100

        results_list.append({
            "Name": file,
            "Step": t,
            "Forecast_Hour": current_forecast_hour,
            "Error (hPa)": error_hpa,
            "Absolute Error (hPa)": abs(error_hpa)
        })

print("✅ Processing complete.")

if not results_list:
    print("No results were processed. Exiting.")
    exit()

error_table = pd.DataFrame(results_list)

print("\n--- Error Table ---")
print("Showing the first 5 rows of the results (including Forecast_Hour):")
print(error_table.head())

mean_mae = error_table["Absolute Error (hPa)"].mean()
print("\n--- Overall Performance Metric ---")
print(f"📊 Mean Absolute Error (MAE) for all data: {mean_mae:.2f} hPa")

print("\nGenerating histogram for all forecast errors...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
plt.hist(error_table["Error (hPa)"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Histogram of All Forecast Errors (Forecast MSLP - ERA5 MSLP)", fontsize=16)
plt.xlabel("Error in Minimum Sea Level Pressure (hPa)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No Error (Forecast = ERA5)')
plt.legend()
plt.tight_layout()
plt.show()

target_hours = [0, 6, 12, 18]
print("\n--- Histograms for Specific Forecast Hours (showing central 95% of data) ---")

for specific_hour in target_hours:
    hourly_data = error_table[error_table["Forecast_Hour"] == specific_hour]

    if hourly_data.empty:
        print(f"\nNo data found for forecast hour +{specific_hour}h. Skipping histogram.")
        continue

    filtered_errors = hourly_data["Error (hPa)"]
    num_samples = len(filtered_errors)
    mae_specific_hour = hourly_data["Absolute Error (hPa)"].mean()

    print(f"\nGenerating histogram for forecast +{specific_hour}h (Samples: {num_samples})...")
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_errors, bins=50, color='dodgerblue', edgecolor='black',
             alpha=0.7)  # Bins calculated on full data for this hour

    plt.title(f"Histogram of Forecast Errors for +{specific_hour}h (Forecast MSLP - ERA5 MSLP)", fontsize=16)
    plt.xlabel("Error in Minimum Sea Level Pressure (hPa)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No Error (Forecast = ERA5)')

    # Add MAE and Number of Samples text to the plot
    plt.text(0.95, 0.95, f'MAE: {mae_specific_hour:.2f} hPa\nSamples (N): {num_samples}',
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5),fontsize=14)

    # Set x-axis to display central 95% of the data
    if num_samples > 1:  # Need at least 2 data points for percentiles to define a range
        lower_bound = np.percentile(filtered_errors, 1.0)
        upper_bound = np.percentile(filtered_errors, 99)

        if lower_bound < upper_bound:
            plt.xlim(lower_bound, upper_bound)
        else:  # Handles cases where all values are the same or percentiles coincide
            # Fallback to a small range around the mean/median or the single value
            data_mean = np.mean(filtered_errors)
            # Check if all values are identical (std dev is 0)
            if np.std(filtered_errors) == 0:
                plt.xlim(data_mean - 1, data_mean + 1)  # Arbitrary small range for constant data
            else:
                # If percentiles are somehow problematic but data isn't constant (should be rare with correct percentile use)
                # Default to a slightly wider view than just a point.
                # This case might indicate very few unique values.
                plt.xlim(data_mean - (np.abs(data_mean) * 0.5 + 1), data_mean + (np.abs(data_mean) * 0.5 + 1))


    elif num_samples == 1:  # If only one sample, center plot around it
        val = filtered_errors.iloc[0]
        plt.xlim(val - 1, val + 1)  # Arbitrary small range

    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n✅ Analysis for specific forecast hours complete.")