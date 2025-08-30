import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import torch
import glob

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
# ▼▼▼ V V V SET YOUR PATHS HERE V V V ▼▼▼

# 1. SET THE REFERENCE DL MODEL FILES
# ---
# These files are used to get the ground truth (`y_true`) and the list of data keys.
# We'll use the same aggregated folds as before.
REFERENCE_MODEL_VERSIONS = ['45', '46', '47', '48']

# 2. SET THE PATHS TO YOUR DATA DIRECTORIES
# ---
MODEL_NAME = "CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast1h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"
DL_BASE_DIRECTORY = f"/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model/{MODEL_NAME}"
BASELINE_PRED_BASE_DIR = f"/home/mansour/ML3300-24a/shreibshtein/predictions_naive_31"
STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"

# 3. SET THE OUTPUT FILENAME FOR THE DEBUG PLOT
# ---
OUTPUT_FILENAME = 'baseline_debug_plot.png'


# ▲▲▲ A A A YOUR SETTINGS END HERE A A A ▲▲▲
# =============================================================================


def find_prediction_files(base_dir, versions, suffix):
    """Finds all prediction files for a given list of versions and suffix."""
    file_paths = []
    print(f"  [INFO] Searching for files with suffix: '{suffix if suffix else 'None'}'", flush=True)
    for version in versions:
        print(f"    [DEBUG] Searching in version_{version}...", flush=True)
        # This logic is refined to correctly find 'predictions.pt' when suffix is None
        if suffix is None:
            target_file = os.path.join(base_dir, f"version_{version}", "checkpoints", 'predictions.pt')
            print(f"      [DEBUG] Attempting to find exact file: {target_file}", flush=True)
            if os.path.exists(target_file):
                correct_files = [target_file]
            else:  # Fallback for older naming conventions
                alt_search_pattern = os.path.join(base_dir, f"version_{version}", "checkpoints", '*predictions.pt')
                print(f"      [DEBUG] Exact file not found. Using glob pattern: {alt_search_pattern}", flush=True)
                files = glob.glob(alt_search_pattern)
                correct_files = [f for f in files if f.endswith('predictions.pt')]
        else:
            target_ending = f"predictions_{suffix}.pt"
            search_pattern = os.path.join(base_dir, f"version_{version}", "checkpoints", '*predictions*.pt')
            print(
                f"      [DEBUG] Using glob pattern: {search_pattern} and filtering for files ending with '{target_ending}'",
                flush=True)
            all_possible_files = glob.glob(search_pattern)
            correct_files = [f for f in all_possible_files if f.endswith(target_ending)]

        if correct_files:
            if len(correct_files) > 1:
                print(f"  [WARNING] Found multiple files for version {version}. Using first one: {correct_files[0]}",
                      flush=True)
            file_paths.append(correct_files[0])
            print(f"    [INFO] Found for version {version}: {correct_files[0]}", flush=True)
        else:
            print(f"    [ERROR] No prediction file found for version {version} with given criteria.", flush=True)

    print(f"  [DEBUG] find_prediction_files finished. Found {len(file_paths)} total files.", flush=True)
    return file_paths


def calculate_and_diagnose_baselines(ref_file_paths, mean, std, baseline_pred_base_dir):
    """
    Calculates baseline errors and prints detailed diagnostics.
    """
    print("\n--- Calculating Baseline Errors (with diagnostics) ---", flush=True)
    baseline_errors = {"persistence": [], "linear": []}
    processed_keys = set()

    # --- DIAGNOSTIC COUNTERS ---
    success_count = 0
    fail_count = 0

    for file_path in ref_file_paths:
        print(f"\nProcessing reference file: {os.path.basename(file_path)}", flush=True)
        predictions = torch.load(file_path, map_location='cpu')
        for key, value in predictions.items():
            if key in processed_keys:
                continue
            processed_keys.add(key)

            # This is the ground truth value
            _, y_true_tensor = value[0].cpu(), value[1].cpu()
            denorm_y_true = (y_true_tensor[:, 0] * std[28] + mean[28]) / 100.0

            try:
                year = key.split('_')[1]
                # Try to load persistence file
                persistence_file = os.path.join(baseline_pred_base_dir, 'naive', year, key)
                persistence_pred_mbar = np.load(persistence_file) / 100.0

                # Try to load linear file
                linear_file = os.path.join(baseline_pred_base_dir, 'linear', year, key)
                linear_pred_mbar = np.load(linear_file) / 100.0

                # If we reach here, both files were found
                success_count += 1

                # Calculate and store errors
                baseline_errors["persistence"].extend(persistence_pred_mbar - denorm_y_true.numpy())
                baseline_errors["linear"].extend(linear_pred_mbar - denorm_y_true.numpy())

            except FileNotFoundError as e:
                # If either file is not found, we land here
                fail_count += 1
                if fail_count < 5:  # Print the first few failures
                    print(f"  [FAIL] Could not find baseline file for key '{key}'. Missing file: {e.filename}",
                          flush=True)
                continue
            except IndexError:
                # This catches errors if the key format is unexpected
                fail_count += 1
                if fail_count < 5:
                    print(f"  [FAIL] Could not parse year from key '{key}'.", flush=True)
                continue

    # --- Print Final Diagnostics ---
    print("\n--- Baseline Calculation Diagnostics ---", flush=True)
    print(f"Total Unique Keys in Reference Files: {len(processed_keys)}", flush=True)
    print(f"✅ Baseline File Pairs Found (Success): {success_count}", flush=True)
    print(f"❌ Baseline File Pairs Not Found (Skipped): {fail_count}", flush=True)
    print("-" * 40, flush=True)

    return baseline_errors


def plot_baseline_distributions(error_data, output_file):
    """
    Generates a simple PDF plot for the baseline errors.
    """
    print("\n📈 Generating baseline distribution plot...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 9))
    xlim_range = (-12, 12)
    x_vals = np.linspace(xlim_range[0], xlim_range[1], 1000)

    styles = {
        "persistence": {"label": "Persistence Baseline", "color": "green", "ls": "--"},
        "linear": {"label": "Linear Baseline", "color": "purple", "ls": ":"}
    }

    for name, errors in error_data.items():
        if len(errors) > 1:
            style = styles.get(name, {})
            kde = st.gaussian_kde(errors)

            # Calculate stats for the legend
            mean_error = np.mean(errors)
            mae = np.mean(np.abs(errors))
            count = len(errors)
            label = f"{style['label']} (MAE={mae:.2f}, ME={mean_error:.2f}, N={count})"

            ax.plot(x_vals, kde(x_vals), label=label, lw=3, **{k: v for k, v in style.items() if k != 'label'})
        else:
            print(f"  ⚠️ Skipping '{name}' plot as it contains insufficient data.")

    # --- Formatting ---
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
    ax.set_title('Baseline Error Distribution (Debug)', fontsize=18, weight='bold')
    ax.set_xlabel('Raw Error (mbar) [Prediction - True Value]', fontsize=14)
    ax.set_ylabel('Normalized Probability Density', fontsize=14)
    ax.set_xlim(xlim_range)
    ax.legend(fontsize='large', loc='upper right')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ Plot saved successfully to: {output_file}")
    plt.show()


if __name__ == "__main__":
    print("--- Starting Baseline Debugging Script ---", flush=True)

    # 1. Load auxiliary stat files
    try:
        mean_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-mean.pt"), map_location='cpu')
        std_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-std.pt"), map_location='cpu')
        print("✅ Successfully loaded mean and std files.", flush=True)
    except Exception as e:
        raise SystemExit(f"FATAL: Could not load mean/std files. Error: {e}")

    # 2. Find the reference prediction files
    reference_files = find_prediction_files(DL_BASE_DIRECTORY, REFERENCE_MODEL_VERSIONS, suffix=None)
    if not reference_files:
        raise SystemExit("FATAL: No reference prediction files found. Cannot proceed.")

    # 3. Calculate baseline errors and get diagnostics
    baseline_errors = calculate_and_diagnose_baselines(
        reference_files, mean_tensor, std_tensor, BASELINE_PRED_BASE_DIR
    )

    # 4. Generate the plot
    if baseline_errors["persistence"] and baseline_errors["linear"]:
        plot_baseline_distributions(baseline_errors, OUTPUT_FILENAME)
    else:
        print("\n❌ Analysis failed: No baseline data was successfully processed.")

    print("\n--- Debug Script End ---", flush=True)
