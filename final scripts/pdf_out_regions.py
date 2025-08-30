import os
import numpy as np
import torch
import glob


# =============================================================================
# SCRIPT TO CALCULATE AND PRINT BASELINE MAE VALUES
# =============================================================================
# This script uses the provided functions to calculate the MAE for the
# Persistence and Linear baselines for each prediction offset (6h, 12h, etc.).
# It does not generate any plots; it only prints the final MAE results.
# =============================================================================

def find_prediction_files(base_dir, versions, suffix):
    """Finds all prediction files for a given list of versions and suffix."""
    file_paths = []
    for version in versions:
        base_search_path = os.path.join(base_dir, f"version_{version}", "checkpoints")
        target_ending = f"predictions_{suffix}.pt" if suffix else "predictions.pt"
        correct_files = [f for f in glob.glob(os.path.join(base_search_path, '*predictions*.pt')) if
                         f.endswith(target_ending)]
        if correct_files:
            file_paths.append(correct_files[0])
    return file_paths


def calculate_baseline_errors(ref_file_paths, mean, std, baseline_pred_base_dir):
    """Calculates raw errors for baseline models using the true values from reference files."""
    print(f"  [DEBUG] Entering 'calculate_baseline_errors' with {len(ref_file_paths)} reference file(s).")
    # This function now correctly calculates both persistence and linear errors
    baseline_errors = {"persistence": [], "linear": []}
    processed_keys = set()
    for file_path in ref_file_paths:
        try:
            predictions = torch.load(file_path, map_location='cpu')
            for key, value in predictions.items():
                if key not in processed_keys:
                    processed_keys.add(key)
                    _, y_true_tensor = value[0].cpu(), value[1].cpu()
                    denorm_y_true = (y_true_tensor[:, 0] * std[28] + mean[28]) / 100.0
                    try:
                        year = key.split('_')[1]
                        # Persistence
                        try:
                            persistence_file = os.path.join(baseline_pred_base_dir, 'naive', year, key)
                            print(f"      [DEBUG] Looking for Persistence file: {persistence_file}")
                            persistence_pred_mbar = np.load(persistence_file) / 100.0
                            baseline_errors["persistence"].extend(persistence_pred_mbar - denorm_y_true.numpy())
                            print("        [DEBUG] -> SUCCESS.")
                        except FileNotFoundError:
                            print("        [DEBUG] -> FAILED: File not found.")
                        # Linear
                        try:
                            linear_file = os.path.join(baseline_pred_base_dir, 'linear', year, key)
                            print(f"      [DEBUG] Looking for Linear file: {linear_file}")
                            linear_pred_mbar = np.load(linear_file) / 100.0
                            baseline_errors["linear"].extend(linear_pred_mbar - denorm_y_true.numpy())
                            print("        [DEBUG] -> SUCCESS.")
                        except FileNotFoundError:
                            print("        [DEBUG] -> FAILED: File not found.")
                    except IndexError:
                        print(f"      [Baseline Warning] Could not extract year from key '{key}'.")
        except Exception as e:
            print(f"      [File Error] Could not load or process file {file_path}: {e}")
    return baseline_errors


if __name__ == "__main__":
    # =========================================================================
    # SCRIPT CONFIGURATION
    # =========================================================================
    MODEL_BASE_DIR_ROOT = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model"
    BASELINE_BASE_DIR_ROOT = "/home/mansour/ML3300-24a/shreibshtein"
    STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"

    MODEL_NAME_PREFIX = "CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-"
    MODEL_NAME_SUFFIX = "-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"

    OFFSET_SCENARIOS = [
        {"offset": 6, "forecast_tag": "forecast1h", "baseline_tag": "31",
         "oos_regions": {"NA": "53", "NP": "52", "SA": "62", "MED": "1", "SO": "63"}},
        {"offset": 12, "forecast_tag": "forecast2h", "baseline_tag": "32",
         "oos_regions": {"NA": "16", "NP": "21", "SA": "26", "MED": "22", "SO": "25"}},
        {"offset": 18, "forecast_tag": "forecast3h", "baseline_tag": "33",
         "oos_regions": {"NA": "20", "NP": "21", "SA": "23", "MED": "22", "SO": "24"}},
        {"offset": 24, "forecast_tag": "forecast4h", "baseline_tag": "34",
         "oos_regions": {"NA": "17", "NP": "18", "SA": "22", "MED": "19"}},
    ]

    # =========================================================================
    # SCRIPT EXECUTION
    # =========================================================================
    print("--- Starting Baseline MAE Calculation ---")

    # Dictionary to store the final MAE results
    final_baseline_maes = {"Persistence": [], "Linear": []}
    offsets_processed = []

    for scenario in OFFSET_SCENARIOS:
        offset = scenario["offset"]
        model_name = f"{MODEL_NAME_PREFIX}{scenario['forecast_tag']}{MODEL_NAME_SUFFIX}"
        dl_model_dir = os.path.join(MODEL_BASE_DIR_ROOT, model_name)
        baseline_pred_dir = os.path.join(BASELINE_BASE_DIR_ROOT, f"predictions_naive_{scenario['baseline_tag']}")

        print(f"\n\n{'=' * 60}\n--- Processing Offset: {offset}h ---\n{'=' * 60}")
        offsets_processed.append(f"{offset}h")

        try:
            mean_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{model_name}-mean.pt"), map_location='cpu')
            std_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{model_name}-std.pt"), map_location='cpu')
        except Exception as e:
            print(f"  FATAL: Could not load mean/std for {model_name}. Skipping offset. Error: {e}")
            final_baseline_maes["Persistence"].append(np.nan)
            final_baseline_maes["Linear"].append(np.nan)
            continue

        all_oos_versions = list(scenario["oos_regions"].values())
        all_dl_files = find_prediction_files(dl_model_dir, all_oos_versions, suffix=None)

        if not all_dl_files:
            print(f"  CRITICAL: No out-of-sample files found for offset {offset}h. Skipping.")
            final_baseline_maes["Persistence"].append(np.nan)
            final_baseline_maes["Linear"].append(np.nan)
            continue

        # --- 1. Calculate Baseline raw errors ---
        print("\n--- Calculating Baseline Errors ---")
        baseline_error_data = calculate_baseline_errors(all_dl_files, mean_tensor, std_tensor, baseline_pred_dir)

        # --- 2. Calculate MAE from raw errors and store them ---
        raw_persistence_errors = baseline_error_data.get("persistence", [])
        persistence_mae = np.mean(np.abs(raw_persistence_errors)) if raw_persistence_errors else np.nan
        final_baseline_maes["Persistence"].append(persistence_mae)

        raw_linear_errors = baseline_error_data.get("linear", [])
        linear_mae = np.mean(np.abs(raw_linear_errors)) if raw_linear_errors else np.nan
        final_baseline_maes["Linear"].append(linear_mae)

    # --- 3. Print the final results ---
    print("\n\n" + "=" * 60)
    print("--- ✅ FINAL BASELINE MAE RESULTS ---")
    print("=" * 60)
    print(f"Offsets:         {', '.join(offsets_processed)}")
    print(f"Persistence MAE: {np.round(final_baseline_maes['Persistence'], 4).tolist()}")
    print(f"Linear MAE:      {np.round(final_baseline_maes['Linear'], 4).tolist()}")
    print("=" * 60)

    print("\n--- Analysis Script End ---")