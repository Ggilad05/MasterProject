import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import torch
import glob
import pandas as pd


# =============================================================================
# SCRIPT OVERALL EXPLANATION
# =============================================================================
# This script analyzes error distributions by first calculating errors for
# baseline models (Persistence, Linear) and then for the DL model.
# It categorizes errors based on predefined text files, calculates MAE for
# each category, and generates separate PDF plots for different category groups.
# This version includes enhanced debugging output and flushes all prints.
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


def load_category_keys(classification_dir):
    """
    Loads all classification keys from .txt files in a directory.
    This version correctly handles files where keys are on a single line, separated by commas.
    """
    print(f"\n--- Loading Category Keys from: {classification_dir} ---", flush=True)
    category_keys = {}
    search_pattern = os.path.join(classification_dir, 'classification_*.txt')
    txt_files = glob.glob(search_pattern)
    print(f"  [DEBUG] Found {len(txt_files)} classification files with pattern: {search_pattern}", flush=True)

    for f_path in txt_files:
        filename = os.path.basename(f_path)
        category_name = filename.replace('classification_', '').replace('.txt', '')
        try:
            with open(f_path, 'r') as f:
                # Read the entire file content into one string
                content = f.read()
                # Split the string by commas, then strip whitespace from each resulting key.
                # Filter out any empty strings that might result from trailing commas or empty files.
                keys = set(key.strip() for key in content.split(',') if key.strip())
                category_keys[category_name] = keys
                print(f"  [INFO] Loaded {len(keys)} keys for category '{category_name}'", flush=True)
        except Exception as e:
            print(f"  [ERROR] Could not read or parse file {filename}. Error: {e}", flush=True)

    if not category_keys:
        print("  [WARNING] No category files found.", flush=True)
    else:
        print(f"  [DEBUG] Finished loading keys. Total categories found: {len(category_keys)}", flush=True)
    return category_keys


def calculate_categorized_baseline_errors(ref_file_paths, mean, std, baseline_pred_base_dir, category_keys):
    """Calculates and categorizes errors for baseline models."""
    print("\n--- Calculating Categorized Baseline Errors ---", flush=True)

    errors = {
        'persistence': {cat: [] for cat in list(category_keys.keys()) + ['unclassified']},
        'linear': {cat: [] for cat in list(category_keys.keys()) + ['unclassified']}
    }
    processed_keys = set()
    key_counter = 0

    for file_path in ref_file_paths:
        print(f"  [INFO] Processing file for baselines: {os.path.basename(file_path)}", flush=True)
        predictions = torch.load(file_path, map_location='cpu')
        for key in predictions:
            if key in processed_keys:
                continue
            processed_keys.add(key)
            key_counter += 1
            if key_counter % 500 == 0:
                print(f"    [DEBUG] Processed {key_counter} unique keys for baselines...", flush=True)

            _, y_true_tensor = predictions[key][0].cpu(), predictions[key][1].cpu()
            denorm_y_true = (y_true_tensor[:, 0] * std[28] + mean[28]) / 100.0

            try:
                year = key.split('_')[1]
                persistence_file = os.path.join(baseline_pred_base_dir, 'naive', year, key)
                persistence_pred_mbar = np.load(persistence_file) / 100.0
                raw_error_persistence = persistence_pred_mbar - denorm_y_true.numpy()
                linear_file = os.path.join(baseline_pred_base_dir, 'linear', year, key)
                linear_pred_mbar = np.load(linear_file) / 100.0
                raw_error_linear = linear_pred_mbar - denorm_y_true.numpy()

                classified = False
                for category, cat_keys in category_keys.items():
                    if key in cat_keys:
                        errors['persistence'][category].extend(raw_error_persistence)
                        errors['linear'][category].extend(raw_error_linear)
                        classified = True
                if not classified:
                    errors['persistence']['unclassified'].extend(raw_error_persistence)
                    errors['linear']['unclassified'].extend(raw_error_linear)
            except (FileNotFoundError, IndexError):
                continue

    total_p = sum(len(v) for v in errors['persistence'].values())
    total_l = sum(len(v) for v in errors['linear'].values())
    print(f"  [DEBUG] Baseline calculation finished. Processed {key_counter} unique keys.", flush=True)
    print(f"    [DEBUG] Total persistence errors categorized: {total_p}", flush=True)
    print(f"    [DEBUG] Total linear errors categorized: {total_l}", flush=True)
    return errors


def calculate_categorized_dl_errors(file_paths, mean, std, category_keys):
    """Calculates and categorizes raw DL errors from prediction files."""
    print("\n--- Calculating Categorized DL Model Errors ---", flush=True)

    errors = {cat: [] for cat in list(category_keys.keys()) + ['unclassified']}
    key_counter = 0

    for file_path in file_paths:
        print(f"  [INFO] Processing DL file: {os.path.basename(file_path)}", flush=True)
        predictions = torch.load(file_path, map_location='cpu')
        for key, value in predictions.items():
            key_counter += 1
            if key_counter % 500 == 0:
                print(f"    [DEBUG] Processed {key_counter} DL predictions...", flush=True)

            y_hat_tensor, y_true_tensor = value[0].cpu(), value[1].cpu()
            denorm_y_hat = (y_hat_tensor[:, 0] * std[28] + mean[28]) / 100.0
            denorm_y_true = (y_true_tensor[:, 0] * std[28] + mean[28]) / 100.0
            raw_error_dl = (denorm_y_hat - denorm_y_true).detach().numpy()

            classified = False
            for category, cat_keys in category_keys.items():
                if key in cat_keys:
                    errors[category].extend(raw_error_dl)
                    classified = True
            if not classified:
                errors['unclassified'].extend(raw_error_dl)

    total_e = sum(len(v) for v in errors.values())
    print(f"  [DEBUG] DL error calculation finished. Processed {key_counter} predictions.", flush=True)
    print(f"    [DEBUG] Total DL errors categorized: {total_e}", flush=True)
    return errors


def print_mae_report(dl_errors, baseline_errors):
    """Calculates and prints ME and MAE for each category."""
    print("\n--- Model Error Report by Category (ME / MAE) ---", flush=True)

    report_data = []
    all_categories = sorted(list(dl_errors.keys()))

    for category in all_categories:
        row = {'Category': category}
        dl_errs = dl_errors.get(category, [])
        if dl_errs:
            row[f'DL (N={len(dl_errs)})'] = f'{np.mean(dl_errs):.2f} / {np.mean(np.abs(dl_errs)):.2f}'
        else:
            row['DL'] = 'N/A'
        for model_name in ['persistence', 'linear']:
            base_errs = baseline_errors[model_name].get(category, [])
            if base_errs:
                row[
                    f'{model_name.capitalize()} (N={len(base_errs)})'] = f'{np.mean(base_errs):.2f} / {np.mean(np.abs(base_errs)):.2f}'
            else:
                row[f'{model_name.capitalize()}'] = 'N/A'
        report_data.append(row)

    df = pd.DataFrame(report_data)
    print(df.to_string(index=False), flush=True)


def plot_scenario_breakdown(base_model_errors, categorized_dl_errors, categories_to_plot, title, output_file,
                            xlim_range=(-12, 12)):
    """Plots error distributions for specified categories against overall performance."""
    print(f"\n--- Generating plot: {title} ---", flush=True)
    print(f"  [DEBUG] Output file will be: {output_file}", flush=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 10))
    x_vals = np.linspace(xlim_range[0], xlim_range[1], 1000)
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories_to_plot)))

    base_styles = {
        "Control": {"color": "black", "lw": 4, "ls": "-", "label": "Control (All Folds)"},
        "Persistence": {"color": "dimgray", "lw": 3.5, "ls": "--", "label": "Persistence (All)"},
        "Linear": {"color": "darkgray", "lw": 3.5, "ls": ":", "label": "Linear (All)"}
    }

    print("  [DEBUG] Plotting base model distributions (overall performance)...", flush=True)
    for name, errors in base_model_errors.items():
        if name in base_styles and len(errors) > 1:
            print(f"    [DEBUG] Plotting '{name}' with {len(errors)} data points.", flush=True)
            style = base_styles[name]
            kde = st.gaussian_kde(errors)
            mae = np.mean(np.abs(errors))
            label = f"{style['label']} (MAE={mae:.2f})"
            plt.plot(x_vals, kde(x_vals), label=label, **{k: v for k, v in style.items() if k != 'label'})

    print("  [DEBUG] Plotting categorized DL model distributions...", flush=True)
    for i, category in enumerate(categories_to_plot):
        errors = categorized_dl_errors.get(category)
        if errors and len(errors) > 1:
            print(f"    [DEBUG] Plotting category '{category}' with {len(errors)} data points.", flush=True)
            kde = st.gaussian_kde(errors)
            mean_error = np.mean(errors)
            mae = np.mean(np.abs(errors))
            count = len(errors)
            label = f"{category} (ME={mean_error:.2f}, MAE={mae:.2f}, N={count})"
            plt.plot(x_vals, kde(x_vals), label=label, color=colors[i], lw=2.5)
        else:
            print(
                f"    [DEBUG] Skipping category '{category}' due to insufficient data (Points: {len(errors) if errors else 0}).",
                flush=True)

    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel('Raw Error (mbar) [Prediction - True Value]', fontsize=16)
    plt.ylabel('Normalized Probability Density', fontsize=16)
    plt.xlim(xlim_range)
    plt.legend(fontsize='medium', loc='upper right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"  [SUCCESS] Plot saved to {output_file}", flush=True)
    plt.close()


if __name__ == "__main__":
    # =========================================================================
    # SCRIPT CONFIGURATION
    # =========================================================================
    print("--- Initializing Script Configuration ---", flush=True)
    MODEL_NAME = "CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast1h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"
    MODEL_VERSION_TAG = "categorized_analysis_v4_fixed"
    DL_BASE_DIRECTORY = f"/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model/{MODEL_NAME}"
    BASELINE_PRED_BASE_DIR = f"/home/mansour/ML3300-24a/shreibshtein/predictions_naive_31"
    STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"
    CLASSIFICATION_TXT_DIR = "/home/mansour/ML3300-24a/shreibshtein/classification_txt_files"

    CONTROL_FOLDS_VERSIONS = ['45', '46', '47', '48']
    PLOT_X_LIMITS = (-15, 15)
    print(f"[DEBUG] MODEL_NAME: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] DL_BASE_DIRECTORY: {DL_BASE_DIRECTORY}", flush=True)
    print(f"[DEBUG] BASELINE_PRED_BASE_DIR: {BASELINE_PRED_BASE_DIR}", flush=True)
    print(f"[DEBUG] CLASSIFICATION_TXT_DIR: {CLASSIFICATION_TXT_DIR}", flush=True)
    print(f"[DEBUG] CONTROL_FOLDS_VERSIONS: {CONTROL_FOLDS_VERSIONS}", flush=True)

    # =========================================================================
    # SCRIPT EXECUTION
    # =========================================================================
    print("\n--- Starting Categorized Error Distribution Analysis ---", flush=True)

    # 1. Load auxiliary files
    try:
        mean_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-mean.pt"), map_location='cpu')
        std_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-std.pt"), map_location='cpu')
        print("[SUCCESS] Successfully loaded mean and std files.", flush=True)
    except Exception as e:
        raise SystemExit(f"FATAL: Could not load mean/std files. Error: {e}")

    category_keys = load_category_keys(CLASSIFICATION_TXT_DIR)
    if not category_keys:
        raise SystemExit("FATAL: No category keys were loaded. Cannot proceed.")

    # 2. Find prediction files
    control_files = find_prediction_files(DL_BASE_DIRECTORY, CONTROL_FOLDS_VERSIONS, suffix=None)
    if not control_files:
        raise SystemExit("FATAL: No control prediction files found. Cannot proceed.")

    # 3. Calculate errors in the desired order
    baseline_categorized_errors = calculate_categorized_baseline_errors(
        control_files, mean_tensor, std_tensor, BASELINE_PRED_BASE_DIR, category_keys
    )
    dl_categorized_errors = calculate_categorized_dl_errors(
        control_files, mean_tensor, std_tensor, category_keys
    )

    # 4. Calculate and print MAE report
    print_mae_report(dl_categorized_errors, baseline_categorized_errors)

    # 5. Prepare data for plotting
    print("\n--- Aggregating data for plotting ---", flush=True)
    base_plot_errors = {
        "Control": sum(dl_categorized_errors.values(), []),
        "Persistence": sum(baseline_categorized_errors['persistence'].values(), []),
        "Linear": sum(baseline_categorized_errors['linear'].values(), []),
    }
    print(f"  [DEBUG] Aggregated Control errors: {len(base_plot_errors['Control'])} points", flush=True)
    print(f"  [DEBUG] Aggregated Persistence errors: {len(base_plot_errors['Persistence'])} points", flush=True)
    print(f"  [DEBUG] Aggregated Linear errors: {len(base_plot_errors['Linear'])} points", flush=True)

    # Define the groups of categories for each plot
    print("\n--- Defining Category Groups for Plots ---", flush=True)
    all_loaded_categories = category_keys.keys()
    nh_seasons = sorted([cat for cat in all_loaded_categories if
                         cat.startswith('NH_') and cat.split('_')[1] in ['DJF', 'MAM', 'JJA', 'SON']])
    sh_seasons = sorted([cat for cat in all_loaded_categories if
                         cat.startswith('SH_') and cat.split('_')[1] in ['DJF', 'MAM', 'JJA', 'SON']])
    nh_places = sorted([cat for cat in all_loaded_categories if
                        cat.startswith('NH_') and cat.split('_')[1] not in ['DJF', 'MAM', 'JJA', 'SON']])
    sh_places = sorted([cat for cat in all_loaded_categories if
                        cat.startswith('SH_') and cat.split('_')[1] not in ['DJF', 'MAM', 'JJA', 'SON']])
    print(f"  [DEBUG] NH Seasons categories: {nh_seasons}", flush=True)
    print(f"  [DEBUG] SH Seasons categories: {sh_seasons}", flush=True)
    print(f"  [DEBUG] NH Phenomena categories: {nh_places}", flush=True)
    print(f"  [DEBUG] SH Phenomena categories: {sh_places}", flush=True)

    # 6. Generate and save the plots
    plot_scenario_breakdown(base_plot_errors, dl_categorized_errors, nh_seasons,
                            title='Error Distribution: Northern Hemisphere Seasons',
                            output_file=f'error_dist_{MODEL_VERSION_TAG}_NH_Seasons.png', xlim_range=PLOT_X_LIMITS)

    plot_scenario_breakdown(base_plot_errors, dl_categorized_errors, sh_seasons,
                            title='Error Distribution: Southern Hemisphere Seasons',
                            output_file=f'error_dist_{MODEL_VERSION_TAG}_SH_Seasons.png', xlim_range=PLOT_X_LIMITS)

    plot_scenario_breakdown(base_plot_errors, dl_categorized_errors, nh_places,
                            title='Error Distribution: Northern Hemisphere Phenomena',
                            output_file=f'error_dist_{MODEL_VERSION_TAG}_NH_Phenomena.png', xlim_range=PLOT_X_LIMITS)

    plot_scenario_breakdown(base_plot_errors, dl_categorized_errors, sh_places,
                            title='Error Distribution: Southern Hemisphere Phenomena',
                            output_file=f'error_dist_{MODEL_VERSION_TAG}_SH_Phenomena.png', xlim_range=PLOT_X_LIMITS)

    print("\n--- Analysis Script End ---", flush=True)