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
# This script performs a comprehensive error analysis of a DL model against
# baselines. It now includes:
# 1. Loading and parsing comma-separated keys for multiple geophysical categories.
# 2. Calculating categorized errors for DL, Persistence, and Linear models.
# 3. Generating PDF plots of error distributions.
# 4. Aggregating categories into broader comparison groups (e.g., "Bomb" vs. "Non-Bomb").
# 5. Saving all final error arrays into a single compressed NumPy file (.npz).
# 6. Saving a comprehensive MAE report for all categories to a CSV file.
# =============================================================================


def find_prediction_files(base_dir, versions, suffix):
    """Finds all prediction files for a given list of versions and suffix."""
    file_paths = []
    print(f"  [INFO] Searching for files with suffix: '{suffix if suffix else 'None'}'", flush=True)
    for version in versions:
        if suffix is None:
            target_file = os.path.join(base_dir, f"version_{version}", "checkpoints", 'predictions.pt')
            if os.path.exists(target_file):
                correct_files = [target_file]
            else:
                alt_search_pattern = os.path.join(base_dir, f"version_{version}", "checkpoints", '*predictions.pt')
                files = glob.glob(alt_search_pattern)
                correct_files = [f for f in files if f.endswith('predictions.pt')]
        else:
            target_ending = f"predictions_{suffix}.pt"
            search_pattern = os.path.join(base_dir, f"version_{version}", "checkpoints", '*predictions*.pt')
            all_possible_files = glob.glob(search_pattern)
            correct_files = [f for f in all_possible_files if f.endswith(target_ending)]

        if correct_files:
            if len(correct_files) > 1: print(
                f"  [WARNING] Found multiple files for version {version}. Using first one: {correct_files[0]}",
                flush=True)
            file_paths.append(correct_files[0])
        else:
            print(f"    [ERROR] No prediction file found for version {version} with given criteria.", flush=True)
    return file_paths


def load_category_keys(classification_dir):
    """Loads comma-separated keys from .txt files."""
    print(f"\n--- Loading Category Keys from: {classification_dir} ---", flush=True)
    category_keys = {}
    search_pattern = os.path.join(classification_dir, 'classification_*.txt')
    txt_files = glob.glob(search_pattern)
    for f_path in txt_files:
        filename = os.path.basename(f_path)
        category_name = filename.replace('classification_', '').replace('.txt', '')
        try:
            with open(f_path, 'r') as f:
                content = f.read()
                keys = set(key.strip() for key in content.split(',') if key.strip())
                category_keys[category_name] = keys
        except Exception as e:
            print(f"  [ERROR] Could not read or parse file {filename}. Error: {e}", flush=True)
    return category_keys


def calculate_categorized_baseline_errors(ref_file_paths, mean, std, baseline_pred_base_dir, category_keys):
    """Calculates and categorizes errors for baseline models."""
    print("\n--- Calculating Categorized Baseline Errors ---", flush=True)
    errors = {'persistence': {cat: [] for cat in list(category_keys.keys()) + ['unclassified']},
              'linear': {cat: [] for cat in list(category_keys.keys()) + ['unclassified']}}
    processed_keys = set()
    for file_path in ref_file_paths:
        predictions = torch.load(file_path, map_location='cpu')
        for key, value in predictions.items():
            if key in processed_keys: continue
            processed_keys.add(key)
            _, y_true_tensor = value[0].cpu(), value[1].cpu()
            print(f"key: {key}")
            print(f"y_true_tensor[:, 0]: {y_true_tensor[:, 0]}")
            denorm_y_true = torch.exp((y_true_tensor[:, 0] * std[28] + mean[28])) / 100.0
            print(f"denorm_y_true: {denorm_y_true}")
            try:
                year = key.split('_')[1]
                p_file = os.path.join(baseline_pred_base_dir, 'naive', year, key)
                p_pred = np.load(p_file) / 100.0
                print(f"p_pred: {p_pred}")
                p_err = p_pred - denorm_y_true.numpy()
                l_file = os.path.join(baseline_pred_base_dir, 'linear', year, key)
                l_pred = np.load(l_file) / 100.0
                print(f"l_pred: {l_pred}")
                l_err = l_pred - denorm_y_true.numpy()
                classified = False
                for category, cat_keys in category_keys.items():
                    if key in cat_keys:
                        errors['persistence'][category].extend(p_err)
                        errors['linear'][category].extend(l_err)
                        classified = True
                if not classified:
                    errors['persistence']['unclassified'].extend(p_err)
                    errors['linear']['unclassified'].extend(l_err)
            except (FileNotFoundError, IndexError):
                continue
    return errors


def calculate_categorized_dl_errors(file_paths, mean, std, category_keys):
    """Calculates and categorizes raw DL errors."""
    print("\n--- Calculating Categorized DL Model Errors ---", flush=True)
    errors = {cat: [] for cat in list(category_keys.keys()) + ['unclassified']}
    for file_path in file_paths:
        predictions = torch.load(file_path, map_location='cpu')
        print(file_path)
        for key, value in predictions.items():
            y_hat, y_true = value[0].cpu(), value[1].cpu()
            print(f"y_hat: {y_hat}")
            print(f"y_true: {y_true}")
            denorm_y_hat = torch.exp((y_hat[:, 0] * std[28] + mean[28])) / 100.0
            denorm_y_true = torch.exp((y_true[:, 0] * std[28] + mean[28])) / 100.0
            print(f"denorm_y_hat: {denorm_y_hat}")
            print(f"denorm_y_true: {denorm_y_true}")
            exit()
            raw_error_dl = (denorm_y_hat - denorm_y_true).detach().numpy()
            classified = False
            for category, cat_keys in category_keys.items():
                if key in cat_keys:
                    errors[category].extend(raw_error_dl)
                    classified = True
            if not classified: errors['unclassified'].extend(raw_error_dl)
    return errors


def plot_scenario_breakdown(base_model_errors, categorized_dl_errors, categories_to_plot, title, output_file,
                            xlim_range=(-12, 12)):
    """Plots PDF error distributions for specified categories."""
    print(f"\n--- Generating PDF Plot: {title} ---", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))
    x_vals = np.linspace(xlim_range[0], xlim_range[1], 1000)
    base_styles = {"Control": {"color": "black", "lw": 4, "ls": "-", "label": "Control (All Folds)"},
                   "Persistence": {"color": "dimgray", "lw": 3.5, "ls": "--", "label": "Persistence (All)"},
                   "Linear": {"color": "darkgray", "lw": 3.5, "ls": ":", "label": "Linear (All)"}}
    for name, errors in base_model_errors.items():
        if name in base_styles and len(errors) > 1:
            style, kde = base_styles[name], st.gaussian_kde(errors)
            label = f"{style['label']} (MAE={np.mean(np.abs(errors)):.2f})"
            ax.plot(x_vals, kde(x_vals), label=label, **{k: v for k, v in style.items() if k != 'label'})
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories_to_plot)))
    for i, category in enumerate(categories_to_plot):
        errors = categorized_dl_errors.get(category)
        if errors and len(errors) > 1:
            kde, me, mae, n = st.gaussian_kde(errors), np.mean(errors), np.mean(np.abs(errors)), len(errors)
            label = f"{category} (ME={me:.2f}, MAE={mae:.2f}, N={n})"
            ax.plot(x_vals, kde(x_vals), label=label, color=colors[i], lw=2.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')
    ax.set(title=title, xlabel='Raw Error (mbar) [Prediction - True Value]', ylabel='Normalized Probability Density',
           xlim=xlim_range)
    ax.legend(fontsize='medium', loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"  [SUCCESS] PDF Plot saved to {output_file}", flush=True)
    plt.close(fig)


def save_mae_to_csv(numpy_errors, output_file):
    """Calculates MAE for all categories and saves them to a CSV file."""
    print(f"\n--- Generating and Saving MAE Report to CSV ---", flush=True)
    report_data = []

    # Get a sorted list of all categories, including aggregated ones
    all_categories = sorted(list(numpy_errors['dl'].keys()))

    for category in all_categories:
        row = {'Category': category}
        for model in ['dl', 'persistence', 'linear']:
            errors = numpy_errors[model].get(category)
            # Check if errors is a non-empty numpy array
            if errors is not None and errors.size > 0:
                mae = np.mean(np.abs(errors))
                sample_size = len(errors)
            else:
                mae = np.nan
                sample_size = 0

            row[f'{model.upper()}_MAE'] = mae
            row[f'{model.upper()}_N'] = sample_size
        report_data.append(row)

    df = pd.DataFrame(report_data)
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"  [SUCCESS] MAE report saved to {output_file}", flush=True)


if __name__ == "__main__":
    # =========================================================================
    # SCRIPT CONFIGURATION
    # =========================================================================
    print("--- Initializing Script Configuration ---", flush=True)
    MODEL_NAME = "CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast3h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"
    MODEL_VERSION_TAG = "analysis_33_log"
    DL_BASE_DIRECTORY = f"/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model/{MODEL_NAME}"
    BASELINE_PRED_BASE_DIR = f"/home/mansour/ML3300-24a/shreibshtein/predictions_naive_33"
    STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"
    CLASSIFICATION_TXT_DIR = "/home/mansour/ML3300-24a/shreibshtein/classification_txt_files"
    CONTROL_FOLDS_VERSIONS = ['25']
    PLOT_X_LIMITS = (-7, 7)

    # =========================================================================
    # STEP 1: LOAD AUXILIARY DATA & CALCULATE ERRORS
    # =========================================================================
    print("\n--- Starting Categorized Error Distribution Analysis ---", flush=True)
    try:
        mean_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-mean.pt"), map_location='cpu')
        std_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-std.pt"), map_location='cpu')
    except Exception as e:
        raise SystemExit(f"FATAL: Could not load mean/std files. Error: {e}")

    category_keys = load_category_keys(CLASSIFICATION_TXT_DIR)
    if not category_keys: raise SystemExit("FATAL: No category keys loaded.")

    # --- User-requested modification: Add SH_SA keys to SH_SO ---
    print("\n--- Applying User Modification: Merging SH_SA into SH_SO ---", flush=True)
    if 'SH_SO' in category_keys and 'SH_SA' in category_keys:
        original_so_count = len(category_keys['SH_SO'])
        category_keys['SH_SO'].update(category_keys['SH_SA'])
        new_so_count = len(category_keys['SH_SO'])
        print(f"  [INFO] Merged keys from 'SH_SA' into 'SH_SO'.")
        print(f"  [INFO] 'SH_SO' key count changed from {original_so_count} to {new_so_count}.", flush=True)
    else:
        print("  [WARNING] Could not merge 'SH_SA' into 'SH_SO' as one or both categories were not found.", flush=True)
    # --- End of modification ---

    control_files = find_prediction_files(DL_BASE_DIRECTORY, CONTROL_FOLDS_VERSIONS, suffix=None)
    if not control_files: raise SystemExit("FATAL: No control prediction files found.")

    baseline_errors_lists = calculate_categorized_baseline_errors(control_files, mean_tensor, std_tensor,
                                                                  BASELINE_PRED_BASE_DIR, category_keys)
    dl_errors_lists = calculate_categorized_dl_errors(control_files, mean_tensor, std_tensor, category_keys)

    # =========================================================================
    # STEP 2: AGGREGATE CATEGORIES & CONVERT TO NUMPY
    # =========================================================================
    print("\n--- Aggregating categories and converting to NumPy arrays ---", flush=True)
    final_numpy_errors = {'dl': {}, 'persistence': {}, 'linear': {}}

    for model in final_numpy_errors:
        source_errors = dl_errors_lists if model == 'dl' else baseline_errors_lists[model]
        for cat, errors in source_errors.items():
            final_numpy_errors[model][cat] = np.array(errors)

    aggregation_map = {
        'BOMB': ['NH_BOMB', 'SH_BOMB'], 'NBOMB': ['NH_NBOMB', 'SH_NBOMB'],
        'GL': ['NH_GL', 'SH_GL'], 'GO': ['NH_GO', 'SH_GO'],
        'MID': ['NH_MID', 'SH_MID'], 'SUB': ['NH_SUB', 'SH_SUB'],
        'NH': [cat for cat in category_keys if cat.startswith('NH_')],
        'SH': [cat for cat in category_keys if cat.startswith('SH_')]
    }

    for agg_name, source_cats in aggregation_map.items():
        for model in final_numpy_errors:
            arrays_to_concat = [final_numpy_errors[model][cat] for cat in source_cats if
                                cat in final_numpy_errors[model] and final_numpy_errors[model][cat].size > 0]
            final_numpy_errors[model][agg_name] = np.concatenate(arrays_to_concat) if arrays_to_concat else np.array([])

    print("[SUCCESS] All errors are aggregated and stored in the final data structure.", flush=True)

    # =========================================================================
    # STEP 3: GENERATE PDF PLOTS
    # =========================================================================
    base_plot_errors = {"Control": np.concatenate(list(dl_errors_lists.values())),
                        "Persistence": np.concatenate(list(baseline_errors_lists['persistence'].values())),
                        "Linear": np.concatenate(list(baseline_errors_lists['linear'].values()))}

    plot_def = {
        'NH_Seasons': [c for c in category_keys if
                       c.startswith('NH_') and c.split('_')[1] in ['DJF', 'MAM', 'JJA', 'SON']],
        'SH_Seasons': [c for c in category_keys if
                       c.startswith('SH_') and c.split('_')[1] in ['DJF', 'MAM', 'JJA', 'SON']],
        'NH_Phenomena': [c for c in category_keys if
                         c.startswith('NH_') and c.split('_')[1] not in ['DJF', 'MAM', 'JJA', 'SON']],
        'SH_Phenomena': [c for c in category_keys if
                         c.startswith('SH_') and c.split('_')[1] not in ['DJF', 'MAM', 'JJA', 'SON']]
    }
    for name, cats in plot_def.items():
        plot_scenario_breakdown(base_plot_errors, dl_errors_lists, sorted(cats),
                                title=f'Error Distribution: {name.replace("_", " ")}',
                                output_file=f'error_dist_{MODEL_VERSION_TAG}_{name}.png', xlim_range=PLOT_X_LIMITS)

    # =========================================================================
    # STEP 4: SAVE FINAL DATA TO FILES
    # =========================================================================

    # --- 4a. Save comprehensive MAE report to CSV ---
    mae_csv_filename = f'mae_report_{MODEL_VERSION_TAG}.csv'
    save_mae_to_csv(final_numpy_errors, mae_csv_filename)

    # --- 4b. Save all raw error arrays to a compressed .npz file ---
    npz_filename = f'all_categorized_errors_{MODEL_VERSION_TAG}.npz'
    print(f"\n--- Saving all error arrays to NPZ file ---", flush=True)

    # np.savez_compressed requires a flat dictionary of key-value pairs
    data_to_save = {}
    for model, model_errors in final_numpy_errors.items():
        for category, errors in model_errors.items():
            key_name = f"{model}_{category}"
            data_to_save[key_name] = errors

    np.savez_compressed(npz_filename, **data_to_save)
    print(f"  [SUCCESS] All error data saved to {npz_filename}", flush=True)
    print("\nTo load this data later, use: data = np.load('your_file.npz')", flush=True)
    print("Then access arrays like: data['dl_BOMB']", flush=True)

    print("\n--- Analysis Script End ---", flush=True)