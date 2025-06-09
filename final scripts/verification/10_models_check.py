# Overall Script Explanation:
# This script is designed to perform a comprehensive raw error analysis of Deep Learning (DL) model predictions
# across multiple trained versions.
# Key functions:
# - Loads and denormalizes DL predictions (using mean/std files, target variable at index 28).
# - Calculates raw errors (Prediction - True Value) in mbar.
# - Aggregates errors per version and globally.
# - Computes metrics: Mean Error (ME), standard deviation of raw errors, counts.
# - Generates plots:
#   1. GRAPH1: Overall Raw Error Distribution (All DL versions combined as a histogram).
#   2. GRAPH2: Per-Version Raw Error Comparison (KDE for each DL version on the same plot
#      for direct comparison of distributions).
# - Prints a detailed summary of raw error metrics for each version.

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st  # For KDE plots
import pickle

NAME ="CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast1h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"


# import traceback # Optional


def calculate_prediction_difference(base_dir, versions_to_analyze, stats_dir):
    """
    Calculates and aggregates prediction differences (raw errors: Prediction - True Value)
    for specified Deep Learning (DL) model versions.
    """
    print(f"Analyzing predictions in DL base directory: {base_dir}")
    print(f"Selected DL versions for analysis: {versions_to_analyze}")
    print(f"Loading stats from: {stats_dir}\n")

    results = {}
    all_raw_diffs_dl = []

    mean_file = os.path.join(stats_dir,
                             f"CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast2h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8-mean.pt")
    std_file = os.path.join(stats_dir,
                            f"CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast2h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8-std.pt")
    mean = None
    std = None
    try:
        mean = torch.load(mean_file).cpu()
        std = torch.load(std_file).cpu()
        print("Successfully loaded mean and std deviation files (moved to CPU).")
        if not (isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor)):
            print("  Warning: Loaded mean or std is not a torch.Tensor. Denormalization will fail.")
            mean, std = None, None
        elif mean.shape[0] <= 28 or std.shape[0] <= 28:
            print(
                f"  Warning: Mean (shape {mean.shape}) or std (shape {std.shape}) tensor does not have index 28. Check structure.")
    except FileNotFoundError as e:
        print(f"Error loading mean/std files: {e}. DL denormalization will be skipped.")
        mean, std = None, None
    except Exception as e:
        print(f"An unexpected error occurred loading mean/std: {e}. DL denormalization will be skipped.")
        mean, std = None, None

    for version in versions_to_analyze:
        version_dir = os.path.join(base_dir, f"version_{version}")
        checkpoints_dir = os.path.join(version_dir, "checkpoints")
        version_str_key = f"version_{version}"
        results[version_str_key] = {}

        print(f"\nProcessing DL {version_str_key}...")

        if not os.path.exists(checkpoints_dir):
            print(f"  Error: Checkpoints directory not found: {checkpoints_dir}")
            results[version_str_key]["error"] = "Checkpoints directory not found"
            continue

        predictions_file = None
        try:
            pt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
            if pt_files:
                predictions_file = os.path.join(checkpoints_dir, pt_files[0])
                if len(pt_files) > 1: print(f"  Warning: Multiple .pt files found. Using: {pt_files[0]}")
                print(f"  Found predictions file: {predictions_file}")
            else:
                print(f"  Error: No .pt file found in {checkpoints_dir}")
                results[version_str_key]["error"] = "No .pt file found"
                continue
        except Exception as e:
            print(f"  Error listing files in checkpoints directory: {e}")
            results[version_str_key]["error"] = f"Error listing files: {e}"
            continue

        try:
            predictions = torch.load(predictions_file, map_location='cpu')

            if not isinstance(predictions, dict):
                print(f"  Error: Predictions file did not load into a dictionary. Skipping.")
                results[version_str_key]["error"] = "Predictions file not a dictionary"
                continue

            version_raw_errors_dl = []
            print(f"  Iterating through {len(predictions)} keys in predictions file...")
            num_keys_processed_for_version = 0
            num_keys_skipped_for_version = 0

            for key, value in predictions.items():
                if not (isinstance(value, (list, tuple)) and len(value) >= 2):
                    num_keys_skipped_for_version += 1
                    continue

                y_hat_dl_tensor, y_true_dl_tensor = value[0], value[1]
                if not (isinstance(y_hat_dl_tensor, torch.Tensor) and isinstance(y_true_dl_tensor, torch.Tensor)):
                    num_keys_skipped_for_version += 1
                    continue
                y_hat_dl_tensor, y_true_dl_tensor = y_hat_dl_tensor.cpu(), y_true_dl_tensor.cpu()

                if mean is not None and std is not None:
                    try:
                        if y_hat_dl_tensor.ndim < 2 or y_true_dl_tensor.ndim < 2 or \
                                y_hat_dl_tensor.shape[1] < 1 or y_true_dl_tensor.shape[1] < 1:
                            raise ValueError("DL tensor dims insufficient for slicing [:, 0]")

                        denorm_y_hat_dl = (y_hat_dl_tensor[:, 0] * std[28] + mean[28])
                        denorm_y_true_dl = (y_true_dl_tensor[:, 0] * std[28] + mean[28])

                        denorm_y_hat_mbar_dl = (denorm_y_hat_dl / 100.0).detach().numpy()
                        denorm_y_true_mbar_dl = (denorm_y_true_dl / 100.0).detach().numpy()

                        raw_error_dl = denorm_y_hat_mbar_dl - denorm_y_true_mbar_dl

                        if not np.all(np.isfinite(raw_error_dl)):
                            raise ValueError("Non-finite values in DL raw error calculation")

                        version_raw_errors_dl.extend(raw_error_dl)
                        all_raw_diffs_dl.extend(raw_error_dl)
                        num_keys_processed_for_version += 1
                    except Exception as e:
                        num_keys_skipped_for_version += 1
                        continue
                else:
                    if num_keys_processed_for_version == 0 and num_keys_skipped_for_version == 0:
                        print(f"  Skipping all DL processing for {version_str_key} due to missing mean/std files.")
                    num_keys_skipped_for_version = len(predictions)
                    break

            print(
                f"  Finished iterating keys for {version_str_key}. Processed: {num_keys_processed_for_version}, Skipped: {num_keys_skipped_for_version}")
            results[version_str_key]["metrics"] = {}

            if version_raw_errors_dl:
                results[version_str_key]["metrics"]["dl"] = {
                    "me": np.mean(version_raw_errors_dl),
                    "mae": np.mean(np.abs(version_raw_errors_dl)),
                    "std_raw": np.std(version_raw_errors_dl),
                    "count": len(version_raw_errors_dl),
                    "raw_errors_version": np.array(version_raw_errors_dl)
                }
            else:
                results[version_str_key]["metrics"]["dl"] = {"error": "No valid DL data"}

        except Exception as e:
            print(f"  Major error processing {version_str_key}, predictions file {predictions_file}: {e}")
            results[version_str_key]["error"] = f"File processing error: {str(e)}"
            continue

    return (results, all_raw_diffs_dl)


def plot_overall_raw_error_distribution(
        dl_raw_errors,
        output_path="overall_raw_error_distribution.png",
        bins=100, xlim_range=(-15, 15)
):
    """
    Plots the overall raw error distribution for all DL models combined.
    DL model raw errors (aggregated) are shown as a histogram.
    """
    print(f"\nGenerating overall raw error distribution plot to {output_path}...")
    plt.figure(figsize=(12, 7))
    alpha_hist = 0.7

    dl_raw_np = np.array(dl_raw_errors)
    if dl_raw_np.size > 0:
        me_dl = np.mean(dl_raw_np)
        std_dl = np.std(dl_raw_np)
        label_dl = f"All DL Versions (ME: {me_dl:.2f}, Std: {std_dl:.2f}, N: {dl_raw_np.size})"
        dl_to_plot = dl_raw_np[(dl_raw_np >= xlim_range[0]) & (dl_raw_np <= xlim_range[1])]
        if dl_to_plot.size > 0:
            plt.hist(dl_to_plot, bins=bins, range=xlim_range, alpha=alpha_hist, label=label_dl, density=True,
                     color='skyblue', edgecolor='black')
        else:
            print(f"  No DL data within range {xlim_range} to plot histogram.")
    else:
        print("  No DL raw error data for overall distribution plot.")

    plt.title('Overall DL Raw Error Distribution (Pred - True)')
    plt.xlabel('Raw Error (mbar)')
    plt.ylabel('Density')
    plt.axvline(0, color='k', linestyle='--', linewidth=0.8, label='Zero Error')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.5)
    plt.xlim(xlim_range)
    try:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Overall raw error distribution plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving overall distribution plot: {e}")
    plt.close()


def plot_dl_version_kde_comparison(
        analysis_results,
        sorted_version_keys,
        output_path="dl_versions_kdes_comparison.png",
        xlim_range=(-15, 15)
):
    """
    Plots Kernel Density Estimates (KDEs) of raw error distributions for ALL individual
    DL model versions on a single figure for direct comparison.
    DL Version distributions are shown as *filled areas* under the KDE curves.

    Args:
        analysis_results (dict): The 'results' dictionary from `calculate_prediction_difference`.
        sorted_version_keys (list): Sorted list of version string keys (e.g., ['version_4', 'version_5']).
        output_path (str): Path to save the generated plot image.
        xlim_range (tuple): (min_val, max_val) for the x-axis (raw error) limits.
    """
    print(f"\nGenerating consolidated DL versions KDE comparison plot to {output_path}...")
    plt.figure(figsize=(14, 8))
    alpha_fill_dl = 0.35

    x_kde_vals = np.linspace(xlim_range[0], xlim_range[1], 500)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_version_keys)))

    print("  Plotting DL Version KDEs (filled)...")
    for i, version_str in enumerate(sorted_version_keys):
        version_data = analysis_results.get(version_str, {})
        if ("metrics" in version_data and "dl" in version_data["metrics"] and
                "raw_errors_version" in version_data["metrics"]["dl"] and
                isinstance(version_data["metrics"]["dl"]["raw_errors_version"], np.ndarray) and
                version_data["metrics"]["dl"]["raw_errors_version"].size > 1):

            dl_raw_errors_version = version_data["metrics"]["dl"]["raw_errors_version"]
            try:
                kde_dl_version = st.gaussian_kde(dl_raw_errors_version)
                kde_values = kde_dl_version(x_kde_vals)

                me_dl = np.mean(dl_raw_errors_version)
                std_dl = np.std(dl_raw_errors_version)
                version_label_short = version_str.replace('version_', 'V')
                label_dl = f"{version_label_short} (ME:{me_dl:.2f}, Std:{std_dl:.2f})"

                plt.fill_between(x_kde_vals, 0, kde_values, color=colors[i], alpha=alpha_fill_dl, label=label_dl)

            except Exception as e:
                print(f"    Could not plot KDE for DL {version_str}: {e}")
        else:
            raw_errors = version_data.get("metrics", {}).get("dl", {}).get("raw_errors_version", None)
            data_size = raw_errors.size if isinstance(raw_errors, np.ndarray) else 0
            print(f"    Skipping KDE plot for DL {version_str}: insufficient data (size={data_size}) or data missing.")

    plt.title('Comparison of Raw Error Distributions for DL Versions (Filled KDEs)')
    plt.xlabel('Raw Error (mbar) [Pred - True]')
    plt.ylabel('Density')
    plt.axvline(0, color='red', linestyle='-', linewidth=1.0, label='Zero Error', zorder=6)
    plt.legend(loc='upper right', fontsize='small', framealpha=0.9)
    plt.grid(True, alpha=0.4)
    plt.xlim(xlim_range)

    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Consolidated KDE comparison plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving consolidated KDE comparison plot: {e}")
    plt.close()


if __name__ == "__main__":
    # --- Configuration Section ---
    model = "31"
    DL_BASE_DIRECTORY = f"/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model/CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast2h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"
    STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"
    VERSIONS_TO_ANALYZE = [4, 5, 6, 12, 17, 31, 32]

    print('--- Analysis Script Start ---')

    analysis_results, all_raw_dl = calculate_prediction_difference(
        DL_BASE_DIRECTORY,
        VERSIONS_TO_ANALYZE,
        STATS_DIRECTORY
    )

    # --- Display Summary Results to Console ---
    print("\n--- Overall Analysis Summary ---")
    sorted_version_keys = []
    if not analysis_results:
        print("No results to display.")
    else:
        sorted_version_keys = sorted([k for k in analysis_results.keys() if k.startswith('version_')],
                                     key=lambda x: int(x.split('_')[-1]))

        for version_str in sorted_version_keys:
            version_data = analysis_results[version_str]
            print(f"\n{version_str.replace('_', ' ').capitalize()}:")
            if "error" in version_data and not "metrics" in version_data:
                print(f"  Status: Error - {version_data['error']}")
                continue

            print(f"  Status: Processed")
            metrics_summary = version_data.get("metrics", {})
            if "dl" in metrics_summary:
                metrics = metrics_summary["dl"]
                method_label = 'DL:'
                if "error" in metrics:
                    print(f"    {method_label:<10} Error - {metrics['error']}")
                else:
                    print(
                        f"    {method_label:<10} "
                        f"ME: {metrics.get('me', float('nan')):<7.4f} | "
                        f"MAE: {metrics.get('mae', float('nan')):<7.4f} | "
                        f"Std(Raw): {metrics.get('std_raw', float('nan')):<7.4f} | "
                        f"Count: {metrics.get('count', 0)}")
            else:
                print(f"    {'DL:':<10} No data processed for this version.")

    # --- Generate Plots ---
    raw_error_plot_xlim = (-15, 15)
    if all_raw_dl:
        data_for_xlim = np.array(all_raw_dl)
        lower_bound = np.percentile(data_for_xlim, 0.5)
        upper_bound = np.percentile(data_for_xlim, 99.5)
        max_abs_bound = max(abs(lower_bound), abs(upper_bound), 15)
        raw_error_plot_xlim = (-max_abs_bound, max_abs_bound)
        print(f"\nAuto-determined xlim for raw error plots: {raw_error_plot_xlim}")

    # Graph 1: Overall Raw Error Distribution plot (DL All Versions Histogram)
    plot_overall_raw_error_distribution(
        dl_raw_errors=all_raw_dl,
        output_path=f"GRAPH1_overall_dl_raw_error_dist_{model}.png",
        bins=150,
        xlim_range=raw_error_plot_xlim
    )

    all_raw_dl_np = np.array(all_raw_dl)
    output_dir = f"saved_graph1_data_{model}"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "all_raw_dl.npy"), all_raw_dl_np)
    print(f"Saved all raw DL errors to {os.path.join(output_dir, 'all_raw_dl.npy')}")


    # Graph 2: All DL Versions KDEs Comparison
    if analysis_results and sorted_version_keys:
        plot_dl_version_kde_comparison(
            analysis_results=analysis_results,
            sorted_version_keys=sorted_version_keys,
            output_path=f"GRAPH2_all_versions_kdes_comparison_{model}.png",
            xlim_range=raw_error_plot_xlim
        )

        output_dir_g2 = f"saved_graph2_data_{model}"
        os.makedirs(output_dir_g2, exist_ok=True)
        save_dict = {
            'analysis_results': analysis_results,
            'sorted_version_keys': sorted_version_keys,
        }
        output_file = os.path.join(output_dir_g2, 'graph2_parameters.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Saved parameters for Graph 2 to {output_file}")
    else:
        print("\nSkipping GRAPH2 (All DL Versions KDEs Comparison) due to no analysis results or versions.")

    print("\n--- Analysis Script End ---")