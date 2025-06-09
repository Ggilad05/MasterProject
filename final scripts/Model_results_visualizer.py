import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st  # For KDE plots
import pickle
import random

def plot_all_dl_versions_kdes_vs_baselines(
        analysis_results,
        sorted_version_keys,  # To plot DL versions in a consistent order
        global_persistence_raw_errors,
        global_new_naive_raw_errors,
        output_path="all_dl_versions_kdes_comparison.png",
        xlim_range=(-15, 15)
):
    """
    Plots Kernel Density Estimates (KDEs) of raw error distributions for ALL individual
    DL model versions and the global baseline models on a single figure.
    DL Version distributions are shown as *filled areas* under the KDE curves.
    Baseline distributions are shown as distinct lines.

    Args:
        analysis_results (dict): The 'results' dictionary from `calculate_prediction_difference`.
        sorted_version_keys (list): Sorted list of version string keys (e.g., ['version_4', 'version_5']).
        global_persistence_raw_errors (list or np.array): Aggregated raw errors for Persistence.
        global_new_naive_raw_errors (list or np.array): Aggregated raw errors for New Naive.
        output_path (str): Path to save the generated plot image.
        xlim_range (tuple): (min_val, max_val) for the x-axis (raw error) limits.
    """
    print(f"\nGenerating consolidated KDE comparison plot (with filled DL areas) to {output_path}...")
    plt.figure(figsize=(14, 8))  # Adjust figure size if needed
    alpha_fill_dl = 0.35  # Alpha for the filled area for DL versions
    alpha_kde_baseline = 0.95  # Alpha for baseline KDE lines
    linewidth_kde_baseline = 2.5  # Make baseline lines slightly thicker

    # X-values for evaluating KDEs, consistent across all plots.
    x_kde_vals = np.linspace(xlim_range[0], xlim_range[1], 500)

    # --- Plot KDEs with fills for each DL Version ---
    # Use a perceptually uniform colormap like 'viridis' or 'cividis'
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_version_keys)))

    print("  Plotting DL Version KDEs (filled)...")
    for i, version_str in enumerate(sorted_version_keys):
        version_data = analysis_results.get(version_str, {})
        # Check if necessary data exists and is valid
        if ("metrics" in version_data and "dl" in version_data["metrics"] and
                "raw_errors_version" in version_data["metrics"]["dl"] and
                isinstance(version_data["metrics"]["dl"]["raw_errors_version"], np.ndarray) and
                version_data["metrics"]["dl"]["raw_errors_version"].size > 1):

            dl_raw_errors_version = version_data["metrics"]["dl"]["raw_errors_version"]
            try:
                kde_dl_version = st.gaussian_kde(dl_raw_errors_version)
                kde_values = kde_dl_version(x_kde_vals)  # Calculate KDE values on the x-grid

                me_dl = np.mean(dl_raw_errors_version)
                std_dl = np.std(dl_raw_errors_version)
                # Create label, maybe shorten version string if too long in legend
                version_label_short = f"V{i + 1}"
                label_dl = f"{version_label_short} (ME:{me_dl:.2f}, Std:{std_dl:.2f})"

                # Plot the filled area using fill_between
                plt.fill_between(x_kde_vals, 0, kde_values, color=colors[i], alpha=alpha_fill_dl, label=label_dl)
                # Optional: Plot the line boundary for the fill (can make edges clearer)
                # plt.plot(x_kde_vals, kde_values, color=colors[i], linewidth=1.0, alpha=alpha_fill_dl + 0.2)

            except Exception as e:
                print(f"    Could not plot KDE for DL {version_str}: {e}")
        else:
            # Check why data is missing or invalid
            raw_errors = version_data.get("metrics", {}).get("dl", {}).get("raw_errors_version", None)
            data_size = raw_errors.size if isinstance(raw_errors, np.ndarray) else 0
            print(f"    Skipping KDE plot for DL {version_str}: insufficient data (size={data_size}) or data missing.")

    # --- Plot Global Baseline KDEs as Lines (on top) ---
    print("  Plotting Baseline KDEs (lines)...")
    persistence_raw_np_global = np.array(global_persistence_raw_errors)
    if persistence_raw_np_global.size > 1:
        try:
            kde_persistence = st.gaussian_kde(persistence_raw_np_global)
            me_p = np.mean(persistence_raw_np_global)
            std_p = np.std(persistence_raw_np_global)
            label_p = f"Persistence (Global ME:{me_p:.2f}, Std:{std_p:.2f})"
            # Plot with higher zorder to be on top of fills, distinct style
            plt.plot(x_kde_vals, kde_persistence(x_kde_vals), label=label_p,
                     linewidth=linewidth_kde_baseline, linestyle='--',
                     alpha=alpha_kde_baseline, color='black', zorder=5)
        except Exception as e:
            print(f"    Could not plot global KDE for Persistence: {e}")

    new_naive_raw_np_global = np.array(global_new_naive_raw_errors)
    if new_naive_raw_np_global.size > 1:
        try:
            kde_new_naive = st.gaussian_kde(new_naive_raw_np_global)
            me_nn = np.mean(new_naive_raw_np_global)
            std_nn = np.std(new_naive_raw_np_global)
            label_nn = f"Naive Baseline (Linear) (Global ME:{me_nn:.2f}, Std:{std_nn:.2f})"
            # Plot with higher zorder, distinct style
            plt.plot(x_kde_vals, kde_new_naive(x_kde_vals), label=label_nn,
                     linewidth=linewidth_kde_baseline, linestyle=':',
                     alpha=alpha_kde_baseline, color='darkred', zorder=5)
        except Exception as e:
            print(f"    Could not plot global KDE for New Naive: {e}")

    # --- Plot Aesthetics and Labels ---
    plt.title('Error Distributions - 6H')
    plt.xlabel('[Pred - True] (mbar)')
    plt.ylabel('Density')
    plt.axvline(0, color='red', linestyle='-', linewidth=1.0, label='Zero Error', zorder=6)  # Make zero line prominent
    # Adjust legend position and size if needed
    plt.legend(loc='upper right', fontsize='x-small', framealpha=0.9)
    plt.grid(True, alpha=0.4)
    plt.xlim(xlim_range)
    # Add a subtle background color? Optional.
    # plt.gca().set_facecolor('#f0f0f0')

    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)  # Save plot with specified DPI
        print(f"Consolidated KDE comparison plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving consolidated KDE comparison plot: {e}")
    plt.close()  # Close the plot figure to free up memory

# === Configuration ===
base_path = r"C:\Users\shrei\PycharmProjects\MasterProject\download\26.05.2025\saved_graph3_"
model = "31"
output_dir = os.path.join(base_path + model)

# === Ensure Folder Exists ===

pkl_path = os.path.join(output_dir, 'graph3_parameters.pkl')


# === Load Parameters and Plot ===
with open(pkl_path, 'rb') as f:
    loaded_data = pickle.load(f)

plot_path = os.path.join(output_dir, f'GRAPH3_all_versions_kdes_comparison_{model}d.png')

plot_all_dl_versions_kdes_vs_baselines(
    analysis_results=loaded_data['analysis_results'],
    sorted_version_keys=loaded_data['sorted_version_keys'],
    global_persistence_raw_errors=loaded_data['global_persistence_raw_errors'],
    global_new_naive_raw_errors=loaded_data['global_new_naive_raw_errors'],
    output_path=plot_path,
    xlim_range=(-5, 5)  # Adjust as needed
)

