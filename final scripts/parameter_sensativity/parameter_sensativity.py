import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch  # For loading .pt files
import traceback

# User-defined list of parameters (used to filter parameters found in CSV Column A)
PARAMETERS_STRING = "date#sin,date#cos,lat,lon#sin,lon#cos,sp,t2m,intensity,u10,v10,tcw,z,ta_250,ta_300,ta_500,ta_850,ua_250,ua_300,ua_500,ua_850,va_250,va_300,va_500,va_850,za_250,za_300,za_500,za_850,msl,slhf,tp,qa_250,qa_300,qa_500,qa_850,sshf"


def parse_parameters(param_string):
    """Parses the comma-separated parameter string into a list."""
    return [p.strip() for p in param_string.split(',')]


def load_normalization_stats(stats_dir, mean_filename_pattern, std_filename_pattern, target_variable_index=28):
    """
    Loads mean and std tensors and extracts the std for the target variable.
    """
    print(f"\nAttempting to load normalization stats from: {stats_dir}")
    print(f"Mean file pattern: '{mean_filename_pattern}', Std file pattern: '{std_filename_pattern}'")

    mean_files = glob.glob(os.path.join(stats_dir, mean_filename_pattern))
    std_files = glob.glob(os.path.join(stats_dir, std_filename_pattern))

    if not mean_files:
        print(f"  Error: Mean file not found in '{stats_dir}' with pattern '{mean_filename_pattern}'")
        return None
    if not std_files:
        print(f"  Error: Std file not found in '{stats_dir}' with pattern '{std_filename_pattern}'")
        return None

    mean_file_path = mean_files[0]
    std_file_path = std_files[0]
    if len(mean_files) > 1:
        print(
            f"  Warning: Multiple mean files found matching pattern. Using the first one: {os.path.basename(mean_file_path)}")
    if len(std_files) > 1:
        print(
            f"  Warning: Multiple std files found matching pattern. Using the first one: {os.path.basename(std_file_path)}")

    try:
        print(f"  Loading mean from: {mean_file_path}")
        mean_tensor = torch.load(mean_file_path, map_location='cpu')
        print(f"  Loading std from: {std_file_path}")
        std_tensor = torch.load(std_file_path, map_location='cpu')
        print(f"  Successfully loaded mean and std tensors.")

        if not isinstance(std_tensor, torch.Tensor):
            print("  Error: Loaded std is not a torch.Tensor.")
            return None
        if std_tensor.ndim == 0 or std_tensor.shape[0] <= target_variable_index:
            print(
                f"  Error: Std tensor (shape {std_tensor.shape}) is empty or does not have index {target_variable_index}.")
            return None

        target_std_value_pa = std_tensor[target_variable_index].item()
        target_std_for_denorm_mbar = target_std_value_pa / 100.0
        print(f"  Target variable (index {target_variable_index}) original std (Pa): {target_std_value_pa:.4f}")
        print(f"  Derived std for denormalization (to mbar): {target_std_for_denorm_mbar:.4f}")
        return target_std_for_denorm_mbar
    except Exception as e:
        print(f"  An error occurred while loading or processing mean/std files: {e}")
        traceback.print_exc()
        return None


def plot_parameter_sensitivity(param_means_denorm, param_stds_denorm, parameter_plot_labels,
                               plot_title_suffix, output_filename_suffix, output_dir=".",
                               horizontal_plot=False, sorted_desc=False):  # Added horizontal_plot and sorted_desc
    """
    Generates and saves a bar plot of parameter sensitivities.
    Can be used for per-version or cross-version plots.
    If horizontal_plot is True and sorted_desc is True, it inverts y-axis for top-to-bottom sorted display.
    """
    if param_means_denorm.empty:
        print(f"  No sensitivity data to plot for {plot_title_suffix}.")
        return

    num_params = len(param_means_denorm)
    # Adjust figure size based on number of parameters
    if horizontal_plot:
        fig_width = 10  # Fixed width for horizontal
        fig_height = max(6, num_params * 0.35)  # Height scales with num_params
    else:
        fig_width = max(12, num_params * 0.7)  # Width scales for vertical
        fig_height = 8

    plt.figure(figsize=(fig_width, fig_height))

    if horizontal_plot:
        plt.barh(parameter_plot_labels, param_means_denorm.values, xerr=param_stds_denorm.values,
                 capsize=4, color='mediumseagreen', edgecolor='black', ecolor='darkgray')  # xerr for barh
        plt.yticks(fontsize=9)
        plt.xticks(fontsize=10)
        plt.ylabel('Parameter', fontsize=14)
        plt.xlabel('MAE Change in mbar', fontsize=14)
        if sorted_desc:  # If data was sorted descending, invert y-axis to show largest at top
            plt.gca().invert_yaxis()
    else:  # Vertical bars
        plt.bar(parameter_plot_labels, param_means_denorm.values, yerr=param_stds_denorm.values,
                capsize=5, color='lightcoral', edgecolor='black', ecolor='darkgray')
        plt.xticks(rotation=60, ha='right', fontsize=9)
        plt.yticks(fontsize=10)
        plt.xlabel('Parameter', fontsize=14)
        plt.ylabel('MAE Change in mbar', fontsize=14)

    plt.title(f'Parameter Sensitivity {plot_title_suffix}', fontsize=16)
    plt.grid(axis='x' if horizontal_plot else 'y', linestyle='--', alpha=0.7)  # Grid on value axis
    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"denorm_sensitivity_plot_{output_filename_suffix}.png")

    try:
        plt.savefig(output_path)
        print(f"  Sensitivity plot saved successfully to: {output_path}")
    except Exception as e:
        print(f"  Error saving sensitivity plot {output_filename_suffix}: {e}")
    plt.close()


def analyze_sensitivity_files(base_dir, versions_to_analyze, user_defined_parameters_filter,
                              target_std_for_denorm,
                              param_name_col_header, sensitivity_value_col_header,
                              output_dir="."):
    print(f"\nAnalyzing sensitivity files in base directory: {base_dir}")
    print(f"Selected versions for analysis: {versions_to_analyze}")
    print(f"Will filter parameters from CSV using {len(user_defined_parameters_filter)} user-defined parameters.")
    print(f"Expecting parameter names in CSV column: '{param_name_col_header}'")
    print(f"Expecting sensitivity values in CSV column: '{sensitivity_value_col_header}'")

    if target_std_for_denorm is None:
        print("Error: Target standard deviation for denormalization is not available. Cannot proceed.")
        return {}

    all_version_results = {}

    for version in versions_to_analyze:
        version_str = f"version_{version}"
        checkpoints_dir = os.path.join(base_dir, version_str, "checkpoints")
        print(f"\nProcessing {version_str}...")

        if not os.path.exists(checkpoints_dir):
            print(f"  Error: Checkpoints directory not found: {checkpoints_dir}")
            continue

        search_pattern = os.path.join(checkpoints_dir, "*-eval_noise_diff-test_mae.csv")
        sensitivity_files = glob.glob(search_pattern)

        if not sensitivity_files:
            search_pattern_any_csv = os.path.join(checkpoints_dir, "*.csv")
            sensitivity_files = glob.glob(search_pattern_any_csv)
            if sensitivity_files:
                print(
                    f"  Warning: No specific '*-eval_noise_diff-test_mae.csv' found. Using first available CSV: {os.path.basename(sensitivity_files[0])}")
            else:
                print(
                    f"  Error: No sensitivity CSV file found in '{checkpoints_dir}' (tried '*-eval_noise_diff-test_mae.csv' and '*.csv').")
                continue

        sensitivity_file_path = sensitivity_files[0]
        if len(sensitivity_files) > 1:
            print(
                f"  Warning: Multiple sensitivity files found matching search. Using the first one: {os.path.basename(sensitivity_file_path)}")
        print(f"  Found sensitivity file: {os.path.basename(sensitivity_file_path)}")

        try:
            df = pd.read_csv(sensitivity_file_path)
            if df.empty:
                print(f"  Warning: Sensitivity file '{os.path.basename(sensitivity_file_path)}' is empty.")
                continue

            if param_name_col_header not in df.columns:
                print(f"  Error: Specified parameter name column '{param_name_col_header}' not found in CSV.")
                continue
            if sensitivity_value_col_header not in df.columns:
                print(f"  Error: Specified sensitivity value column '{sensitivity_value_col_header}' not found in CSV.")
                continue

            df[param_name_col_header] = df[param_name_col_header].astype(str)
            try:
                df[sensitivity_value_col_header] = pd.to_numeric(df[sensitivity_value_col_header], errors='coerce')
                if df[sensitivity_value_col_header].isnull().any():
                    print(
                        f"  Warning: Column '{sensitivity_value_col_header}' contains non-numeric values or NaNs after coercion. These will be ignored.")
            except Exception as e:
                print(f"  Error converting sensitivity value column '{sensitivity_value_col_header}' to numeric: {e}")
                continue

            grouped = df.groupby(param_name_col_header)[sensitivity_value_col_header]
            means_norm_all_csv_params = grouped.mean()
            stds_norm_all_csv_params = grouped.std().fillna(0)

            csv_param_names = means_norm_all_csv_params.index.tolist()
            parameter_labels_for_this_version_plot = [p for p in csv_param_names if p in user_defined_parameters_filter]

            if not parameter_labels_for_this_version_plot:
                continue

            means_norm = means_norm_all_csv_params.loc[parameter_labels_for_this_version_plot]
            stds_norm = stds_norm_all_csv_params.loc[parameter_labels_for_this_version_plot]

            means_denorm = means_norm * target_std_for_denorm
            stds_denorm = stds_norm * target_std_for_denorm

            all_version_results[version] = {
                "means_denorm": means_denorm,  # Series indexed by parameter_labels_for_this_version_plot
                "stds_denorm": stds_denorm,  # Series indexed by parameter_labels_for_this_version_plot
                # "plot_labels" is not strictly needed here as series index holds the names
            }
            # Per-version plots are not sorted by impact by default, keeping original order from filter list
            plot_parameter_sensitivity(means_denorm, stds_denorm, parameter_labels_for_this_version_plot,
                                       plot_title_suffix=f"for Model Version {version}",
                                       output_filename_suffix=f"version_{version}",
                                       output_dir=output_dir,
                                       horizontal_plot=True,  # Make per-version plots horizontal too
                                       sorted_desc=False)  # Not sorting per-version plot by default

        except Exception as e:
            print(
                f"  An unexpected error occurred processing {os.path.basename(sensitivity_file_path)} for version {version}: {e}")
            traceback.print_exc()
        print("-" * 40)
    return all_version_results


if __name__ == "__main__":
    # --- User Configuration ---
    BASE_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model/CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast4h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"
    VERSIONS_TO_ANALYZE = [9, 10, 11, 12]
    STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"
    OUTPUT_PLOT_DIR = "sensitivity_analysis_FOLDS_34"

    PARAMETER_NAME_COLUMN_HEADER = "Parameter"
    SENSITIVITY_VALUE_COLUMN_HEADER = "Difference from Baseline"

    MEAN_FILENAME_PATTERN = "*mean.pt"
    STD_FILENAME_PATTERN = "*std.pt"
    TARGET_VARIABLE_INDEX = 28

    # --- Script Execution ---
    user_parameters_filter_list = parse_parameters(PARAMETERS_STRING)

    if not os.path.exists(BASE_DIRECTORY):
        print(f"CRITICAL ERROR: Base directory for models does not exist: {BASE_DIRECTORY}")
    elif not os.path.exists(STATS_DIRECTORY):
        print(f"CRITICAL ERROR: Stats directory for mean/std files does not exist: {STATS_DIRECTORY}")
    else:
        target_std_denorm_factor = load_normalization_stats(
            STATS_DIRECTORY,
            MEAN_FILENAME_PATTERN,
            STD_FILENAME_PATTERN,
            TARGET_VARIABLE_INDEX
        )

        if target_std_denorm_factor is not None:
            per_version_sensitivity_results = analyze_sensitivity_files(
                BASE_DIRECTORY,
                VERSIONS_TO_ANALYZE,
                user_parameters_filter_list,
                target_std_denorm_factor,
                PARAMETER_NAME_COLUMN_HEADER,
                SENSITIVITY_VALUE_COLUMN_HEADER,
                OUTPUT_PLOT_DIR
            )

            print("\n--- Per-Version Denormalized Parameter Sensitivity Summary ---")
            # (Per-version summary print logic can remain similar)
            if not per_version_sensitivity_results:
                print("No per-version sensitivity results were processed.")
            else:
                for version, data in per_version_sensitivity_results.items():
                    print(f"\nVersion {version}:")
                    if "means_denorm" not in data or data["means_denorm"].empty:
                        print("  No parameters analyzed or plotted for this version.")
                    else:
                        print("  Mean denormalized sensitivities (e.g., MAE Change in mbar):")
                        # means_denorm is already indexed by parameter names
                        summary_df = pd.DataFrame({
                            'Mean_Denorm_Sens': data['means_denorm'],
                            'Std_Denorm_Sens': data['stds_denorm']
                        })
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width',
                                               1000):
                            print(summary_df)

            # --- Aggregate results across versions for the new plot ---
            if per_version_sensitivity_results:
                print("\n--- Aggregating Results for Cross-Version Sensitivity Plot ---")
                cross_version_sens_values = {}

                for version_num, version_data in per_version_sensitivity_results.items():
                    if "means_denorm" in version_data and not version_data["means_denorm"].empty:
                        for param_label, mean_value in version_data["means_denorm"].items():  # Iterate over Series
                            if param_label not in cross_version_sens_values:
                                cross_version_sens_values[param_label] = []
                            cross_version_sens_values[param_label].append(mean_value)

                overall_param_means_dict = {}
                overall_param_stds_dict = {}

                # Calculate mean and std only for parameters that appeared in results
                # The order will be based on user_parameters_filter_list, but only for existing params
                temp_plot_labels_with_data = []
                for param_label in user_parameters_filter_list:
                    if param_label in cross_version_sens_values and cross_version_sens_values[param_label]:
                        values_list = cross_version_sens_values[param_label]
                        overall_param_means_dict[param_label] = np.mean(values_list)
                        overall_param_stds_dict[param_label] = np.std(values_list)
                        temp_plot_labels_with_data.append(param_label)

                if temp_plot_labels_with_data:
                    # Convert to Series for easier handling and consistent indexing
                    overall_means_series = pd.Series(overall_param_means_dict).reindex(temp_plot_labels_with_data)
                    overall_stds_series = pd.Series(overall_param_stds_dict).reindex(temp_plot_labels_with_data)

                    # --- Sort by absolute impact for the cross-version plot ---
                    # Create a temporary series of absolute mean values for sorting
                    abs_means_for_sorting = overall_means_series.abs().sort_values(ascending=False)

                    # Get the sorted parameter names (these are the labels for the plot)
                    sorted_plot_labels = abs_means_for_sorting.index.tolist()

                    # Reorder the actual mean and std series according to these sorted labels
                    final_means_series_sorted = overall_means_series.loc[sorted_plot_labels]
                    final_stds_series_sorted = overall_stds_series.loc[sorted_plot_labels]

                    print(f"\nPlotting overall sensitivity for parameters (sorted by impact): {sorted_plot_labels}")

                    plot_parameter_sensitivity(
                        final_means_series_sorted,
                        final_stds_series_sorted,
                        sorted_plot_labels,
                        plot_title_suffix="24 Hour Forecast",
                        output_filename_suffix="overall_cross_version_sorted",
                        output_dir=OUTPUT_PLOT_DIR,
                        horizontal_plot=True,  # Use horizontal plot for overall summary
                        sorted_desc=True  # Indicate data is sorted for plot function (for y-axis inversion)
                    )
                    print("\nCross-Version Sensitivity Summary (Mean of per-version means, sorted by impact):")
                    summary_cross_df = pd.DataFrame({
                        'Parameter': sorted_plot_labels,  # Use sorted labels
                        'Overall_Mean_Sens': final_means_series_sorted.values,
                        'Overall_Std_Sens (Variability across versions)': final_stds_series_sorted.values
                    }).set_index('Parameter')
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width',
                                           1000):
                        print(summary_cross_df)
                else:
                    print("No data available to generate the cross-version sensitivity plot.")

            if OUTPUT_PLOT_DIR and per_version_sensitivity_results:
                print(f"\nPlots saved in: {os.path.abspath(OUTPUT_PLOT_DIR)}")
        else:
            print("Could not proceed with analysis due to issues loading normalization stats.")

        print("\nAnalysis complete.")
