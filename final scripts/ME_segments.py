import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
# ▼▼▼ V V V SET YOUR FILE PATH AND SETTINGS HERE V V V ▼▼▼

# 1. SET THE DIRECTORY WHERE YOUR .npz FILES ARE LOCATED
# ---
# DATA_DIRECTORY = r'C:\Users\shrei\PycharmProjects\MasterProject\final scripts'
DATA_DIRECTORY = '/home/mansour/ML3300-24a/shreibshtein/scripts'

# 2. PROVIDE A LIST OF NPZ FILES TO AGGREGATE AND PLOT
# ---
NPZ_FILENAMES = [
    'all_categorized_errors_analysis_31.npz'
]

# 3. SET THE DL MODEL CATEGORIES TO PLOT ON THE SAME FIGURE
# ---
CATEGORIES_TO_PLOT = ['NH_DJF', 'NH_MAM', 'NH_JJA', 'NH_SON', 'SH_DJF', 'SH_MAM', 'SH_JJA', 'SH_SON']

# 4. CHOOSE WHETHER TO PLOT THE OVERALL CONTROL BASELINES
# ---
PLOT_CONTROL_BASELINES = True

# 5. CHOOSE A DIRECTORY, TITLE, AND FILENAME FOR THE OUTPUT PLOT
# ---
OUTPUT_DIRECTORY = 'plots'  # The folder where the final plot will be saved
PLOT_TITLE = 'DL Model Seasonal Error Distributions vs. Control Baselines'
OUTPUT_FILENAME = 'pdf_plot_seasonal_vs_control.png'

# 6. PLOT SETTINGS
# ---
PLOT_X_LIMITS = (-15, 15)


# ▲▲▲ A A A YOUR SETTINGS END HERE A A A ▲▲▲
# =============================================================================


def plot_error_distributions(error_data, plot_styles, title, output_file, xlim_range):
    """
    Generates and saves a plot of the PDF for different error distributions.
    """
    print(f"\n📈 Generating PDF distribution plot: {title}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 10))
    x_vals = np.linspace(xlim_range[0], xlim_range[1], 1000)

    for name, errors in error_data.items():
        if errors.size > 1:
            style = plot_styles.get(name, {})
            kde = st.gaussian_kde(errors)
            mean_error = np.mean(errors)
            count = len(errors)
            label = f"{name} (ME={mean_error:.2f}, N={count})"
            ax.plot(x_vals, kde(x_vals), label=label, **style)
        else:
            print(f"  ⚠️ Skipping '{name}' as it contains insufficient data.")

    # --- Formatting ---
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel('Raw Error (mbar) [Prediction - True Value]', fontsize=14)
    ax.set_ylabel('Normalized Probability Density', fontsize=14)
    ax.set_xlim(xlim_range)
    ax.legend(fontsize='medium', loc='best')
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    fig.tight_layout()
    # Save the figure to the specified output file path
    plt.savefig(output_file, dpi=300)
    print(f"✅ Plot saved successfully to: {output_file}")
    plt.show()


def main():
    """
    Main function to load and aggregate data from multiple .npz files and plot.
    """
    print("--- PDF Error Distribution Plotting Script ---")

    # --- STEP 1: Load and aggregate data from all specified files ---
    temp_data = {cat: [] for cat in CATEGORIES_TO_PLOT}
    if PLOT_CONTROL_BASELINES:
        temp_data['control_persistence'] = []
        temp_data['control_linear'] = []

    for filename in NPZ_FILENAMES:
        full_file_path = os.path.join(DATA_DIRECTORY, filename)
        print(f"\n🔍 Loading data from: '{full_file_path}'")

        if not os.path.exists(full_file_path):
            print(f"  ❌ ERROR: File not found. Skipping.")
            continue
        try:
            data = np.load(full_file_path)
            # Collect DL model errors for specified categories
            for category in CATEGORIES_TO_PLOT:
                key = f"dl_{category}"
                if key in data:
                    temp_data[category].append(data[key])

            # Collect errors for control baselines
            if PLOT_CONTROL_BASELINES:
                p_nh, p_sh = data.get('persistence_NH'), data.get('persistence_SH')
                if p_nh is not None and p_sh is not None:
                    temp_data['control_persistence'].append(np.concatenate([p_nh, p_sh]))
                l_nh, l_sh = data.get('linear_NH'), data.get('linear_SH')
                if l_nh is not None and l_sh is not None:
                    temp_data['control_linear'].append(np.concatenate([l_nh, l_sh]))

            data.close()
        except Exception as e:
            print(f"  ❌ ERROR: Could not process file '{filename}'. Reason: {e}")
            continue

    # --- STEP 2: Finalize aggregated data and define plot styles ---
    error_data_to_plot = {}
    plot_styles = {}

    colors = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES_TO_PLOT)))
    category_colors = {category: color for category, color in zip(CATEGORIES_TO_PLOT, colors)}

    for category in CATEGORIES_TO_PLOT:
        if temp_data[category]:
            label = f"DL Model ({category})"
            error_data_to_plot[label] = np.concatenate(temp_data[category])
            plot_styles[label] = {'linestyle': '-', 'lw': 3, 'color': category_colors[category]}

    if PLOT_CONTROL_BASELINES:
        if temp_data['control_persistence']:
            label = "Persistence Baseline (Control)"
            error_data_to_plot[label] = np.concatenate(temp_data['control_persistence'])
            plot_styles[label] = {'linestyle': '--', 'lw': 2.5, 'color': 'black'}

        if temp_data['control_linear']:
            label = "Linear Baseline (Control)"
            error_data_to_plot[label] = np.concatenate(temp_data['control_linear'])
            plot_styles[label] = {'linestyle': ':', 'lw': 2.5, 'color': 'dimgray'}

    # --- STEP 3: Generate the plot ---
    if not error_data_to_plot:
        print("\n❌ ERROR: No data was successfully collected to create a plot.")
        return

    # --- NEW: Create the output directory if it doesn't exist ---
    print(f"\n📁 Ensuring output directory '{OUTPUT_DIRECTORY}' exists...")
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Construct the full path for the output file
    full_output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)

    plot_error_distributions(
        error_data=error_data_to_plot,
        plot_styles=plot_styles,
        title=PLOT_TITLE,
        output_file=full_output_path,  # Pass the full path to the function
        xlim_range=PLOT_X_LIMITS
    )


if __name__ == '__main__':
    main()