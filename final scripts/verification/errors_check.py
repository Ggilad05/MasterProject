import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # for linear regression

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = "CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast1h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"
DL_BASE_DIRECTORY = f"/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model/{MODEL_NAME}"
BASELINE_PRED_BASE_DIR = f"/home/mansour/ML3300-24a/shreibshtein/predictions_naive_31"
STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"

FOLDS_TO_AGGREGATE = ['45', '46', '47', '48']

OUTPUT_PLOT_FILE_ALL = "dl_error_vs_persistence_error_with_fit.png"
OUTPUT_PLOT_FILE_SEASONS = "dl_error_vs_persistence_error_by_season.png"

# =============================================================================
# HELPERS
# =============================================================================
def categorize_key(key: str) -> str:
    """
    Return one of: 'Winter', 'Summer', 'Spring', 'Fall', or 'Other'
    Rules (case-insensitive):
      - Winter (blue): (DJF & NH) or (JJA & SH)
      - Summer (red):   (JJA & NH) or (DJF & SH)
      - Spring (green): (MAM & NH) or (SON & SH)
      - Fall (yellow):  (SON & NH) or (MAM & SH)
    """
    k = key.upper()
    has = lambda s: s in k
    nh = has("NH")
    sh = has("SH")
    if (has("DJF") and nh) or (has("JJA") and sh):
        return "Winter"
    if (has("JJA") and nh) or (has("DJF") and sh):
        return "Summer"
    if (has("MAM") and nh) or (has("SON") and sh):
        return "Spring"
    if (has("SON") and nh) or (has("MAM") and sh):
        return "Fall"
    return "Other"

CATEGORY_STYLES = {
    "Winter": dict(color="blue",   label="Winter (DJF NH / JJA SH)"),
    "Summer": dict(color="red",    label="Summer (JJA NH / DJF SH)"),
    "Spring": dict(color="green",  label="Spring (MAM NH / SON SH)"),
    "Fall":   dict(color="yellow", label="Fall (SON NH / MAM SH)"),
    "Other":  dict(color="gray",   label="Other"),
}
SEASONS_ORDER = ["Winter", "Summer", "Spring", "Fall"]

def finite_pair_mask(a, b):
    """Return boolean mask where both arrays are finite and same shape."""
    return np.isfinite(a) & np.isfinite(b) & (np.shape(a) == np.shape(b))

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("--- Starting Error vs. (Persistence - y_true) Analysis ---")

    # Load mean/std
    try:
        mean_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-mean.pt"), map_location='cpu')
        std_tensor  = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-std.pt"),  map_location='cpu')
        print("[INFO] Successfully loaded mean and std files.")
    except Exception as e:
        raise SystemExit(f"[FATAL] Could not load mean/std files from '{STATS_DIRECTORY}'. Error: {e}")

    # Containers
    all_dl_errors = []
    all_persistence_errors = []
    cat_data = {name: {"x": [], "y": []} for name in CATEGORY_STYLES.keys()}

    # Iterate folds
    for fold_version in FOLDS_TO_AGGREGATE:
        print(f"\n--- Processing Fold Version: {fold_version} ---")
        search_path = os.path.join(DL_BASE_DIRECTORY, f"version_{fold_version}", "checkpoints", "*predictions.pt")
        prediction_files = glob.glob(search_path)

        if not prediction_files:
            print(f"[WARNING] No prediction file found for fold {fold_version} at '{search_path}'. Skipping.")
            continue
        if len(prediction_files) > 1:
            print(f"[WARNING] Multiple prediction files found; using first: {prediction_files[0]}")

        dl_predictions_file = prediction_files[0]
        print(f"[INFO] Found prediction file: {os.path.basename(dl_predictions_file)}")

        try:
            predictions_data = torch.load(dl_predictions_file, map_location='cpu')
        except Exception as e:
            print(f"[ERROR] Could not load or process file {dl_predictions_file}: {e}")
            continue

        for key, value in predictions_data.items():
            try:
                y_hat_tensor, y_true_tensor = value[0].cpu(), value[1].cpu()

                # De-normalize (to mbar)
                denorm_y_pred = (y_hat_tensor[:, 0] * std_tensor[28] + mean_tensor[28]) / 100.0
                denorm_y_true = (y_true_tensor[:, 0] * std_tensor[28] + mean_tensor[28]) / 100.0

                # Errors (arrays)
                dl_error = (denorm_y_pred - denorm_y_true).detach().numpy()

                # Load persistence (Pa -> mbar), compute error
                year = key.split('_')[1]
                persistence_path = os.path.join(BASELINE_PRED_BASE_DIR, 'naive', year, key)
                persistence_pred_pa = np.load(persistence_path)  # shape must match
                persistence_pred_mbar = persistence_pred_pa / 100.0
                persistence_error = persistence_pred_mbar - denorm_y_true.numpy()

                # Filter NaNs/Infs and aggregate
                mask = np.isfinite(dl_error) & np.isfinite(persistence_error)
                if not np.any(mask):
                    continue

                dl_err_valid = dl_error[mask]
                pers_err_valid = persistence_error[mask]

                all_dl_errors.extend(dl_err_valid.tolist())
                all_persistence_errors.extend(pers_err_valid.tolist())

                # Per-category (by key)
                cat = categorize_key(key)
                cat_data[cat]["x"].extend(pers_err_valid.tolist())
                cat_data[cat]["y"].extend(dl_err_valid.tolist())

            except FileNotFoundError:
                # Missing persistence file; skip quietly
                pass
            except Exception as e:
                print(f"[ERROR] Failed to process key '{key}'. Error: {e}")

    # Abort if nothing
    if not all_dl_errors or not all_persistence_errors:
        print("\n[FATAL] No data was successfully processed. Cannot generate plots.")
        raise SystemExit()

    # ===========================
    # PLOT 1: All points
    # ===========================
    print(f"\n[INFO] Plotting ALL points: {len(all_dl_errors)}")

    # Regression (only if >=2 points)
    if len(all_persistence_errors) >= 2:
        slope_all, intercept_all, r_all, p_all, se_all = stats.linregress(all_persistence_errors, all_dl_errors)
        r2_all = r_all ** 2
    else:
        slope_all = intercept_all = r2_all = np.nan

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))

    for cat, style in CATEGORY_STYLES.items():
        x_vals = np.array(cat_data[cat]["x"])
        y_vals = np.array(cat_data[cat]["y"])
        if x_vals.size:
            plt.scatter(x_vals, y_vals, alpha=0.35, s=10, color=style["color"],
                        label=f"{style['label']} (N={x_vals.size})")

    all_x = np.array(all_persistence_errors)
    all_y = np.array(all_dl_errors)
    mn = float(np.min(np.concatenate([all_x, all_y])))
    mx = float(np.max(np.concatenate([all_x, all_y])))
    lims = [mn, mx]

    # y=x and fit
    plt.plot(lims, lims, 'r--', alpha=0.6, lw=2, label='y = x (Error Equivalence)')
    if np.isfinite(slope_all):
        fit_x = np.array(lims)
        fit_y = intercept_all + slope_all * fit_x
        fit_label = f'Linear Fit (All)\ny = {slope_all:.2f}x + {intercept_all:.2f}\n$R^2$ = {r2_all:.3f}'
        plt.plot(fit_x, fit_y, '--', color='purple', lw=2, label=fit_label)

    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    plt.title('DL Model Error vs. Persistence Baseline Error (All Points)', fontsize=18)
    plt.xlabel('Persistence Error (Persistence - True) [mbar]', fontsize=14)
    plt.ylabel('DL Model Error (Prediction - True) [mbar]', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(lims); plt.ylim(lims)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILE_ALL, dpi=300)
    print(f"[SUCCESS] Saved: {OUTPUT_PLOT_FILE_ALL}")
    plt.close()

    # ===========================
    # PLOT 2: Four panels (one per season)
    # ===========================
    print("[INFO] Plotting 4 seasonal panels")

    # Collect limits across seasons for consistency
    sx = [np.asarray(cat_data[s]["x"]) for s in SEASONS_ORDER if len(cat_data[s]["x"]) > 0]
    sy = [np.asarray(cat_data[s]["y"]) for s in SEASONS_ORDER if len(cat_data[s]["y"]) > 0]
    all_x = np.concatenate(sx or [np.array([0.0])])
    all_y = np.concatenate(sy or [np.array([0.0])])
    data_min = float(np.min([all_x.min(), all_y.min()]))
    data_max = float(np.max([all_x.max(), all_y.max()]))
    pad = 0.05 * (data_max - data_min if data_max > data_min else 1.0)
    lims = [data_min - pad, data_max + pad]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, SEASONS_ORDER):
        x_vals = np.asarray(cat_data[season]["x"])
        y_vals = np.asarray(cat_data[season]["y"])
        style = CATEGORY_STYLES[season]

        if x_vals.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{season} (N=0)", fontsize=14)
        else:
            ax.scatter(x_vals, y_vals, alpha=0.4, s=12, color=style["color"])
            ax.plot(lims, lims, 'r--', alpha=0.6, lw=1.5, label='y = x')

            if x_vals.size >= 2:
                s, b, r, p, se = stats.linregress(x_vals, y_vals)
                ax.plot(lims, b + s * np.array(lims), '--', color='purple', lw=1.8,
                        label=f"Fit: y={s:.2f}x+{b:.2f}, $R^2$={r**2:.3f}")

            ax.axhline(0, color='black', linewidth=1, linestyle='--')
            ax.axvline(0, color='black', linewidth=1, linestyle='--')
            ax.set_title(f"{style['label']} (N={x_vals.size})", fontsize=12)
            ax.legend(fontsize=9, loc="best")

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle("DL Error vs. Persistence Error — 4 Seasons", fontsize=18)
    fig.supxlabel("Persistence Error (Persistence - True) [mbar]", fontsize=14)
    fig.supylabel("DL Model Error (Prediction - True) [mbar]", fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(OUTPUT_PLOT_FILE_SEASONS, dpi=300)
    print(f"[SUCCESS] Saved: {OUTPUT_PLOT_FILE_SEASONS}")
    plt.close()

    print("\n--- Analysis Script End ---")
