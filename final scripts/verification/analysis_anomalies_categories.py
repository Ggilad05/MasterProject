import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# ============================
# PATHS (match your previous script)
# ============================
OUT_DIR = r"C:\Users\shrei\PycharmProjects\MasterProject\final scripts\verification\composites"  # where anomalies live: composites/<Season>/<CatReadable>/<var[_lev]_anomaly.npy>

# ============================
# CONSTANTS (match your previous script)
# ============================
SEASONS = ["Winter", "Summer", "Spring", "Fall"]

# Field list (same structure as before)
FIELDS = {
    "sshf": None, "slhf": None
}

# Category folder names you used when saving
CATEGORIES_ORDER = [
    "smallDL_smallPers",
    "smallDL_bigPers",
    "bigDL_smallPers",
    "bigDL_bigPers",
]
CATEGORY_TITLES = {
    "smallDL_smallPers": "DL small  · PE small",
    "smallDL_bigPers":   "DL small  · PE big",
    "bigDL_smallPers":   "DL big    · PE small",
    "bigDL_bigPers":     "DL big    · PE big",
}

# ============================
# HELPERS
# ============================
def anomaly_path(season: str, cat_readable: str, var: str, level=None) -> str:
    """Return expected path of the anomaly .npy file."""
    base = f"{var}_anomaly.npy" if level is None else f"{var}_{level}_anomaly.npy"
    return os.path.join(OUT_DIR, season, cat_readable, base)

def load_anomaly_safe(path: str):
    """Load a .npy file if it exists; otherwise return None."""
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)
        # validate 2D
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        return arr if arr.ndim == 2 else None
    except Exception:
        return None

def plot_4cats_with_shared_cbar(season: str, var: str, level=None, cmap="seismic", dpi=200):
    """
    For one (season, field[, level]) plot a 2x2 grid of anomaly maps
    (one per category) sharing the same symmetric colorbar.
    """
    # Collect anomalies
    anomalies = {}
    amax = 0.0
    for cat in CATEGORIES_ORDER:
        p = anomaly_path(season, cat, var, level)
        arr = load_anomaly_safe(p)
        anomalies[cat] = arr
        if arr is not None and np.isfinite(arr).any():
            amax = max(amax, float(np.nanmax(np.abs(arr))))

    if amax == 0.0:
        print(f"[SKIP] {season} / {var}{'' if level is None else f'_{level}'}: no valid anomalies.")
        return

    vmin, vmax = -amax, amax

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    # Title suffix for level
    title_field = var if level is None else f"{var}-{level} hPa"

    # Plot each category
    ims = []
    for ax, cat in zip(axes, CATEGORIES_ORDER):
        arr = anomalies[cat]
        if arr is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_title(f"{CATEGORY_TITLES.get(cat, cat)}", fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            ims.append(None)
            continue

        im = ax.imshow(arr, origin="lower", interpolation="nearest",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{CATEGORY_TITLES.get(cat, cat)}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)

    # Shared colorbar (attach to the first valid image)
    valid_ims = [im for im in ims if im is not None]
    if valid_ims:
        cbar = fig.colorbar(valid_ims[0], ax=axes, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Anomaly (category − season mean)", rotation=90)

    # Super title
    fig.suptitle(f"{season}  ·  {title_field}  ·  Anomaly (4 categories, shared color scale)", fontsize=16)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=0.8, h_pad=0.8)

    # Output path
    figs_dir = os.path.join(OUT_DIR, season, "_figs")
    os.makedirs(figs_dir, exist_ok=True)
    out_name = f"{var}_anomaly_4cats.png" if level is None else f"{var}_{level}_anomaly_4cats.png"
    out_path = os.path.join(figs_dir, out_name)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    print(f"[INFO] Creating 4-category anomaly figures from: {os.path.abspath(OUT_DIR)}")

    for season in SEASONS:
        print(f"\n[SEASON] {season}")
        for var, levels in FIELDS.items():
            if levels is None:
                plot_4cats_with_shared_cbar(season, var, level=None)
            else:
                for lev in levels:
                    plot_4cats_with_shared_cbar(season, var, level=lev)

    print("\n[DONE] All requested figures created.")
