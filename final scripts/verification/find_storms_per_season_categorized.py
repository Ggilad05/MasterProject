import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # regression

# =============================================================================
# CONFIG
# =============================================================================
OUTPUT_DIR = r"C:\Users\shrei\PycharmProjects\MasterProject\final scripts\verification\seasonal_event_lists"   # where your CSVs + thresholds.json are
OUTPUT_PLOT_FILE_ALL = "dl_vs_pers__from_csv_all.png"
OUTPUT_PLOT_FILE_SEASONS = "dl_vs_pers__from_csv_seasons.png"

# Seasons and category files created by your exporter
SEASONS = ["Winter", "Summer", "Spring", "Fall"]
CATEGORIES = {
    "cat1_small_small": dict(label="DL small & PE small", color="blue"),
    "cat2_small_big":   dict(label="DL small & PE big",   color="green"),
    "cat3_big_small":   dict(label="DL big & PE small",   color="orange"),
    "cat4_big_big":     dict(label="DL big & PE big",     color="red"),
}

# Marker per season (optional; helps visually separate seasons on the ALL plot)
SEASON_MARKERS = {
    "Winter": "o",
    "Summer": "s",
    "Spring": "D",
    "Fall":   "^",
}

POINT_SIZE = 14
ALPHA = 0.7

# =============================================================================
# HELPERS
# =============================================================================
def load_thresholds(path):
    """Load thresholds.json if present (for reference / titles)."""
    thr_path = os.path.join(path, "thresholds.json")
    if not os.path.exists(thr_path):
        return None
    with open(thr_path, "r") as f:
        return json.load(f)

def load_category_csv(path):
    """
    Read one category CSV (with headers: key, step, dl_error_mbar, persistence_error_mbar).
    Returns arrays x (pers error), y (dl error).
    """
    xs, ys = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # stored as strings with formatting; cast to float
                y = float(row["dl_error_mbar"])
                x = float(row["persistence_error_mbar"])
                xs.append(x)
                ys.append(y)
            except Exception:
                continue
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

def add_y_equals_x_and_fit(ax, x, y, color_fit="purple", lw_fit=2.0, show_fit=True, fit_label_prefix="Linear fit"):
    """Add y=x and (optional) linear fit line + annotate R^2."""
    if x.size == 0:
        return None
    # y=x line based on joint limits
    xy = np.concatenate([x, y])
    lo, hi = float(np.min(xy)), float(np.max(xy))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", alpha=0.4, lw=1.2, label="y = x")

    if show_fit and x.size >= 2:
        slope, intercept, r, p, se = stats.linregress(x, y)
        fit_x = np.array([lo, hi])
        fit_y = intercept + slope * fit_x
        ax.plot(fit_x, fit_y, "--", color=color_fit, lw=lw_fit,
                label=f"{fit_label_prefix}: y={slope:.2f}x+{intercept:.2f}, $R^2$={r*r:.3f}")

def set_square_limits(ax, x, y, pad_ratio=0.05):
    """Square axes with a small padding around min/max of data."""
    if x.size == 0:
        return
    data_min = float(np.min([x.min(), y.min()]))
    data_max = float(np.max([x.max(), y.max()]))
    span = data_max - data_min
    pad = pad_ratio * (span if span > 0 else 1.0)
    lo, hi = data_min - pad, data_max + pad
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("[INFO] Loading categorized CSVs from:", os.path.abspath(OUTPUT_DIR))

    thresholds = load_thresholds(OUTPUT_DIR)
    if thresholds is None:
        print("[WARN] thresholds.json not found (that's OK). Colors are still applied by category.")
    else:
        # Optional: print a small summary
        for s in SEASONS:
            th = thresholds.get(s, {})
            if th:
                print(f"[THR] {s}: DL≤{th.get('dl_small', np.nan):.3f} / DL≥{th.get('dl_big', np.nan):.3f}; "
                      f"PE≤{th.get('pers_small', np.nan):.3f} / PE≥{th.get('pers_big', np.nan):.3f}")

    # Gather data: per season per category
    data = {s: {cat: {"x": np.array([]), "y": np.array([])} for cat in CATEGORIES} for s in SEASONS}

    total_points = 0
    for season in SEASONS:
        for cat in CATEGORIES:
            csv_path = os.path.join(OUTPUT_DIR, f"{season}_{cat}.csv")
            x, y = load_category_csv(csv_path)
            data[season][cat]["x"] = x
            data[season][cat]["y"] = y
            total_points += x.size

    if total_points == 0:
        raise SystemExit("[FATAL] No points found in CSVs. Did you run the exporter / write the CSVs?")

    # ===========================
    # FIGURE 1: ALL points (only highlighted categories)
    # ===========================
    print("[INFO] Plotting ALL-points figure (highlighted categories only)")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_all, ax_all = plt.subplots(figsize=(12, 8))

    # Plot each season × category with consistent category color and season marker
    for season in SEASONS:
        mk = SEASON_MARKERS.get(season, "o")
        for cat, meta in CATEGORIES.items():
            x = data[season][cat]["x"]
            y = data[season][cat]["y"]
            if x.size == 0:
                continue
            label = f"{meta['label']} – {season} (N={x.size})"
            ax_all.scatter(x, y, s=POINT_SIZE, alpha=ALPHA, color=meta["color"], marker=mk, label=label)

    # Regression and y=x on combined highlighted points
    all_x = np.concatenate([data[s][c]["x"] for s in SEASONS for c in CATEGORIES if data[s][c]["x"].size > 0])
    all_y = np.concatenate([data[s][c]["y"] for s in SEASONS for c in CATEGORIES if data[s][c]["y"].size > 0])
    add_y_equals_x_and_fit(ax_all, all_x, all_y, color_fit="purple", lw_fit=2.0, show_fit=True,
                           fit_label_prefix="Fit (all highlighted)")

    ax_all.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_all.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_all.set_title("DL Error vs Persistence Error — Highlighted by Threshold Categories (from CSV)", fontsize=16)
    ax_all.set_xlabel("Persistence Error (Persistence − True) [mbar]", fontsize=13)
    ax_all.set_ylabel("DL Error (Prediction − True) [mbar]", fontsize=13)
    set_square_limits(ax_all, all_x, all_y)
    ax_all.tick_params(axis='both', which='major', labelsize=11)
    ax_all.legend(fontsize=9, loc="best", ncols=2)
    fig_all.tight_layout()
    fig_all.savefig(OUTPUT_PLOT_FILE_ALL, dpi=300)
    print(f"[SAVE] {OUTPUT_PLOT_FILE_ALL}")
    plt.close(fig_all)

    # ===========================
    # FIGURE 2: Four seasonal panels
    # ===========================
    print("[INFO] Plotting seasonal 2×2 panels")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, season in zip(axes, SEASONS):
        # Plot points per category
        season_points = 0
        for cat, meta in CATEGORIES.items():
            x = data[season][cat]["x"]
            y = data[season][cat]["y"]
            if x.size == 0:
                continue
            season_points += x.size
            ax.scatter(x, y, s=POINT_SIZE, alpha=ALPHA, color=meta["color"], marker=SEASON_MARKERS.get(season, "o"),
                       label=f"{meta['label']} (N={x.size})")

        # y=x + seasonal fit (on highlighted points of that season only)
        sx = np.concatenate([data[season][c]["x"] for c in CATEGORIES if data[season][c]["x"].size > 0]) \
             if season_points > 0 else np.array([])
        sy = np.concatenate([data[season][c]["y"] for c in CATEGORIES if data[season][c]["y"].size > 0]) \
             if season_points > 0 else np.array([])

        add_y_equals_x_and_fit(ax, sx, sy, color_fit="purple", lw_fit=1.8, show_fit=True,
                               fit_label_prefix=f"Fit ({season})")

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{season} (N={season_points})", fontsize=13)
        set_square_limits(ax, sx, sy)
        ax.grid(True)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("DL Error vs Persistence Error — Seasonal Panels (from CSV)", fontsize=18)
    fig.supxlabel("Persistence Error (Persistence − True) [mbar]", fontsize=14)
    fig.supylabel("DL Error (Prediction − True) [mbar]", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_PLOT_FILE_SEASONS, dpi=300)
    print(f"[SAVE] {OUTPUT_PLOT_FILE_SEASONS}")
    plt.close(fig)

    print("[DONE] Plots created from CSVs only.")
