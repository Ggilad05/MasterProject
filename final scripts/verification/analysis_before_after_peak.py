import os
import csv
import glob
from collections import defaultdict, namedtuple

import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_DIR = r"C:\Users\shrei\PycharmProjects\MasterProject\final scripts\save\seasonal_event_lists"        # where your season-category CSVs live
OUT_DIR = "peak_histograms"             # where to save plots + summaries
SEASONS = ["Winter", "Summer", "Spring", "Fall"]

# Support either old or new category file naming
#   - new exporter:  <Season>_cat1.csv .. cat4
#   - old exporter:  <Season>_cat1_small_small.csv .. etc
CAT_MAP = {
    "cat1": "DL small · PE small",
    "cat2": "DL small · PE big",
    "cat3": "DL big   · PE small",
    "cat4": "DL big   · PE big"
}

STEP_HOURS = 6  # every step is 6h → convert to hours for labels/titles

# =========================
# HELPERS
# =========================
def parse_cat_from_filename(path):
    base = os.path.splitext(os.path.basename(path))[0]
    # expected: Season_<cat token>
    parts = base.split("_", 1)
    if len(parts) < 2:
        return None, None
    season = parts[0]
    cat_token = parts[1]
    if season not in SEASONS:
        return None, None
    label = CAT_MAP.get(cat_token, cat_token)
    return season, label

def load_rows(csv_path):
    """Load rows, tolerant to column name variants."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames}  # map lower->actual
        # required columns
        need = ["key", "step", "pos_max_intensity"]
        for n in need:
            if n not in cols:
                raise ValueError(f"Required column '{n}' not found in {csv_path}. Found: {reader.fieldnames}")

        # optional: time_index (not used here), lengths, errors etc.
        for r in reader:
            try:
                key = r[cols["key"]]
                step = int(float(r[cols["step"]]))  # robust cast
                pos = int(float(r[cols["pos_max_intensity"]]))
                rows.append((key, step, pos))
            except Exception:
                # skip malformed lines
                continue
    return rows

def build_storm_sets(rows):
    """
    Given list of (key, step, pos_max), build:
      storms: dict key -> {"pos": int, "steps": set(int)}
    Assumes pos_max is identical for all rows of a given key (as exported).
    """
    storms = {}
    for key, step, pos in rows:
        if key not in storms:
            storms[key] = {"pos": pos, "steps": set()}
        storms[key]["steps"].add(step)
    return storms

def classify_storm(storm):
    """
    Return booleans (has_before, has_at, has_after) for one storm.
    'Before' means any step < pos, 'After' any step > pos, 'At' any step == pos.
    """
    steps = storm["steps"]
    pos = storm["pos"]
    has_before = any(s < pos for s in steps)
    has_at     = (pos in steps)
    has_after  = any(s > pos for s in steps)
    return has_before, has_at, has_after

def plot_three_bar(ax, counts, total_storms, title):
    """
    counts: dict {"Before": int, "At": int, "After": int}
    total_storms: denominator for normalization
    """
    labels = ["Before peak", "At peak", "After peak"]
    values = [counts["Before"], counts["At"], counts["After"]]
    fracs = [v / total_storms if total_storms > 0 else 0.0 for v in values]

    bars = ax.bar(labels, fracs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of storms")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # annotate absolute counts above bars
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"N={v}", ha="center", va="bottom", fontsize=9)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not csv_paths:
        raise SystemExit(f"[FATAL] No CSV files found under {CSV_DIR}")

    # season → category_label → plot info
    grouped = {s: {} for s in SEASONS}

    # also collect a small summary to CSV later
    summary_rows = []

    for path in csv_paths:
        season, cat_label = parse_cat_from_filename(path)
        if season is None:
            continue  # skip non-matching files

        rows = load_rows(path)
        if not rows:
            print(f"[WARN] Empty or invalid: {path}")
            continue

        # per-storm aggregation
        storms = build_storm_sets(rows)
        total_storms = len(storms)

        # count storms with any step before/at/after the peak
        n_before = n_at = n_after = 0
        for key, info in storms.items():
            has_before, has_at, has_after = classify_storm(info)
            n_before += 1 if has_before else 0
            n_at     += 1 if has_at else 0
            n_after  += 1 if has_after else 0

        counts = {"Before": n_before, "At": n_at, "After": n_after}

        # keep for plotting (one fig per season with up to 4 categories)
        grouped[season][cat_label] = (counts, total_storms)

        # save a per-file summary CSV line
        summary_rows.append({
            "season": season,
            "category": cat_label,
            "storms_total": total_storms,
            "storms_with_before": n_before,
            "storms_with_at": n_at,
            "storms_with_after": n_after,
            "frac_before": n_before / total_storms if total_storms > 0 else 0.0,
            "frac_at":     n_at     / total_storms if total_storms > 0 else 0.0,
            "frac_after":  n_after  / total_storms if total_storms > 0 else 0.0,
        })

    # ---------- PLOTS ----------
    for season in SEASONS:
        cats = grouped.get(season, {})
        if not cats:
            continue

        # layout (up to 4 categories)
        n = len(cats)
        cols = 2 if n > 1 else 1
        rows = 2 if n > 2 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(10, 6))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.ravel()

        for ax, (cat_label, (counts, total_storms)) in zip(axes, cats.items()):
            title = f"{season} · {cat_label}\n(step = {STEP_HOURS} h)"
            plot_three_bar(ax, counts, total_storms, title)

        # hide any extra axes
        for i in range(len(cats), len(axes)):
            axes[i].axis("off")

        fig.suptitle(f"Storm position relative to peak — {season}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_png = os.path.join(OUT_DIR, f"{season}_before_at_after_peak_hist.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[SAVE] {out_png}")

    # ---------- SUMMARY CSV ----------
    sum_csv = os.path.join(OUT_DIR, "peak_position_summary.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "season", "category", "storms_total",
                "storms_with_before", "storms_with_at", "storms_with_after",
                "frac_before", "frac_at", "frac_after"
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[SAVE] {sum_csv}")
