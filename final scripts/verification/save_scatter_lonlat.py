# save as: plot_maps_by_category.py
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Try Cartopy; fallback to plain scatter if unavailable
USE_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    USE_CARTOPY = False

# ============================
# CONFIG
# ============================
INPUT_CSV = "/data/shreibshtein/scripts/lonlat_outputs/lonlat_points_by_category.csv"  # <-- path to the CSV you saved earlier
OUT_DIR = "maps_by_category"

SEASONS = ["Winter", "Summer", "Spring", "Fall"]

CATEGORY_LABELS = {
    "cat1_small_small": "smallDL_smallPers",
    "cat2_small_big":   "smallDL_bigPers",
    "cat3_big_small":   "bigDL_smallPers",
    "cat4_big_big":     "bigDL_bigPers",
}
# One color per category (keeps figures visually consistent)
CATEGORY_COLORS = {
    "cat1_small_small": "tab:blue",
    "cat2_small_big":   "tab:red",
    "cat3_big_small":   "tab:green",
    "cat4_big_big":     "gold",
}

# ============================
# HELPERS
# ============================
def read_points(csv_path):
    """
    Returns a list of dicts with keys: season, category, key, step, lon, lat
    """
    pts = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
                season = row["season"]
                category = row["category"]
                step = int(row["step"])
                key = row["key"]
                # Basic sanity on coordinates
                if lon > 180: lon -= 360.0
                if lon < -180: lon += 360.0
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    continue
                pts.append({
                    "season": season, "category": category, "key": key,
                    "step": step, "lon": lon, "lat": lat
                })
            except Exception:
                continue
    return pts

def group_by_category_and_season(points):
    grouped = {cat: {s: [] for s in SEASONS} for cat in CATEGORY_LABELS.keys()}
    for p in points:
        cat = p["category"]
        sea = p["season"]
        if cat in grouped and sea in grouped[cat]:
            grouped[cat][sea].append(p)
    return grouped

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"CSV not found: {INPUT_CSV}")

    pts = read_points(INPUT_CSV)
    if not pts:
        raise SystemExit("No rows found in CSV (or all invalid coordinates).")

    grouped = group_by_category_and_season(pts)

    for cat_code, per_season in grouped.items():
        cat_name = CATEGORY_LABELS.get(cat_code, cat_code)
        color = CATEGORY_COLORS.get(cat_code, "tab:gray")

        if USE_CARTOPY:
            fig = plt.figure(figsize=(14, 10))
            axes = [
                plt.subplot(2, 2, i + 1, projection=ccrs.PlateCarree())
                for i in range(4)
            ]
            for ax, season in zip(axes, SEASONS):
                ax.set_global()
                ax.coastlines(linewidth=0.7)
                ax.add_feature(cfeature.BORDERS, linewidth=0.4)
                ax.gridlines(draw_labels=False, linewidth=0.3, linestyle="--")

                rows = per_season.get(season, [])
                xs = [r["lon"] for r in rows]
                ys = [r["lat"] for r in rows]
                if len(xs) > 0:
                    ax.scatter(xs, ys, s=12, alpha=0.85,
                               transform=ccrs.PlateCarree(),
                               color=color, marker="o")
                ax.set_title(f"{season} (N={len(xs)})", fontsize=13)

            fig.suptitle(f"Event Locations — {cat_name}", fontsize=16)
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
            axes = axes.ravel()
            for ax, season in zip(axes, SEASONS):
                rows = per_season.get(season, [])
                xs = [r["lon"] for r in rows]
                ys = [r["lat"] for r in rows]
                if len(xs) > 0:
                    ax.scatter(xs, ys, s=12, alpha=0.85, color=color, marker="o")
                ax.set_xlim([-180, 180]); ax.set_ylim([-90, 90])
                ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                ax.grid(True, linestyle="--", linewidth=0.3)
                ax.set_title(f"{season} (N={len(xs)})", fontsize=13)

            fig.suptitle(f"Event Locations — {cat_name}", fontsize=16)
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])

        out_png = os.path.join(OUT_DIR, f"{cat_code}_4maps.png")
        plt.savefig(out_png, dpi=250)
        plt.close()
        print(f"[SAVE] {out_png}")

    print("\n[DONE] Wrote per-category 2x2 maps to:", os.path.abspath(OUT_DIR))
