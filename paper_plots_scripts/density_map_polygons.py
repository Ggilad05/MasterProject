
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from shapely.geometry import Polygon

# ------------- USER SETTINGS -------------
NPY_PATH = r"C:\Users\shrei\PycharmProjects\MasterProject\paper_plots_scripts\density map\mean_1979_2020.npy"  # path to your saved npy
TAPX = 9   # lon bin size (deg) used when creating the npy
TAPY = 3   # lat bin size (deg) used when creating the npy
FIG_DPI = 300
CMAP = "Reds"
TITLE = "Storm Tracking Density (1979–2020)"
# ----------------------------------------


def load_density(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D density array, got shape {arr.shape}")
    return arr


def grid_edges(tapx: float, tapy: float):
    """
    Build latitude/longitude edges for pcolormesh, consistent with a density grid
    of shape (ysize, xsize) where ysize = 180/tapy + 1, xsize = 360/tapx + 1.
    Centers are at -90..90 and 0..360 (inclusive) with steps tapy/tapx.
    """
    lat_centers = np.arange(-90, 90 + tapy, tapy)
    lon_centers = np.arange(0, 360 + tapx, tapx)

    # Convert centers -> edges
    lat_edges = np.concatenate([
        [lat_centers[0] - tapy / 2],
        (lat_centers[:-1] + lat_centers[1:]) / 2,
        [lat_centers[-1] + tapy / 2],
    ])
    lon_edges = np.concatenate([
        [lon_centers[0] - tapx / 2],
        (lon_centers[:-1] + lon_centers[1:]) / 2,
        [lon_centers[-1] + tapx / 2],
    ])

    # Clip lat edges to valid bounds
    lat_edges = np.clip(lat_edges, -90, 90)
    return lon_edges, lat_edges


def main(npy_path: str, tapx: float, tapy: float):
    density = load_density(npy_path)

    # Build edges and validate shape
    lon_edges, lat_edges = grid_edges(tapx, tapy)
    if density.shape != (len(lat_edges) - 1, len(lon_edges) - 1):
        raise ValueError(
            f"density shape {density.shape} not compatible with edges "
            f"({len(lat_edges)-1}, {len(lon_edges)-1}). "
            f"Check TAPX/TAPY match the npy file."
        )

    # Polygons (exactly as provided; no longitude wrapping)
    polygons = {
        "North Atlantic": Polygon([(-80, 35), (-10, 35), (-10, 70), (-80, 70)]),


        "North Pacific": Polygon([(-180, 30), (-120, 30), (-120, 70), (-180, 70)]),

        "Mediterranean": Polygon([(-10, 30), (50, 30), (50, 45), (-10, 45)]),


        "Southern Ocean": Polygon([(-180, -70), (180, -70), (180, -30), (-180, -30)]),

        "East Asia / North Pacific": Polygon([(120, 30), (180, 30), (180, 60), (120, 60)])
    }

    # Output next to the npy
    out_dir = Path(npy_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / (Path(npy_path).stem + ".png")

    # --- Plot ---
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 5.5))

    # Base map
    ax.set_global()
    ax.add_feature(cfeature.LAND, edgecolor="black", linewidth=0.3, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor="whitesmoke", zorder=0)

    # Graticule
    gl = ax.gridlines(crs=proj, draw_labels=True,
                      linewidth=.5, color='gray', alpha=0.6, linestyle='-.')
    try:
        gl.top_labels = False
        gl.right_labels = False
    except Exception:
        pass

    # Density via pcolormesh (needs edges)
    LonE, LatE = np.meshgrid(lon_edges, lat_edges)
    density_plot = np.ma.masked_where(density == 0, density)  # optional: hide zeros

    pcm = ax.pcolormesh(LonE, LatE, density_plot, transform=proj, cmap=CMAP)
    cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", pad=0.03, shrink=0.9)
    cbar.set_label("Track density (counts)")

    # Add polygons exactly as given
    for name, poly in polygons.items():
        feat = ShapelyFeature([poly], crs=proj, edgecolor="black", facecolor="none", linewidth=1.2, zorder=3)
        ax.add_feature(feat)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, name, fontsize=8, transform=proj,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))

    ax.set_title(TITLE)
    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    print(f"Saved figure: {out_png}")
    plt.show()


if __name__ == "__main__":
    main(NPY_PATH, TAPX, TAPY)
