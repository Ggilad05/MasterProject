#!/usr/bin/env python3
"""
Plots seasonal MAE results for DL, Transfer Learning, and Persistence.

Reads CSVs from a directory (matching a filename pattern),
computes weighted seasonal MAEs (NH + SH paired), and produces:

1) MAE vs Forecast Horizon (2x2 figure with one subplot per season)
2) Absolute difference plot: (Transfer − DL)
3) Relative difference plot: (Transfer − DL)/DL × 100

Author: Gilad
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# ------------------------
# CONFIG
# ------------------------
DATA_DIRECTORY = r"C:\Users\shrei\PycharmProjects\MasterProject\final scripts"
CSV_PATTERN = "mae_report_analysis_*.csv"

OFFSET_MAPPING = {31: 6, 32: 12, 33: 18, 34: 24}

SEASON_ROWS = {"NH_DJF","NH_MAM","NH_JJA","NH_SON","SH_DJF","SH_MAM","SH_JJA","SH_SON"}
PAIRS = {
    "Winter": ("NH_DJF", "SH_JJA"),
    "Spring": ("NH_MAM", "SH_SON"),
    "Summer": ("NH_JJA", "SH_DJF"),
    "Autumn": ("NH_SON", "SH_MAM"),
}

# Transfer Learning results provided manually
TRANSFER = {
    "6h":  {"Winter": 1.45, "Spring": 1.24, "Summer": 1.00, "Autumn": 1.30},
    "12h": {"Winter": 2.19, "Spring": 1.91, "Summer": 1.53, "Autumn": 1.88},
    "18h": {"Winter": 3.35, "Spring": 2.83, "Summer": 2.26, "Autumn": 2.97},
    "24h": {"Winter": 4.20, "Spring": 3.48, "Summer": 2.92, "Autumn": 3.77},
}

# Output file names
OUT_MAE_ALL = "mae_vs_forecast_all_seasons.png"
OUT_DIFF_ABS = "diff_TL_minus_DL.png"
OUT_DIFF_REL = "rel_diff_TL_vs_DL_percent.png"

# ------------------------
# HELPERS
# ------------------------
def compute_weighted_mae(df: pd.DataFrame, value_col: str, weight_col: str) -> dict[str, float]:
    out = {}
    for season, (nh, sh) in PAIRS.items():
        nh_row = df.loc[df["Category"] == nh].iloc[0]
        sh_row = df.loc[df["Category"] == sh].iloc[0]
        num = nh_row[value_col] * nh_row[weight_col] + sh_row[value_col] * sh_row[weight_col]
        den = nh_row[weight_col] + sh_row[weight_col]
        out[season] = float(num / den)
    return out

def load_weighted_summaries() -> tuple[dict, dict]:
    summary_dl, summary_pers = {}, {}
    for path in glob.glob(str(Path(DATA_DIRECTORY) / CSV_PATTERN)):
        # Extract the number (31, 32, …) to map to horizon
        stem = Path(path).stem
        num = int(stem.split("_")[-1])
        offset_hours = OFFSET_MAPPING[num]
        label = f"{offset_hours}h"

        df = pd.read_csv(path)
        df_seasons = df[df["Category"].isin(SEASON_ROWS)].copy()
        summary_dl[label]   = compute_weighted_mae(df_seasons, "DL_MAE", "DL_N")
        summary_pers[label] = compute_weighted_mae(df_seasons, "PERSISTENCE_MAE", "PERSISTENCE_N")
    return summary_dl, summary_pers

def to_series_by_season(summary: dict, season: str) -> list[float]:
    """Extract [6h, 12h, 18h, 24h] values for a given season."""
    offsets = ["6h", "12h", "18h", "24h"]
    return [summary[o][season] for o in offsets]

# ------------------------
# PLOTS
# ------------------------
def plot_mae_vs_forecast_all(summary_dl: dict, summary_tl: dict, summary_pers: dict) -> None:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        dl = to_series_by_season(summary_dl, season)
        tl = to_series_by_season(summary_tl, season)
        pe = to_series_by_season(summary_pers, season)

        ax.plot(["6h","12h","18h","24h"], dl, marker="o", label="DL")
        ax.plot(["6h","12h","18h","24h"], tl, marker="s", label="Transfer")
        ax.plot(["6h","12h","18h","24h"], pe, marker="^", label="Persistence")
        ax.set_title(season)
        ax.grid(True)

    fig.suptitle("MAE vs Forecast Horizon — All Seasons", fontsize=14)
    fig.text(0.5, 0.04, "Forecast Horizon", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "MAE", va="center", rotation="vertical", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False)

    # plt.tight_layout(rect=[0.03, 0.06, 1, 0.92])
    plt.savefig(Path(DATA_DIRECTORY) / OUT_MAE_ALL, dpi=180, bbox_inches="tight")
    plt.close()

def plot_diff_abs(summary_dl: dict, summary_tl: dict) -> None:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    plt.figure(figsize=(8, 6))
    for season in seasons:
        dl = to_series_by_season(summary_dl, season)
        tl = to_series_by_season(summary_tl, season)
        diffs = [t - d for t, d in zip(tl, dl)]
        plt.plot(["6h","12h","18h","24h"], diffs, marker="o", label=season)

    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.title("Difference in MAE: Transfer − DL")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("MAE Difference")
    plt.legend(title="Season")
    plt.grid(True)
    plt.savefig(Path(DATA_DIRECTORY) / OUT_DIFF_ABS, dpi=180, bbox_inches="tight")
    plt.close()

def plot_diff_rel(summary_dl: dict, summary_tl: dict) -> None:
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    plt.figure(figsize=(8, 6))
    for season in seasons:
        dl = to_series_by_season(summary_dl, season)
        tl = to_series_by_season(summary_tl, season)
        rel = [ (t - d) / d * 100.0 for t, d in zip(tl, dl) ]
        plt.plot(["6h","12h","18h","24h"], rel, marker="o", label=season)

    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.title("Relative Difference in MAE: (Transfer − DL) / DL")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("Relative Difference (%)")
    plt.legend(title="Season")
    plt.grid(True)
    plt.savefig(Path(DATA_DIRECTORY) / OUT_DIFF_REL, dpi=180, bbox_inches="tight")
    plt.close()

# ------------------------
# MAIN
# ------------------------
def main():
    summary_dl, summary_pers = load_weighted_summaries()
    summary_tl = TRANSFER

    plot_mae_vs_forecast_all(summary_dl, summary_tl, summary_pers)
    plot_diff_abs(summary_dl, summary_tl)
    plot_diff_rel(summary_dl, summary_tl)
    print("All plots saved in", DATA_DIRECTORY)

if __name__ == "__main__":
    main()
