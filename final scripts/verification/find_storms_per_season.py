# import os
# import glob
# import json
# import csv
# import torch
# import numpy as np
#
# # =============================================================================
# # CONFIG
# # =============================================================================
# MODEL_NAME = "CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast1h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"
# DL_BASE_DIRECTORY = f"/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model/{MODEL_NAME}"
# BASELINE_PRED_BASE_DIR = f"/home/mansour/ML3300-24a/shreibshtein/predictions_naive_31"
# STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"
# OUTPUT_DIR = "seasonal_event_lists"
#
# FOLDS_TO_AGGREGATE = ['45', '46', '47', '48']
#
# # Quantiles that define “small” and “big” per season (on |error|)
# SMALL_Q = 0.25
# BIG_Q   = 0.75
#
# # =============================================================================
# # HELPERS
# # =============================================================================
# def categorize_key(key: str) -> str:
#     """
#     Return one of: 'Winter', 'Summer', 'Spring', 'Fall', or 'Other'
#     Rules (case-insensitive):
#       - Winter: (DJF & NH) or (JJA & SH)
#       - Summer: (JJA & NH) or (DJF & SH)
#       - Spring: (MAM & NH) or (SON & SH)
#       - Fall:   (SON & NH) or (MAM & SH)
#     """
#     k = key.upper()
#     has = lambda s: s in k
#     nh = has("NH")
#     sh = has("SH")
#     if (has("DJF") and nh) or (has("JJA") and sh):
#         return "Winter"
#     if (has("JJA") and nh) or (has("DJF") and sh):
#         return "Summer"
#     if (has("MAM") and nh) or (has("SON") and sh):
#         return "Spring"
#     if (has("SON") and nh) or (has("MAM") and sh):
#         return "Fall"
#     return "Other"
#
# SEASONS = ["Winter", "Summer", "Spring", "Fall"]
#
# def iter_predictions():
#     """Yield (key, dl_error[np.ndarray], pers_error[np.ndarray], season) per sample."""
#     for fold_version in FOLDS_TO_AGGREGATE:
#         search_path = os.path.join(DL_BASE_DIRECTORY, f"version_{fold_version}", "checkpoints", "*predictions.pt")
#         prediction_files = glob.glob(search_path)
#         if not prediction_files:
#             print(f"[WARNING] No prediction file for fold {fold_version} at {search_path}")
#             continue
#         dl_predictions_file = prediction_files[0]
#         try:
#             predictions_data = torch.load(dl_predictions_file, map_location='cpu')
#         except Exception as e:
#             print(f"[ERROR] Could not load {dl_predictions_file}: {e}")
#             continue
#
#         for key, value in predictions_data.items():
#             try:
#                 y_hat_tensor, y_true_tensor = value[0].cpu(), value[1].cpu()
#                 # de-normalize to mbar
#                 denorm_y_pred = (y_hat_tensor[:, 0] * std_tensor[28] + mean_tensor[28]) / 100.0
#                 denorm_y_true = (y_true_tensor[:, 0] * std_tensor[28] + mean_tensor[28]) / 100.0
#
#                 dl_error = (denorm_y_pred - denorm_y_true).detach().numpy()
#
#                 year = key.split('_')[1]
#                 persistence_path = os.path.join(BASELINE_PRED_BASE_DIR, 'naive', year, key)
#                 persistence_pred_pa = np.load(persistence_path)
#                 persistence_pred_mbar = persistence_pred_pa / 100.0
#                 pers_error = persistence_pred_mbar - denorm_y_true.numpy()
#
#                 # Remove NaNs/Infs
#                 mask = np.isfinite(dl_error) & np.isfinite(pers_error)
#                 if not np.any(mask):
#                     continue
#                 dl_error = dl_error[mask]
#                 pers_error = pers_error[mask]
#
#                 season = categorize_key(key)
#                 yield key, dl_error, pers_error, season
#             except FileNotFoundError:
#                 # Missing persistence; skip silently
#                 pass
#             except Exception as e:
#                 print(f"[ERROR] Key '{key}' failed: {e}")
#
# # =============================================================================
# # MAIN
# # =============================================================================
# if __name__ == "__main__":
#     print("--- Start: categorize events by error/persistence-error ---")
#
#     # Load mean/std for de-normalization
#     try:
#         mean_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-mean.pt"), map_location='cpu')
#         std_tensor  = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-std.pt"),  map_location='cpu')
#         print("[INFO] Loaded mean/std.")
#     except Exception as e:
#         raise SystemExit(f"[FATAL] Failed loading stats from {STATS_DIRECTORY}: {e}")
#
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#     # -------------------------------------------------------------------------
#     # PASS 1: collect absolute errors per season to compute thresholds
#     # -------------------------------------------------------------------------
#     abs_err_per_season = {s: [] for s in SEASONS}   # |DL error|
#     abs_pers_per_season = {s: [] for s in SEASONS}  # |Persistence error|
#
#     # We also keep a lightweight cache in memory for re-use in pass 2 if desired
#     # (to avoid reloading files). This is optional; comment out to lower memory.
#     cache = []  # each item: (key, dl_err[np], pers_err[np], season)
#
#     for key, dl_err, pers_err, season in iter_predictions():
#         if season not in SEASONS:
#             continue
#         abs_err_per_season[season].extend(np.abs(dl_err).tolist())
#         abs_pers_per_season[season].extend(np.abs(pers_err).tolist())
#         cache.append((key, dl_err, pers_err, season))
#
#     thresholds = {}
#     for s in SEASONS:
#         dl_abs = np.asarray(abs_err_per_season[s]) if len(abs_err_per_season[s]) else np.array([0.0])
#         pe_abs = np.asarray(abs_pers_per_season[s]) if len(abs_pers_per_season[s]) else np.array([0.0])
#         small_dl = float(np.quantile(dl_abs, SMALL_Q))
#         big_dl   = float(np.quantile(dl_abs, BIG_Q))
#         small_pe = float(np.quantile(pe_abs, SMALL_Q))
#         big_pe   = float(np.quantile(pe_abs, BIG_Q))
#         thresholds[s] = {
#             "dl_small": small_dl,
#             "dl_big":   big_dl,
#             "pers_small": small_pe,
#             "pers_big":   big_pe,
#         }
#         print(f"[INFO] {s} thresholds | DL small<= {small_dl:.2f}, big>= {big_dl:.2f} | "
#               f"Pers small<= {small_pe:.2f}, big>= {big_pe:.2f}")
#
#     # Save thresholds for record
#     with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w") as f:
#         json.dump(thresholds, f, indent=2)
#
#     # -------------------------------------------------------------------------
#     # PASS 2: classify each (key, step) into the four categories
#     # -------------------------------------------------------------------------
#     results = {
#         s: {
#             "cat1_small_small": [],  # small DL & small Pers
#             "cat2_small_big":   [],  # small DL & big   Pers
#             "cat3_big_small":   [],  # big   DL & small Pers
#             "cat4_big_big":     [],  # big   DL & big   Pers
#         } for s in SEASONS
#     }
#
#     def classify_and_store(key, dl_err, pers_err, season):
#         th = thresholds[season]
#         for i, (e_dl, e_pe) in enumerate(zip(dl_err, pers_err)):
#             ae_dl = abs(float(e_dl))
#             ae_pe = abs(float(e_pe))
#             # Category rules
#             if (ae_dl <= th["dl_small"]) and (ae_pe <= th["pers_small"]):
#                 results[season]["cat1_small_small"].append(
#                     {"key": key, "step": int(i), "dl_error": float(e_dl), "pers_error": float(e_pe)}
#                 )
#             elif (ae_dl <= th["dl_small"]) and (ae_pe >= th["pers_big"]):
#                 results[season]["cat2_small_big"].append(
#                     {"key": key, "step": int(i), "dl_error": float(e_dl), "pers_error": float(e_pe)}
#                 )
#             elif (ae_dl >= th["dl_big"]) and (ae_pe <= th["pers_small"]):
#                 results[season]["cat3_big_small"].append(
#                     {"key": key, "step": int(i), "dl_error": float(e_dl), "pers_error": float(e_pe)}
#                 )
#             elif (ae_dl >= th["dl_big"]) and (ae_pe >= th["pers_big"]):
#                 results[season]["cat4_big_big"].append(
#                     {"key": key, "step": int(i), "dl_error": float(e_dl), "pers_error": float(e_pe)}
#                 )
#             # else: medium cases are ignored
#
#     # Use cached pass-1 data (fast). If you removed cache above, re-iterate with iter_predictions().
#     for key, dl_err, pers_err, season in cache:
#         if season in SEASONS:
#             classify_and_store(key, dl_err, pers_err, season)
#
#     # -------------------------------------------------------------------------
#     # WRITE OUTPUT CSVs
#     # -------------------------------------------------------------------------
#     def write_csv(path, rows):
#         with open(path, "w", newline="") as f:
#             w = csv.DictWriter(f, fieldnames=["key", "step", "dl_error_mbar", "persistence_error_mbar"])
#             w.writeheader()
#             for r in rows:
#                 w.writerow({
#                     "key": r["key"],
#                     "step": r["step"],
#                     "dl_error_mbar": f"{r['dl_error']:.4f}",
#                     "persistence_error_mbar": f"{r['pers_error']:.4f}",
#                 })
#
#     for s in SEASONS:
#         for cat, rows in results[s].items():
#             out_path = os.path.join(OUTPUT_DIR, f"{s}_{cat}.csv")
#             write_csv(out_path, rows)
#             print(f"[SAVE] {s} / {cat}: {len(rows)} rows -> {out_path}")
#
#     print("\n--- Done. Files written to:", os.path.abspath(OUTPUT_DIR))

import os
import glob
import json
import csv
import torch
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = "CNN_SKIP_CONNECTION-config-v5-msl-1979-2024-lookback3h-forecast1h-circular-norm-cnn-skip-empty_norm-max_pool-depth_8"
DL_BASE_DIRECTORY = f"/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/model/{MODEL_NAME}"
BASELINE_PRED_BASE_DIR = f"/home/mansour/ML3300-24a/shreibshtein/predictions_naive_31"
STATS_DIRECTORY = "/home/mansour/ML3300-24a/shreibshtein/DL-precipitation-prediction/stats"
OUTPUT_DIR = "seasonal_event_lists"

# NEW: intensity store
INTENSITY_BASE_DIR = "/home/mansour/ML3300-24a/omersela3/v5/v5_0/intensity"

FOLDS_TO_AGGREGATE = ['45', '46', '47', '48']

SMALL_Q = 0.25
BIG_Q   = 0.75

# =============================================================================
# HELPERS
# =============================================================================
def categorize_key(key: str) -> str:
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

SEASONS = ["Winter", "Summer", "Spring", "Fall"]

def extract_year_from_key(key: str):
    name = key[:-4] if key.endswith(".npy") else key
    for token in name.split("_"):
        if token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100:
            return token
    parts = name.split("_")
    if len(parts) > 1 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[1]
    return None

def intensity_path_for_key(key: str):
    year = extract_year_from_key(key) or ""
    fname = key if key.endswith(".npy") else f"{key}.npy"
    candidates = [
        os.path.join(INTENSITY_BASE_DIR, year, fname),
        os.path.join(INTENSITY_BASE_DIR, year, fname[:-4]),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def get_pos_of_max_intensity(key: str):
    """Return the time index (axis=0) where intensity is maximum."""
    fpath = intensity_path_for_key(key)
    if not os.path.exists(fpath):
        return ""
    try:
        arr = np.load(fpath)   # shape: (T, 1, 1)
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return int(np.nanargmax(arr[:, 0, 0]))
        elif arr.ndim == 1:
            return int(np.nanargmax(arr))
        else:
            # try squeezing and take axis=0
            flat = np.squeeze(arr)
            return int(np.nanargmax(flat))
    except Exception:
        return ""

def iter_predictions():
    for fold_version in FOLDS_TO_AGGREGATE:
        search_path = os.path.join(DL_BASE_DIRECTORY, f"version_{fold_version}", "checkpoints", "*predictions.pt")
        prediction_files = glob.glob(search_path)
        if not prediction_files:
            print(f"[WARNING] No prediction file for fold {fold_version} at {search_path}")
            continue
        dl_predictions_file = prediction_files[0]
        try:
            predictions_data = torch.load(dl_predictions_file, map_location='cpu')
        except Exception as e:
            print(f"[ERROR] Could not load {dl_predictions_file}: {e}")
            continue

        for key, value in predictions_data.items():
            try:
                y_hat_tensor, y_true_tensor = value[0].cpu(), value[1].cpu()
                storm_len = int(y_true_tensor.shape[0]) + 4

                denorm_y_pred = (y_hat_tensor[:, 0] * std_tensor[28] + mean_tensor[28]) / 100.0
                denorm_y_true = (y_true_tensor[:, 0] * std_tensor[28] + mean_tensor[28]) / 100.0
                dl_error = (denorm_y_pred - denorm_y_true).detach().numpy()

                year = key.split('_')[1]
                persistence_path = os.path.join(BASELINE_PRED_BASE_DIR, 'naive', year, key)
                persistence_pred_pa = np.load(persistence_path)
                persistence_pred_mbar = persistence_pred_pa / 100.0
                pers_error = persistence_pred_mbar - denorm_y_true.numpy()

                mask = np.isfinite(dl_error) & np.isfinite(pers_error)
                if not np.any(mask):
                    continue
                valid_idx = np.nonzero(mask)[0]
                dl_error = dl_error[mask]
                pers_error = pers_error[mask]

                season = categorize_key(key)

                pos_max_intensity = get_pos_of_max_intensity(key)

                yield key, dl_error, pers_error, season, storm_len, valid_idx, pos_max_intensity

            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[ERROR] Key '{key}' failed: {e}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("--- Start: categorize events by error/persistence-error ---")

    try:
        mean_tensor = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-mean.pt"), map_location='cpu')
        std_tensor  = torch.load(os.path.join(STATS_DIRECTORY, f"{MODEL_NAME}-std.pt"),  map_location='cpu')
        print("[INFO] Loaded mean/std.")
    except Exception as e:
        raise SystemExit(f"[FATAL] Failed loading stats from {STATS_DIRECTORY}: {e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    abs_err_per_season = {s: [] for s in SEASONS}
    abs_pers_per_season = {s: [] for s in SEASONS}

    cache = []  # (key, dl_err, pers_err, season, storm_len, valid_idx, pos_max_intensity)

    for tup in iter_predictions():
        key, dl_err, pers_err, season, storm_len, valid_idx, pos_max_intensity = tup
        if season not in SEASONS:
            continue
        abs_err_per_season[season].extend(np.abs(dl_err).tolist())
        abs_pers_per_season[season].extend(np.abs(pers_err).tolist())
        cache.append(tup)

    thresholds = {}
    for s in SEASONS:
        dl_abs = np.asarray(abs_err_per_season[s]) if len(abs_err_per_season[s]) else np.array([0.0])
        pe_abs = np.asarray(abs_pers_per_season[s]) if len(abs_pers_per_season[s]) else np.array([0.0])
        thresholds[s] = {
            "dl_small": float(np.quantile(dl_abs, SMALL_Q)),
            "dl_big":   float(np.quantile(dl_abs, BIG_Q)),
            "pers_small": float(np.quantile(pe_abs, SMALL_Q)),
            "pers_big":   float(np.quantile(pe_abs, BIG_Q)),
        }
        th = thresholds[s]
        print(f"[INFO] {s} thresholds | DL ≤{th['dl_small']:.2f} / ≥{th['dl_big']:.2f} | "
              f"PE ≤{th['pers_small']:.2f} / ≥{th['pers_big']:.2f}")

    with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    results = {s: {f"cat{i}": [] for i in range(1,5)} for s in SEASONS}

    def classify_and_store(key, dl_err, pers_err, season, storm_len, valid_idx, pos_max_intensity):
        th = thresholds[season]
        for i, (e_dl, e_pe) in enumerate(zip(dl_err, pers_err)):
            ae_dl = abs(float(e_dl))
            ae_pe = abs(float(e_pe))
            row = {
                "key": key,
                "step": int(i) + 4,
                "time_index": int(valid_idx[i]),
                "dl_error_mbar": float(e_dl),
                "persistence_error_mbar": float(e_pe),
                "storm_length": int(storm_len),
                "pos_max_intensity": pos_max_intensity,
            }
            if (ae_dl <= th["dl_small"]) and (ae_pe <= th["pers_small"]):
                results[season]["cat1"].append(row)
            elif (ae_dl <= th["dl_small"]) and (ae_pe >= th["pers_big"]):
                results[season]["cat2"].append(row)
            elif (ae_dl >= th["dl_big"]) and (ae_pe <= th["pers_small"]):
                results[season]["cat3"].append(row)
            elif (ae_dl >= th["dl_big"]) and (ae_pe >= th["pers_big"]):
                results[season]["cat4"].append(row)

    for tup in cache:
        classify_and_store(*tup)

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "key",
                    "step",
                    "time_index",
                    "dl_error_mbar",
                    "persistence_error_mbar",
                    "storm_length",
                    "pos_max_intensity",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)

    for s in SEASONS:
        for cat, rows in results[s].items():
            out_path = os.path.join(OUTPUT_DIR, f"{s}_{cat}.csv")
            write_csv(out_path, rows)
            print(f"[SAVE] {s}/{cat}: {len(rows)} rows -> {out_path}")

    print("\n--- Done. Files written to:", os.path.abspath(OUTPUT_DIR))

