#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Inputs
# =========================
# Forecast MAE (mbar)
forecast_mae = {
    6:  0.71,
    12: 0.93,
    18: 1.05,
    24: 1.25,
}

# Cross-validation MAEs per version (mbar) — original sets
dl_mae = {
    6:  [1.1240, 1.0840, 1.1388, 1.1375],
    12: [1.7666, 1.7465, 1.5849, 1.5294],
    18: [2.4393, 2.6851, 2.7101, 2.6618],
    24: [3.4086, 3.2911, 3.4631, 3.6157],
}
persistence_mae = {
    6:  [2.5311, 2.4914, 2.4756, 2.6369],
    12: [3.9772, 4.1581, 3.9665, 3.8842],
    18: [5.4654, 5.5470, 5.5433, 5.8663],
    24: [6.8369, 6.7634, 6.7645, 7.2809],
}
naive_mae = {
    6:  [2.5827, 2.5393, 2.5870, 2.6500],
    12: [4.0481, 4.0760, 3.9680, 3.7360],
    18: [5.8542, 6.1480, 6.3390, 6.4146],
    24: [8.3844, 8.3805, 8.8412, 9.2018],
}

# =========================
# NEW: Leakage (DL) MAEs per version (mbar)
# =========================
leakage_dl_mae = {
    # 6h: V11, V12, V13, V16, V17
    6:  [0.8221, 0.8934, 0.8310, 0.9794, 0.8959],
    # 12h: V3, V5, V6, V7, V8
    12: [1.4874, 1.4373, 1.4196, 1.4083, 1.4871],
    # 18h: V3, V5, V7, V8, V10
    18: [2.1561, 2.1908, 2.1418, 2.0108, 2.0365],
    # 24h: V3, V5, V7, V8, V10
    24: [2.6438, 2.7669, 2.6573, 2.7290, 2.5980],
}

# =========================
# Helpers
# =========================
steps = [6, 12, 18, 24]

def mean_std(values_by_step):
    means, stds = [], []
    for s in steps:
        arr = np.asarray(values_by_step[s], dtype=float)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr, ddof=1)))  # sample std across folds
    return np.array(means), np.array(stds)

def skill_score_from_pairs(model_errs, ref_errs):
    """
    Compute skill score using paired model/ref errors if lengths match,
    otherwise use SS against the reference MEAN.
    """
    m = np.asarray(model_errs, dtype=float)
    r = np.asarray(ref_errs, dtype=float)

    if m.size == r.size:
        # Per-fold SS then average (more faithful when folds align)
        ss = 1.0 - (m / r)
        return float(np.mean(ss))
    else:
        # Fallback: SS using mean reference error
        r_mean = float(np.mean(r))
        m_mean = float(np.mean(m))
        return 1.0 - (m_mean / r_mean)

def skill_score_scalar(model_err, ref_err):
    return 1.0 - (float(model_err) / float(ref_err))

# =========================
# Aggregation
# =========================
dl_mean, dl_std       = mean_std(dl_mae)
pe_mean, pe_std       = mean_std(persistence_mae)
nn_mean, nn_std       = mean_std(naive_mae)
leak_mean, leak_std   = mean_std(leakage_dl_mae)
forecast_vec          = np.array([forecast_mae[s] for s in steps], dtype=float)

# =========================
# Compute Skill Scores & Leakage Improvements
# =========================
ss_dl_vs_pers     = []
ss_leak_vs_pers   = []
ss_fore_vs_pers   = []
leak_improve_abs  = []
leak_improve_pct  = []

for i, s in enumerate(steps):
    # SS for DL vs Persistence — prefer per-fold pairing
    ss_dl = skill_score_from_pairs(dl_mae[s], persistence_mae[s])
    ss_dl_vs_pers.append(ss_dl)

    # SS for Leakage vs Persistence — lengths differ, so vs persistence MEAN
    ss_leak = skill_score_from_pairs(leakage_dl_mae[s], persistence_mae[s])
    ss_leak_vs_pers.append(ss_leak)

    # SS for Forecast vs Persistence (scalars)
    ss_fore = skill_score_scalar(forecast_vec[i], pe_mean[i])
    ss_fore_vs_pers.append(ss_fore)

    # Leakage improvement vs original DL (Δ and %)
    d_mae = dl_mean[i] - leak_mean[i]
    leak_improve_abs.append(d_mae)
    leak_improve_pct.append(100.0 * (d_mae / dl_mean[i]))

ss_dl_vs_pers   = np.array(ss_dl_vs_pers)
ss_leak_vs_pers = np.array(ss_leak_vs_pers)
ss_fore_vs_pers = np.array(ss_fore_vs_pers)
leak_improve_abs = np.array(leak_improve_abs)
leak_improve_pct = np.array(leak_improve_pct)

# =========================
# Summary table
# =========================
rows = []
for i, s in enumerate(steps):
    rows.append({
        "Step (h)": s,
        "DL mean MAE": dl_mean[i],
        "DL std": dl_std[i],
        "Leakage DL mean MAE": leak_mean[i],
        "Leakage DL std": leak_std[i],
        "Persistence mean MAE": pe_mean[i],
        "Persistence std": pe_std[i],
        "Naive mean MAE": nn_mean[i],
        "Naive std": nn_std[i],
        "Forecast MAE": forecast_vec[i],
        "SS(DL|Persistence)": ss_dl_vs_pers[i],
        "SS(Leakage|Persistence)": ss_leak_vs_pers[i],
        "SS(Forecast|Persistence)": ss_fore_vs_pers[i],
        "Leakage gain ΔMAE": leak_improve_abs[i],
        "Leakage gain %": leak_improve_pct[i],
    })
summary_df = pd.DataFrame(rows)

print("\n=== Summary vs Persistence (MAE in mbar; Skill Scores unitless) ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# =========================
# Save artifacts
# =========================
out_dir = Path("mae_compare_outputs")
out_dir.mkdir(exist_ok=True, parents=True)
summary_df.to_csv(out_dir / "mae_summary_with_skill.csv", index=False)

# =========================
# Plot: Lines + shaded std (no explicit colors)
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

# DL
ax.plot(steps, dl_mean, marker="o", label="DL mean")
ax.fill_between(steps, dl_mean - dl_std, dl_mean + dl_std, alpha=0.2)

# Leakage DL
ax.plot(steps, leak_mean, marker="v", label="Leakage (DL) mean")
ax.fill_between(steps, leak_mean - leak_std, leak_mean + leak_std, alpha=0.2)

# Persistence
ax.plot(steps, pe_mean, marker="s", label="Persistence mean")
ax.fill_between(steps, pe_mean - pe_std, pe_mean + pe_std, alpha=0.2)

# Naive Linear
ax.plot(steps, nn_mean, marker="^", label="Naive Linear mean")
ax.fill_between(steps, nn_mean - nn_std, nn_mean + nn_std, alpha=0.2)

# Forecast (no std)
ax.plot(steps, forecast_vec, marker="d", linewidth=2.0, label="Forecast MAE")

ax.set_title("MAE vs Forecast Steps (mbar)")
ax.set_xlabel("Forecast Step (hours)")
ax.set_ylabel("MAE (mbar)")
ax.set_xticks(steps, [str(s) for s in steps])
ax.grid(True, axis='y', alpha=0.3)
ax.legend()

png_path = out_dir / "mae_lines_vs_forecast_with_leakage.png"
plt.tight_layout()
plt.show()
# plt.savefig(png_path, dpi=150); plt.close()
# print(f"\nSaved:\n - CSV: {out_dir / 'mae_summary_with_skill
