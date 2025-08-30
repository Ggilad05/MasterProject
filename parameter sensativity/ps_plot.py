# plot_overall_sensitivity_bar_v5.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---- Data (updated) ----
data = {
    "Parameter": [
        "msl","tp","z","sshf","ta_850",
        "ua_300","za_300","intensity","za_850","sp"
    ],
    "Overall_Mean_Sens": [
        0.146452082, 0.044237786, 0.029408357, 0.012716153, 0.009788097,
        0.009092183, 0.007832669, 0.006399036, 0.006324123, 0.006038255
    ],
    "Overall_Std_Sens": [0,0,0,0,0,0,0,0,0,0],
}

df = pd.DataFrame(data).sort_values("Overall_Mean_Sens", ascending=False).reset_index(drop=True)

# ---- Plot ----
fig_height = max(6, 0.45 * len(df) + 2)
fig, ax = plt.subplots(figsize=(10, fig_height))

# Only add error bars if any std > 0
xerr = df["Overall_Std_Sens"].values
use_err = np.any(np.asarray(xerr) > 0)

ax.barh(
    df["Parameter"],
    df["Overall_Mean_Sens"],
    xerr=xerr if use_err else None,
    capsize=4 if use_err else 0,
    edgecolor="black"
)

ax.invert_yaxis()
ax.set_xlabel("MAE change [mbar]", fontsize=20)
ax.set_ylabel("Parameter", fontsize=20)
ax.set_title("Parameter Sensitivity Winter (6H)", fontsize=30)
ax.grid(axis="x", linestyle="--", alpha=0.6)
ax.set_xlim(0, 1)

# Value labels to the right of bars (kept inside axis)
offset_map = {0: 0.02}  # slightly bigger offset for the top bar
for i, mean_val in enumerate(df["Overall_Mean_Sens"]):
    offset = offset_map.get(i, 0.02)
    x_text = min(mean_val + offset, 0.98)
    ha = "left" if x_text >= mean_val else "right"
    ax.text(x_text, i, f"{mean_val:.3f}", va="center", ha=ha, fontsize=15)
# Tick label sizes
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
# plt.savefig("overall_sensitivity_bar_scaled_v5.png", dpi=200)
# print("Saved figure to overall_sensitivity_bar_scaled_v5.png")
plt.show()
