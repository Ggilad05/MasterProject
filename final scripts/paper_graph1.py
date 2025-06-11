import matplotlib.pyplot as plt
import numpy as np

# === MAE data ===

# 4, 5, 6, 12, 17, 32 (6 elements)

# model_31_mae = [1.0586, 1.0551, 1.0731, 0.9890, 1.0577, 1.0422]
# model_31_mae_mean = np.mean(model_31_mae)
#
# Persistence_31_mae = [2.3421, 2.3466, 2.3483, 2.3447, 2.3405, 2.3372]
# Persistence_31_mae_mean = np.mean(Persistence_31_mae)
#
# linear_31_mae = [2.2129, 2.2143, 2.2083, 2.2039, 2.2010, 2.2211]
# linear_31_mae_mean = np.mean(linear_31_mae)

# Folds - 45, 46, 47, 48
model_31_mae = [1.1613, 1.1669, 1.1585, 1.1724]
model_31_mae_mean = np.mean(model_31_mae)

Persistence_31_mae = [2.3201, 2.31, 2.3192, 2.3221]
Persistence_31_mae_mean = np.mean(Persistence_31_mae)

linear_31_mae = [2.1286, 2.1225, 2.1263, 2.1278]
linear_31_mae_mean = np.mean(linear_31_mae)

# 3, 4, 5, 6, 7 (5 elements)
# model_32_mae = [1.6641, 1.5803, 1.6225, 1.6742, 1.5410]
# model_32_mae_mean = np.mean(model_32_mae)
#
# Persistence_32_mae = [4.0366, 4.0444, 4.0176, 4.0618, 4.0488]
# Persistence_32_mae_mean = np.mean(Persistence_32_mae)
#
# linear_32_mae = [3.9455, 3.9506, 3.9156, 3.9615, 3.9599]
# linear_32_mae_mean = np.mean(linear_32_mae)

# folds 10, 11, 12, 13
model_32_mae = [1.7669, 1.7812, 1.5917, 1.5809]
model_32_mae_mean = np.mean(model_32_mae)

Persistence_32_mae = [4.0292, 4.0355, 4.0172, 4.0274]
Persistence_32_mae_mean = np.mean(Persistence_32_mae)

linear_32_mae = [3.9428, 3.9456, 3.9339, 3.9339]
linear_32_mae_mean = np.mean(linear_32_mae)

# 5, 6, 7, 9 (Comment implies 4, list has 5 elements)
# model_33_mae = [2.2588, 2.2460, 2.2338, 2.7287, 2.2503]
# model_33_mae_mean = np.mean(model_33_mae)
#
# Persistence_33_mae = [5.6120, 5.6291, 5.6806, 5.6417, 5.6253]
# Persistence_33_mae_mean = np.mean(Persistence_33_mae)
#
# linear_33_mae = [6.1152, 6.1899, 6.1925, 6.1583, 6.1381]
# linear_33_mae_mean = np.mean(linear_33_mae)

# folds 11, 13, 14, 15

model_33_mae = [2.5954, 2.6040, 2.6204, 2.5793]
model_33_mae_mean = np.mean(model_33_mae)

Persistence_33_mae = [5.6442, 5.6434, 5.6366, 5.6283]
Persistence_33_mae_mean = np.mean(Persistence_33_mae)

linear_33_mae = [6.1736, 6.1772, 6.1563, 6.1569]
linear_33_mae_mean = np.mean(linear_33_mae)

# 4, 5, 6, 7, 8 (5 elements)
# model_34_mae = [2.9459, 3.1058, 2.9982, 3.2073, 3.1336]
# model_34_mae_mean = np.mean(model_34_mae)
#
# Persistence_34_mae = [7.0220, 7.0040, 7.0282, 7.0684, 7.0696]
# Persistence_34_mae_mean = np.mean(Persistence_34_mae)
#
# linear_34_mae = [8.4177, 8.3819, 8.4290, 8.4569, 8.4376]
# linear_34_mae_mean = np.mean(linear_34_mae)


# folds 9 10 11 12

model_34_mae = [3.4652, 3.4742, 3.5271, 3.4194]
model_34_mae_mean = np.mean(model_34_mae)

Persistence_34_mae = [7.0282, 7.0323, 7.0122,  7.0177]
Persistence_34_mae_mean = np.mean(Persistence_34_mae)

linear_34_mae = [8.4172, 8.4201, 8.3871, 8.7527]
linear_34_mae_mean = np.mean(linear_34_mae)

# 11, 12, 13, 16, 17 (5 elements)
model_6d_mae = [0.8733, 0.8746, 0.8647, 0.9819, 0.9365]
model_6d_mae_mean = np.mean(model_6d_mae)

Persistence_6d_mae = [2.3017, 2.3090, 2.2947, 2.3125, 2.2915]
Persistence_6d_mae_mean = np.mean(Persistence_6d_mae)

linear_6d_mae = [2.1347, 2.1422, 2.1265, 2.1336, 2.1230]
linear_6d_mae_mean = np.mean(linear_6d_mae)

# (Assuming 5 elements based on context, no comment provided)
model_12d_mae = [1.5251, 1.4736, 1.4629, 1.4119, 1.4959]
model_12d_mae_mean = np.mean(model_12d_mae)

Persistence_12d_mae = [3.9912, 3.9824, 4.0127, 4.0236, 3.9763]
Persistence_12d_mae_mean = np.mean(Persistence_12d_mae)

linear_12d_mae = [3.9597, 3.9470, 3.9549, 3.9710, 3.9384]
linear_12d_mae_mean = np.mean(linear_12d_mae)
# -----------------------------------------------------------------------------------------------------------------------
# 3, 5, 7, 8, 10 (Comment implies 5, list has 6 elements)
model_18d_mae = [2.2063, 2.2617, 2.2010, 2.0322, 2.0853]
model_18d_mae_mean = np.mean(model_18d_mae)

Persistence_18d_mae = [5.6553, 5.6568, 5.6712, 5.6482, 5.6588] # Note: one value (3.1217) is an outlier compared to others
Persistence_18d_mae_mean = np.mean(Persistence_18d_mae)

linear_18d_mae = [6.1628, 6.1468, 6.1891, 6.1373, 6.1845] # Note: one value (3.9316) is an outlier
linear_18d_mae_mean = np.mean(linear_18d_mae)

# 3, 5, 7, 8, 10 (5 elements)
model_24d_mae = [2.6715, 2.7215, 2.6761, 2.7179, 2.7642]
model_24d_mae_mean = np.mean(model_24d_mae)

Persistence_24d_mae = [7.1583, 7.0725, 7.1761, 7.1351, 7.0932]
Persistence_24d_mae_mean = np.mean(Persistence_24d_mae)

linear_24d_mae = [8.3161, 8.3013, 8.4236, 8.3465, 8.2829]
linear_24d_mae_mean = np.mean(linear_24d_mae)

# === Grouping and X-axis ===

# Mean MAE values
model_no_leakage_means = [model_31_mae_mean, model_32_mae_mean, model_33_mae_mean, model_34_mae_mean]
model_with_leakage_means = [model_6d_mae_mean, model_12d_mae_mean, model_18d_mae_mean, model_24d_mae_mean]

persistence_no_leakage_means = [Persistence_31_mae_mean, Persistence_32_mae_mean, Persistence_33_mae_mean, Persistence_34_mae_mean]
linear_no_leakage_means = [linear_31_mae_mean, linear_32_mae_mean, linear_33_mae_mean, linear_34_mae_mean]

persistence_with_leakage_means = [Persistence_6d_mae_mean, Persistence_12d_mae_mean, Persistence_18d_mae_mean, Persistence_24d_mae_mean]
linear_with_leakage_means = [linear_6d_mae_mean, linear_12d_mae_mean, linear_18d_mae_mean, linear_24d_mae_mean]

time_step = np.array([6, 12, 18, 24]) # Assuming these correspond to 31/6d, 32/12d, 33/18d, 34/24d pairs

# Standard deviations (error bars)
model_no_leakage_err = [np.std(model_31_mae), np.std(model_32_mae), np.std(model_33_mae), np.std(model_34_mae)]
model_with_leakage_err = [np.std(model_6d_mae), np.std(model_12d_mae), np.std(model_18d_mae), np.std(model_24d_mae)]

persistence_no_leakage_err = [np.std(Persistence_31_mae), np.std(Persistence_32_mae), np.std(Persistence_33_mae), np.std(Persistence_34_mae)]
linear_no_leakage_err = [np.std(linear_31_mae), np.std(linear_32_mae), np.std(linear_33_mae), np.std(linear_34_mae)]

persistence_with_leakage_err = [np.std(Persistence_6d_mae), np.std(Persistence_12d_mae), np.std(Persistence_18d_mae), np.std(Persistence_24d_mae)]
linear_with_leakage_err = [np.std(linear_6d_mae), np.std(linear_12d_mae), np.std(linear_18d_mae), np.std(linear_24d_mae)]

# forecast
forecast_mae_data = [0.71, 0.93, 1.05]
forecast_time_steps = np.array([6, 12, 18])



leakage_improvement_pct = 100 * (np.array(model_no_leakage_means) - np.array(model_with_leakage_means)) / np.array(model_no_leakage_means)

# Print improvements
print("--- Leakage Improvement ---")
for t, imp in zip(time_step, leakage_improvement_pct):
    print(f"Improvement at {t}h (Model No Leakage vs Model With Leakage): {imp:.2f}%")
print("---------------------------")

# === Plot ===

fig, ax = plt.subplots(figsize=(10, 7)) # Increased figure size for better legend readability

# Plotting model lines
ax.errorbar(time_step, model_no_leakage_means, yerr=model_no_leakage_err, label='Model', fmt='-o', capsize=5, color='blue', linewidth=2)
# ax.errorbar(time_step, model_with_leakage_means, yerr=model_with_leakage_err, label='Model (With Leakage)', fmt='-s', capsize=5, color='orange', linewidth=2)

# Plotting baseline lines for "No Leakage" scenario
ax.errorbar(time_step, persistence_no_leakage_means, yerr=persistence_no_leakage_err, label='Persistence', fmt='--^', capsize=3, color='green', alpha=0.7)
ax.errorbar(time_step, linear_no_leakage_means, yerr=linear_no_leakage_err, label='Linear', fmt=':x', capsize=3, color='red', alpha=0.7)

# Plotting baseline lines for "With Leakage" scenario
# ax.errorbar(time_step, persistence_with_leakage_means, yerr=persistence_with_leakage_err, label='Persistence (Baseline for With Leakage)', fmt='--v', capsize=3, color='purple', alpha=0.7)
# ax.errorbar(time_step, linear_with_leakage_means, yerr=linear_with_leakage_err, label='Linear (Baseline for With Leakage)', fmt=':P', capsize=3, color='brown', alpha=0.7)

ax.plot(forecast_time_steps, forecast_mae_data, label='ERA5 Forecast', marker='d', linestyle='-', color='magenta', linewidth=2, zorder=10)

# Annotate leakage improvement on plot (relative to the "Model With Leakage" line)
# for i, x_pos in enumerate(time_step):
#     y_pos = model_with_leakage_means[i]
#     imp = leakage_improvement_pct[i]
#     # Adjust text position to avoid overlap with data points or other text
#     vertical_offset = 0.2 if imp > 0 else -0.3 # Nudge text up for positive, down for negative
#     if y_pos + vertical_offset > ax.get_ylim()[1] * 0.95 : # If too high, nudge down
#          vertical_offset = -0.4
#     if y_pos + vertical_offset < ax.get_ylim()[0] * 1.05 : # If too low, nudge up
#          vertical_offset = 0.3
#
#     ax.text(x_pos, y_pos + 0.5, f"{imp:.1f}%", color='orange', fontsize=9, ha='center', weight='bold')
# Annotate MAE values on each point

# For Model
for i, (x, y) in enumerate(zip(time_step, model_no_leakage_means)):
    ax.text(x, y + 0.15, f"{y:.2f}", color='blue', fontsize=12, ha='center')

# For Persistence
for i, (x, y) in enumerate(zip(time_step, persistence_no_leakage_means)):
        ax.text(x, y + 0.15, f"{y:.2f}", color='green', fontsize=12, ha='center')

# For Linear
for i, (x, y) in enumerate(zip(time_step, linear_no_leakage_means)):
    if x == 6:
        ax.text(x, y - 0.3, f"{y:.2f}", color='red', fontsize=12, ha='center')
    if x == 12:
        ax.text(x, y - 0.3, f"{y:.2f}", color='red', fontsize=12, ha='center')
    else:
        ax.text(x, y + 0.15, f"{y:.2f}", color='red', fontsize=12, ha='center')

# For ERA5 Forecast
for i, (x, y) in enumerate(zip(forecast_time_steps, forecast_mae_data)):
    ax.text(x, y + 0.15, f"{y:.2f}", color='magenta', fontsize=12, ha='center')

ax.set_xlabel('Forecast Horizon (Hours)', fontsize=20)
ax.set_ylabel('Mean Absolute Error', fontsize=20)
ax.set_title('Prediction MAE Across Forecast Horizons', fontsize=28)
fig.patch.set_facecolor('#81CFF3')
# background_color = '#81CFF3'

ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', fontsize=18) # Adjusted legend location and size
ax.set_xticks(time_step)
ax.set_xticklabels([str(t) for t in time_step])
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.show()