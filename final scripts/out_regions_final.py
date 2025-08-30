import matplotlib.pyplot as plt
import numpy as np

# Data
locations = ['Persistence', 'N.ATL', 'N.PAC', 'S.ATL', 'MED', 'S.O.']
offsets_hours = [6, 12, 18, 24]
offset_labels = ['6', '12', '18', '24']

# Final MAE results array
mae_results = np.array([
    [2.62036342, 16.72729877, 13.76896835,  7.32754368],
    [1.2769177 ,  2.02672243,  2.94104123,  4.00964069],
    [1.21723783,  3.37314534,  2.75617981,  3.9340446 ],
    [1.19666672,  3.57624078,  2.73666716,  3.7612288 ],
    [1.11929035,  1.72925031,  2.23625135,  2.76900864],
    [1.64380574,  4.38015461,  5.69308901,          np.nan]
])

# Plot
fig = plt.figure(figsize=(12, 8))

for i, location in enumerate(locations):
    plt.plot(offsets_hours, mae_results[i], marker='o', label=location)

# Customize plot
plt.xlabel('Offset (hours)', fontsize=20)
plt.ylabel('Mean Absolute Error (MAE)', fontsize=20)
plt.title('Regional Transfer Learning skill on different offsets', fontsize=28)
plt.xticks(offsets_hours, offset_labels)

plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
fig.patch.set_facecolor('#81CFF3')  # Set background color

# Show plot
plt.show()

