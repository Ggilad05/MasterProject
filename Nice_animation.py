import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

# Replace the sample data with your actual 10 by 10 matrices
wind_velocity_data = [np.load('C:/Users/shrei/PycharmProjects/MasterProject/OrCyclones/Tensors/ua/250/20.npy'),
                      np.load('C:/Users/shrei/PycharmProjects/MasterProject/OrCyclones/Tensors/ua/300/20.npy'),
                      np.load('C:/Users/shrei/PycharmProjects/MasterProject/OrCyclones/Tensors/ua/500/20.npy'),
                      np.load('C:/Users/shrei/PycharmProjects/MasterProject/OrCyclones/Tensors/ua/850/20.npy')]

v_min_w = np.min(wind_velocity_data)
v_max_w = np.max(wind_velocity_data)

temperature_data = [np.load('C:/Users/shrei/PycharmProjects/MasterProject/OrCyclones/Tensors/ta/250/20.npy'),
                    np.load('C:/Users/shrei/PycharmProjects/MasterProject/OrCyclones/Tensors/ta/300/20.npy'),
                    np.load('C:/Users/shrei/PycharmProjects/MasterProject/OrCyclones/Tensors/ta/500/20.npy'),
                    np.load('C:/Users/shrei/PycharmProjects/MasterProject/OrCyclones/Tensors/ta/850/20.npy')]

v_min_t = np.min(temperature_data)
v_max_t = np.max(temperature_data)

# Create subplots with 2 rows and 4 columns
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# Plot wind velocity data (initial frame)
names = [250, 300, 500, 850]
norm_wind = Normalize(vmin=v_min_w, vmax=v_max_w)
images_wind = [axes[0, i].imshow(np.flipud(wind_velocity_data[i][0, :, :]), cmap='viridis', norm=norm_wind) for i in range(4)]
for i in range(4):
    axes[0, i].set_title(f'Wind Velocity {names[i]}')

# Plot temperature data (initial frame)
norm_temp = Normalize(vmin=v_min_t, vmax=v_max_t)
images_temp = [axes[1, i].imshow(np.flipud(temperature_data[i][0, :, :]), cmap='plasma', norm=norm_temp) for i in range(4)]
for i in range(4):
    axes[1, i].set_title(f'Temperature {names[i]}')

# Add a single colorbar for wind velocity
cax_wind = fig.add_axes([0.1, 0.07, 0.35, 0.02])  # [x, y, width, height]
cbar_wind = fig.colorbar(images_wind[0], cax=cax_wind, orientation='horizontal', label='Wind Velocity')

# Add a single colorbar for temperature
cax_temp = fig.add_axes([0.55, 0.07, 0.35, 0.02])  # [x, y, width, height]
cbar_temp = fig.colorbar(images_temp[0], cax=cax_temp, orientation='horizontal', label='Temperature')

# Add title for the 'time'
title = fig.suptitle('Time: 0', y=0.98)


# Function to update the frames
def update(frame):
    for i in range(4):
        images_wind[i].set_array(np.flipud(wind_velocity_data[i][frame, :, :]))
        images_temp[i].set_array(np.flipud(temperature_data[i][frame, :, :]))
    title.set_text(f'Time: {frame}')  # Update the title with the current 'time'
    return images_wind + images_temp + [title]


# Create an animation
animation = FuncAnimation(fig, update, frames=range(1, 10), interval=200, repeat=True)

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0.09, 1, 0.95])  # Adjust the rect parameter to raise the temperature figures

# Show the animation
plt.show()
