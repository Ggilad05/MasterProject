# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np


# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
era5 = xr.open_dataset(r'C:\Users\shrei\PycharmProjects\MasterProject\models\Results_1.25\lookback3h-forecast1h\msl_3hr_reanalysis_ERA5_20100101_20100131.nc')
print(era5)

forecast = xr.open_dataset(r'C:\Users\shrei\PycharmProjects\MasterProject\models\Results_1.25\lookback3h-forecast1h\mslp_2010_01_01_converted.nc')
print(forecast)

era5_0 = era5['msl'].sel()
# %%
#######################################

# era5_example  = np.load(r"\\wsl.localhost\Ubuntu-20.04\home\gilad\era5\DJF_2010_NH_neg_1.npy")
# forecast_example  = np.load(r"\\wsl.localhost\Ubuntu-20.04\home\gilad\forecast\DJF_2010_NH_neg.nc_1.npy")
#
# print(era5_example)
# print(" ")
# print(forecast_example)
num = 6
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")

# Load data
era5_example = np.load(fr"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\DJF_2010_NH_neg_{num}.npy")
forecast_example = np.load(fr"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\forecast\\DJF_2010_NH_neg.nc_{num}.npy")

assert era5_example.shape == forecast_example.shape
n_frames, ny, nx = era5_example.shape
center_y, center_x = ny // 2, nx // 2

# Create figure
fig, ax = plt.subplots()

def update(frame):
    ax.clear()

    # Show ERA5 as a grayscale image
    ax.contour(era5_example[frame], origin='lower', cmap='Blues',
              vmin=np.min(era5_example), vmax=np.max(era5_example))

    # Overlay forecast contours as red lines
    ax.contour(forecast_example[frame], colors='red', linewidths=1.5)

    # Fixed center marker
    ax.plot(center_x, center_y, 'rx', markersize=10, markeredgewidth=2)

    ax.set_title(f'Frame {frame}')
    return ax

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=500)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")

# Load data
era5_example = np.load(fr"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\DJF_2010_NH_neg_{num}.npy")
forecast_example = np.load(fr"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\forecast\\DJF_2010_NH_neg.nc_{num}.npy")

assert era5_example.shape == forecast_example.shape
n_frames, ny, nx = era5_example.shape
center_y, center_x = ny // 2, nx // 2

# Precompute the difference
difference = forecast_example - era5_example

# Calculate symmetric color scale around 0
vmax = np.max(np.abs(difference))
vmin = -vmax

# Create figure
fig, ax = plt.subplots()
cbar = None

def update(frame):
    global cbar
    ax.clear()

    # Show difference using imshow
    im = ax.imshow(difference[frame], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)

    # Mark fixed center
    ax.plot(center_x, center_y, 'rx', markersize=10, markeredgewidth=2)

    ax.set_title(f'Difference: Forecast - ERA5 | Frame {frame}')

    # Add colorbar once
    if cbar is None:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label("Forecast - ERA5")

    return im,

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=500)
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")

# Load data
era5_example = np.load(fr"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\DJF_2010_NH_neg_{num}.npy")
forecast_example = np.load(fr"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\forecast\\DJF_2010_NH_neg.nc_{num}.npy")

assert era5_example.shape == forecast_example.shape
n_frames, ny, nx = era5_example.shape

# Precompute the difference
difference = forecast_example - era5_example

# Calculate symmetric color scale around 0
vmax = np.max(np.abs(difference))
vmin = -vmax

# Create figure
fig, ax = plt.subplots()
cbar = None

def update(frame):
    global cbar
    ax.clear()

    # Show difference using imshow
    im = ax.imshow(difference[frame], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)

    # Add contours
    ax.contour(era5_example[frame], colors='blue', linewidths=1.2)
    ax.contour(forecast_example[frame], colors='red', linewidths=1.2, linestyles='dashed')

    # Mark minimum point of ERA5 with a black X
    min_y, min_x = np.unravel_index(np.argmin(era5_example[frame]), era5_example[frame].shape)
    ax.plot(min_x, min_y, 'kx', markersize=10, markeredgewidth=2)

    ax.set_title(f'Difference: Forecast - ERA5 | Frame {frame}')

    # Add colorbar only once
    if cbar is None:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label("Forecast - ERA5")

    return im,

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=500)
plt.show()

# %%
num = 17
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")

# Load data
era5_example = np.load(r"\\wsl$\Ubuntu-20.04\\/home/gilad/era5/msl/2010/MAM_2010_NH_neg_238.npy ")
forecast_example = np.load(r"\\wsl$\\Ubuntu-20.04\\/home/gilad/forecast_0/msl/2010/MAM_2010_NH_neg.nc_238.npy ")


assert era5_example.shape == forecast_example.shape
n_frames, ny, nx = era5_example.shape

# Compute symmetric difference
difference = forecast_example - era5_example
vmax = np.max(np.abs(difference))
vmin = -vmax

# Central 8×8 region indices (based on your update)
start_y = (ny - 8) // 2
end_y = start_y + 8
start_x = (nx - 8) // 2
end_x = start_x + 8

# Create figure
fig, ax = plt.subplots()
cbar = None

def update(frame):
    global cbar
    ax.clear()

    # Slice central 8x8 window
    era5 = era5_example[frame, start_y:end_y, start_x:end_x]
    forecast = forecast_example[frame, start_y:end_y, start_x:end_x]
    diff = difference[frame, start_y:end_y, start_x:end_x]

    # Show difference with imshow
    im = ax.imshow(diff, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)

    # Contours for ERA5 and Forecast
    ax.contour(era5, colors='blue', linewidths=1.2)
    ax.contour(forecast, colors='red', linewidths=1.2, linestyles='dashed')

    # Dynamic ERA5 minimum marker (black 'X')
    min_y_era5, min_x_era5 = np.unravel_index(np.argmin(era5), era5.shape)
    ax.plot(min_x_era5, min_y_era5, 'kx', markersize=10, markeredgewidth=2, label='ERA5 Min')

    # Dynamic Forecast minimum marker (green 'X')
    min_y_fcst, min_x_fcst = np.unravel_index(np.argmin(forecast), forecast.shape)
    ax.plot(min_x_fcst, min_y_fcst, 'gx', markersize=10, markeredgewidth=2, label='Forecast Min')

    ax.set_title(f'Difference (Forecast - ERA5) | Frame {frame}')

    # Add colorbar once
    if cbar is None:
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label("Forecast - ERA5")

    return im,

ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=500)
plt.show()

# %%
num = 17
import numpy as np

# Load your data
era5_example = np.load(fr"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\era5\\2010\\DJF_2010_NH_neg_{num}.npy")
forecast_example = np.load(fr"\\wsl.localhost\\Ubuntu-20.04\\home\\gilad\\forecast\\DJF_2010_NH_neg_{num}.npy")

era5_examplen_frames, ny, nx = forecast_example.shape

# Define central 8×8 slice indices
start_y = (ny - 8) // 2
end_y = start_y + 8
start_x = (nx - 8) // 2
end_x = start_x + 8

# Collect absolute errors between ERA5 and Forecast minima
errors = []
for i in range(n_frames):
    era5_min = np.min(era5_example[i, start_y:end_y, start_x:end_x])
    forecast_min = np.min(forecast_example[i, start_y:end_y, start_x:end_x])
    errors.append(abs(era5_min - forecast_min))

# Compute MAE
mae = np.mean(errors)
print(f"MAE between minimum values in 8×8 region: {mae:.3f} Pa")

# %%


