import multiprocessing
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import glob
import re
from Arrange_data_corrected import get_nc


def correct_longitudes(longitudes):
    """Adjust longitudes to be within [0, 360) range."""
    longitudes = np.where(longitudes >= 360, longitudes - 360, longitudes)
    longitudes = np.where(longitudes < 0, longitudes + 360, longitudes)
    return longitudes


def cut_polygon(reanalysis_data, time, x, y, space_lon, space_lat, resolution, var, level):
    """Cut a polygon around the storm's location from the reanalysis data for a specific level."""
    longitudes = np.arange(x - space_lon, x + space_lon, resolution)
    latitudes = np.arange(y - space_lat, y + space_lat, resolution)

    # Ensure dimensions match the expected shape by trimming if needed
    if len(latitudes) == 121:
        latitudes = latitudes[:-1]
    if len(longitudes) == 141:
        longitudes = longitudes[:-1]

    # Determine time and level dimensions in dataset
    time_dim = 'valid_time' if 'valid_time' in reanalysis_data.dims else 'time'
    level_dim = 'pressure_level' if 'pressure_level' in reanalysis_data.dims else 'level'

    # Handle longitude wrap-around cases
    if np.any(longitudes > 360) or np.any(longitudes < 0):
        lon1 = longitudes[(longitudes >= 0) & (longitudes < 360)]
        lon2 = longitudes[(longitudes < 0) | (longitudes >= 360)]
        corrected_longitudes = correct_longitudes(lon2)

        ds1 = reanalysis_data.sel(**{time_dim: time, "longitude": lon1, "latitude": latitudes, level_dim: level},
                                  method='nearest')
        ds2 = reanalysis_data.sel(**{time_dim: time, "longitude": corrected_longitudes, "latitude": latitudes, level_dim: level},
                                  method='nearest')

        if np.mean(lon1) > np.mean(corrected_longitudes):
            combined_ds = xr.concat([ds1, ds2], dim='longitude')
        elif np.mean(lon1) < np.mean(corrected_longitudes):
            combined_ds = xr.concat([ds2, ds1], dim='longitude')
        else:
            combined_ds = ds1 if np.shape(ds1[var]) != (120, 0) else ds2
    else:
        combined_ds = reanalysis_data.sel(
            **{time_dim: time, "longitude": longitudes, "latitude": latitudes, level_dim: level}, method='nearest')

    # Return the processed data for the specified variable at the given level
    return np.flipud(combined_ds[var])

def process_track_level(data, track_id, v, level, reanalysis_directory, space_lon, space_lat, resolution, starting_season_date, file_name, year):
    """Process each variable and level for a cyclone track and save results."""
    tensor_p_storm = []

    time_indexes = data.sel(trackid=track_id)['t'].to_numpy()
    time_indexes = np.delete(time_indexes, np.where(time_indexes == 0.))

    for i in time_indexes:
        x = data.sel(trackid=track_id)['lon'].where(data.sel(trackid=track_id)['t'] == i).dropna(dim='points').data[0]
        y = data.sel(trackid=track_id)['lat'].where(data.sel(trackid=track_id)['t'] == i).dropna(dim='points').data[0]
        time = starting_season_date + timedelta(hours=3) * (i - 1)

        reanalysis_data, lons, lats = get_nc(time.year, time.month, reanalysis_directory)

        # Process the data for the specific level
        result = cut_polygon(reanalysis_data, time, x, y, space_lon, space_lat, resolution, v, level)
        tensor_p_storm.append(result)

    # Save computed result for the specific variable and level
    output_path = f"/data/iacdc/ECMWF/ERA5/Gilad/0.25/{v}a/{level}/{year}/{file_name}_{int(track_id.values)}.npy"
    np.save(output_path, np.array(tensor_p_storm))
    print(f"Saved result for {v} at level {level} to {output_path}")

    return tensor_p_storm

def pre_tracks(file_path):
    """Load and filter cyclone tracks based on criteria."""

    def filter_tracks(ds):
        nonz = (ds.t.data != 0).sum(axis=0)  # Lifetime of the track
        E = ds.intensity.data
        b = np.argmax(E, axis=0)  # Time of maximum intensity
        lati = np.abs(ds.lat.data[0])  # Latitude at genesis
        loni = ds.lon.data[0]  # Longitude at genesis
        lonm = ds.lon.data[b, np.arange(len(b))]  # Longitude at max intensity
        dlon = np.mod(lonm - loni, 360)  # Distance between genesis and max intensity
        dldt = dlon / (b + 1)  # Speed of the storm
        ind = (lati < 60) & (lati > 20) & (nonz > 16) & (dldt > 0.3) & (dlon < 200)
        return ds.isel(trackid=ind)

    ds = xr.open_dataset(file_path).load().data
    names = ['t', 'lon', 'lat', 'intensity']
    datasets = [ds[i, :, :].to_dataset(name=names[i]).drop('variables') for i in range(4)]
    ds = xr.merge(datasets)
    return filter_tracks(ds)

# Main execution
if __name__ == '__main__':
    data_path = "/data/shreibshtein/OrCyclones"
    start_year, end_year = 1979, 2022
    space_lon, space_lat, resolution = 17.5, 15, 0.25
    variables = ['t', 'q', 'v', 'u', 'z']
    levels = [250, 300, 500, 850]
    reanalysis_directories = [
        "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/ta/ta_3hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/qa/qa_3hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/va/va_3hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/ua/ua_3hrPlev_reanalysis_ERA5_",
        "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/za/za_3hrPlev_reanalysis_ERA5_"
    ]

    year_pattern = re.compile(r'_(\d{4})_')
    all_files = glob.glob(f"{data_path}/*.nc")
    file_paths = [file for file in all_files if start_year <= int(year_pattern.search(file).group(1)) <= end_year]

    for file in file_paths:
        print(f"Processing file: {file}")
        file_name = file.split('/')[-1].split('.')[0]
        season, year = file_name.split('_')[0], file_name.split('_')[1]
        starting_season_date = datetime(int(year), {'MAM': 3, 'DJF': 12, 'JJA': 6, 'SON': 9}[season], 1)

        data = pre_tracks(file)

        for track_id in data["trackid"]:
            tasks = []
            for v, reanalysis_directory in zip(variables, reanalysis_directories):
                for level in levels:
                    tasks.append((data, track_id, v, level, reanalysis_directory, space_lon, space_lat, resolution, starting_season_date, file_name, year))

            with multiprocessing.Pool(processes=20) as pool:
                pool.starmap(process_track_level, tasks)

            print(f"Completed processing for track {track_id.values} in file {file_name}")

    print("All files processed.")
