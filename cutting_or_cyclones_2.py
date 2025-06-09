import xarray as xr
import pandas as pd
from Arrange_data_corrected import read_trajectories, get_nc
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import timedelta
import glob
import re


def correct_longitudes(longitudes):
    # Adjust longitudes to be within the range of 0 to 360 degrees
    longitudes = np.where(longitudes >= 360, longitudes - 360, longitudes)
    longitudes = np.where(longitudes < 0, longitudes + 360, longitudes)
    return longitudes


def cut_polygon(reanalysis_data, time, x, y, space_lon, space_lat, resolution, var, level):
    longitudes = np.arange(x - space_lon, x + space_lon, resolution)
    latitudes = np.arange(y - space_lat, y + space_lat, resolution)


    if np.any(longitudes > 360) or np.any(longitudes < 0):
        # Handle wrap-around case
        corrected_longitudes = correct_longitudes(longitudes)

        lon1 = corrected_longitudes[corrected_longitudes >= 0]
        lon2 = corrected_longitudes[corrected_longitudes <= 360]

        ds1 = reanalysis_data.sel(time=time, longitude=lon1, latitude=latitudes,level=level, method='nearest')
        ds2 = reanalysis_data.sel(time=time, longitude=lon2, latitude=latitudes,level=level, method='nearest')

        combined_ds = xr.concat([ds1, ds2], dim='longitude').isel(longitude=slice(None, len(longitudes)))
        # combined_ds = combined_ds.sortby('longitude')

    else:
        combined_ds = reanalysis_data.sel(time=time, longitude=longitudes, latitude=latitudes,level=level, method='nearest')
        # combined_ds = combined_ds.sortby('longitude')

    return combined_ds[var].to_numpy()


def pre_tracks(file_path):
    def filter_tracks(ds):
        nonz = (ds.t.data != 0).sum(axis=0)  # find the lifetime
        E = ds.intensity.data
        b = np.argmax(E, axis=0)  # find the time of maximum intensity
        lati = np.abs(ds.lat.data[0])  # find the latitude at genesis
        loni = ds.lon.data[0]  # find the longitude at genesis
        lonm = ds.lon.data[b, np.arange(len(b))]  # find the longitude at maximum intensity
        dlon = np.mod(lonm - loni, 360)  # find the distance between genesis and maximum intensity
        dldt = dlon / (b + 1)  # find the speed of the storm
        ind = (lati < 60) & (lati > 20) & (nonz > 16) & (dldt > 0.3) & (dlon < 200)
        return ds.isel(trackid=ind)

    ds = xr.open_dataset(file_path).load().data
    names = ['t', 'lon', 'lat', 'intensity']
    datasets = []
    for i in range(4):
        datasets.append(ds[i, :, :].to_dataset(name=names[i]).drop('variables'))  # cut the raw data into 4 datasets
    ds = xr.merge(datasets)
    return filter_tracks(ds)


if __name__ == '__main__':

    data_path = "/data/shreibshtein/OrCyclones"
    start_year = 1958
    end_year = 2021

    """ space and resolution needs to be changed manually """
    space_lon = 17.5
    space_lat = 15
    resolution = 1.25
    levels = [250, 300, 500, 850]
    variables = ['v', 'u', 't', 'z']
    reanalysis_directories = ["/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/va/va_6hrPlev_reanalysis_ERA5_",
                              "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ua/ua_6hrPlev_reanalysis_ERA5_",
                              "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/ta/ta_6hrPlev_reanalysis_ERA5_",
                              "/data/iacdc/ECMWF/ERA5/4xday_1.25_global_1000-200hPa/z/z_6hrPlev_reanalysis_ERA5_"]

    # Create a regex pattern to extract the year from the file name
    year_pattern = re.compile(r'_(\d{4})_')

    # Use glob to get the list of all files in the folder
    all_files = glob.glob(f"{data_path}/*.nc")

    # Filter files based on the year range
    file_paths = []

    for file_path in all_files:
        match = year_pattern.search(file_path)
        if match:
            year = int(match.group(1))
            if start_year <= year <= end_year:
                file_paths.append(file_path)

    for file in file_paths:
        print(file)
        file_full_path = file.split('/')
        file_name = file_full_path[-1].split('.')

        season = str(file)[30:33]
        if season == 'MAM':
            month = 3
        if season == 'DJF':
            month = 12
        if season == 'JJA':
            month = 6
        if season == 'SON':
            month = 9
        year = str(file)[34:38]
        starting_season_date = datetime.datetime(year=int(year), month=month, day=1, hour=0)
        delta = timedelta(hours=3)  # Multiply by the even time index for 6h time resolution

        data = pre_tracks(file)
        # print(data)

        num_tracks = data["trackid"]
        # even_num = num_tracks[num_tracks %2==0] # Only even indexes means 6h resolution
        # print(even_num)
        # exit()
        for v in range(len(variables)):
            var = variables[v]
            print(var)
            reanalysis_dir = reanalysis_directories[v]
            reanalysis_data, lons, lats = get_nc(int(year), month, reanalysis_dir)

            for l in levels:
                print(l)

                for num in num_tracks:

                    time_indexes = data.sel(trackid=num)['t']
                    even_time_indexes = time_indexes[
                        time_indexes % 2 == 0].to_numpy()  # Only even indexes means 6h resolution
                    even_time_indexes = np.delete(even_time_indexes, np.where(even_time_indexes == 0.))
                    tensor_p_storm = []
                    for i in even_time_indexes:
                        i = int(i)
                        x = data.sel(trackid=num)['lon'].where(data.sel(trackid=num)['t'] == i).dropna(dim='points').data[0]
                        y = \
                        data.sel(trackid=num)['lat'].where(data.sel(trackid=num)['t'] == i).dropna(dim='points').data[0]


                        longitudes = np.arange(x - space_lon, x + space_lon, resolution)
                        latitudes = np.arange(y - space_lat, y + space_lat, resolution)

                        time = starting_season_date + i * delta
                        tensor_p_storm.append(
                        cut_polygon(reanalysis_data, time, x, y, space_lon, space_lat, resolution, var, l))

                    tensor_p_storm_p_l = np.array(tensor_p_storm )
                    np.save(
                        "/data/iacdc/ECMWF/ERA5/Gilad/OrTensors/" + var + "a/" + str(l) + "/" + year + "/" + file_name[0]+'_'+ str(i),
                        tensor_p_storm_p_l)
