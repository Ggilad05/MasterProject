import xarray as xr
import pandas as pd
from Arrange_data_corrected import read_trajectories, get_nc
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import timedelta
import glob
import re


def cutting_around_the_center(cyclones_data, reanalysis_dir, year):
    """ space and resolution needs to be changed manually """
    space = 25
    resolution = 1.25

    for storm in cyclones_data:
        index = str(int(storm.iloc[0]["Index"]))
        tensor_per_storm = []
        for i in range(len(storm)):
            reanalysis_data, lons, lats = get_nc(int(storm.iloc[i]["Year"]), storm.iloc[i]["Month"], reanalysis_dir)
            print(reanalysis_data)
            variable_info = str(reanalysis_data.data_vars)
            variable_name = variable_info[variable_info.find(':') + 6: variable_info.find('(') - 6]
            reanalysis_data = reanalysis_data[variable_name]

            time = datetime.datetime(year=int(storm.iloc[i]["Year"]), month=int(storm.iloc[i]["Month"]),
                                     day=int(storm.iloc[i]["Day"]), hour=int(storm.iloc[i]["Hour"]))

            x = storm.iloc[i]["Longitude"]
            y = storm.iloc[i]["Latitude"]

            longitudes = np.arange(x - space, x + space, resolution)
            latitudes = np.arange(y - space, y + space, resolution)
            # reanalysis_data.sel(time=time, longitude=longitudes, latitude=latitudes).plot()
            # plt.savefig("/data/shreibshtein/plots/pr_test_3")
            # exit()

            tensor_per_storm.append(
                reanalysis_data.sel(time=time, longitude=longitudes, latitude=latitudes).to_numpy())

        tensor_per_storm = np.array(tensor_per_storm)

        np.save(
            "/data/iacdc/ECMWF/ERA5/Tensors/" + variable_name + year + "/" + index, tensor_per_storm)


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
    start_year = 2000
    end_year = 2019

    """ space and resolution needs to be changed manually """
    space = 15
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
                    tensor_p_storm_p_l = []
                    for i in even_time_indexes:
                        i = int(i)
                        x = data.sel(trackid=num)['lon'].where(data.sel(trackid=num)['t'] == i).dropna(dim='points').data[0]
                        y = \
                        data.sel(trackid=num)['lat'].where(data.sel(trackid=num)['t'] == i).dropna(dim='points').data[0]


                        longitudes = np.arange(x - space, x + space, resolution)
                        latitudes = np.arange(y - space, y + space, resolution)

                        time = starting_season_date + i * delta

                        # reanalysis_data.sel(time=time, longitude=longitudes, latitude=latitudes, level=l,
                        #                     method="nearest").to_array().plot()
                        # plt.savefig("/data/shreibshtein/plots/pr_test_4")
                        # exit()

                        tensor_p_storm_p_l.append(
                            reanalysis_data.sel(time=time, longitude=longitudes, latitude=latitudes, level=l,
                                                method="nearest")[var].to_numpy())

                    tensor_p_storm_p_l = np.array(tensor_p_storm_p_l)
                    np.save(
                        "/data/iacdc/ECMWF/ERA5/OrTensors/" + var + "a/" + str(l) + "/" + year + "/" + str(i),
                        tensor_p_storm_p_l)
