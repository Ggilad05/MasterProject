import xarray as xr
import pandas as pd
from Arrange_data_corrected import read_trajectories, get_nc
import matplotlib.pyplot as plt
import numpy as np
import datetime


def cutting_around_the_center(cyclones_data, reanalysis_dir, year):
    """ space and resolution needs to be changed manually """
    space = 10
    resolution = 1.25

    for storm in cyclones_data:
        index = str(int(storm.iloc[0]["Index"]))
        tensor_per_storm = []
        for i in range(len(storm)):


            reanalysis_data, lons, lats = get_nc(int(storm.iloc[i]["Year"]), storm.iloc[i]["Month"], reanalysis_dir)
            print(reanalysis_data)
            variable_info = str(reanalysis_data.data_vars)
            variable_name = variable_info[variable_info.find(':')+6: variable_info.find('(') - 6]
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
            "/data/iacdc/ECMWF/ERA5/Tensors/"+variable_name+year + "/" + index, tensor_per_storm)


def filter_storms_data_by_month(storms_data, month):
    storms_data_filterd_by_month = []
    for storm in storms_data["Grouped data"]:
        if storm.iloc[0]["Month"] == month:
            storms_data_filterd_by_month.append(storm)

    return storms_data_filterd_by_month


if __name__ == '__main__':
    years = np.arange(1979, 2021, 1)

    """I can control the months that interest me """
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    # reanalysis_dir = "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/pr/pr_yr_reanalysis_ERA5_"
    reanalysis_dir = "/data/iacdc/ECMWF/ERA5/tpw/tpw_6hr_reanalysis_ERA5_"
    # reanalysis_dir_list = ["ua/ua_6hrPlev_reanalysis_ERA5_", "va/va_6hrPlev_reanalysis_ERA5_",
    #                        "uas/uas_6hrPlev_reanalysis_ERA5_", "vas/vas_6hrPlev_reanalysis_ERA5_",
    #                        "z/z_6hrPlev_reanalysis_ERA5_", "ta/ta_6hrPlev_reanalysis_ERA5_", "ps/ps_6hrPlev_reanalysis_ERA5_"]
    # reanalysis_dir_list = ["uas/uas_6hrPlev_reanalysis_ERA5_", "vas/vas_6hrPlev_reanalysis_ERA5_",
    #                        "ps/ps_6hrPlev_reanalysis_ERA5_"]


    for y in years:
        print(y)
        life_cycles_per_year = []
        storms_data = read_trajectories(y)
        for month in months:
            print(month)
            int_month = int(month)
            storms_data_filterd_by_month = filter_storms_data_by_month(storms_data, int_month)
            """ The field inside -->[] need to be change for every field """
            cutting_around_the_center(storms_data_filterd_by_month, reanalysis_dir, str(int(y)))
