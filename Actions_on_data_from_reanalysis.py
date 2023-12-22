import xarray as xr
import pandas as pd
from Arrange_data_corrected import read_trajectories, get_nc
import matplotlib.pyplot as plt
import numpy as np
import datetime


def cutting_around_the_center(cyclones_data, reanalysis_dir):
    """ space and resolution needs to be changed manually """
    space = 10
    resolution = 0.25
    index = []
    year = []
    tensors = []
    for storm in cyclones_data:
        tensor_per_storm = []
        index.append(storm.iloc[0]["Index"])
        year.append(storm.iloc[0]["Year"])
        for i in range(len(storm)):
            reanalysis_data = get_nc(int(storm.iloc[i]["Year"]), storm.iloc[i]["Month"], reanalysis_dir)['tp']

            time = datetime.datetime(year=int(storm.iloc[i]["Year"]), month=int(storm.iloc[i]["Month"]),
                                     day=int(storm.iloc[i]["Day"]), hour=int(storm.iloc[i]["Hour"]))

            ########
            # reanalysis = get_nc(1979, "02", reanalysis_dir)
            # reanalysis["tp"].sel(time=datetime.datetime(1979, 2, 28, 12, 0)).plot()
            # plt.savefig("/data/shreibshtein/plots/pic_test_2")
            # exit()
            x = storm.iloc[i]["Longitude"]
            y = storm.iloc[i]["Latitude"]

            longitudes = np.arange(x - space, x + space, resolution)
            latitudes = np.arange(y - space, y + space, resolution)
            # reanalysis_data.sel(time=time, longitude=longitudes, latitude=latitudes).plot()
            # plt.savefig("/data/shreibshtein/plots/pr_test_3")
            # exit()
            tensor_per_storm.append(
                reanalysis_data.sel(time=time, longitude=longitudes, latitude=latitudes).to_numpy())
        tensors.append(tensor_per_storm)

    return tensors, index, year


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
    reanalysis_dir = "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/pr/pr_yr_reanalysis_ERA5_"

    index = []
    year = []
    tensor = []
    for y in years:
        print(y)
        life_cycles_per_year = []
        storms_data = read_trajectories(y)
        for month in months:
            print(month)
            int_month = int(month)
            storms_data_filterd_by_month = filter_storms_data_by_month(storms_data, int_month)
            """ The field inside -->[] need to be change for every field """
            tensor_per_month, index_per_month, year_per_month = cutting_around_the_center(
                                                                                          storms_data_filterd_by_month,
                                                                                          reanalysis_dir)
            index.extend(index_per_month)
            year.extend(year_per_month)
            tensor.extend(tensor_per_month)
            print(" ")
            print(len(index))
            print(len(year))
            print(len(tensor))
            print(" ")
        data_cropped_through_life_cycle = pd.DataFrame({"Index": index, "Year": year, "Tensor": tensor}).to_csv('/data/shreibshtein/Tensors/tp_'+str(y)+'.csv')
        index = []
        year = []
        tensor = []



