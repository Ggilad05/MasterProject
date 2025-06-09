import pandas as pd
import glob
import xarray as xr
from netCDF4 import Dataset
import time
import numpy as np


def arrange_data():
    folder_path = 'C:/Users/shrei/PycharmProjects/MasterProject/tracks_ERA5_1979-2020_0.25deg_1hr'
    file_list = glob.glob(folder_path + "/*.txt")
    # folder_path = 'C:/Users/shrei/PycharmProjects/MasterProject/1979'
    file_list = glob.glob(folder_path + "/*.txt")
    main_data = pd.DataFrame()
    for i in range(0, len(file_list)):
        data = pd.read_table(file_list[i], sep=" ", names=["Index", "Longitude", "Latitude", "Year", "Month", "Day",
                                                           "Hour", "Lowest MSLP value"])
        main_data = pd.concat([main_data, data])


    return main_data


def get_nc(year, month):
    number_of_days = pd.Timestamp(year=year, month=int(month), day=1).daysinmonth

    fn_tp = "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/pr/pr_yr_reanalysis_ERA5_" + str(
        year) + month + "01_" + str(year) + month + str(number_of_days) + ".nc"
    # fn_slp = "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/psl/psl_1hrPlev_reanalysis_ERA5_" + str(
    #     year) + month + "01_" + str(year) + month + str(number_of_days) + ".nc"
    # fn_ta850 = "/data/iacdc/ECMWF/ERA5/hourly_0.25_global_1000-200hPa/ta850/ta850_1hrPlev_reanalysis_ERA5_" + str(year) + month + "01_" + str(
    #     year) + month + str(number_of_days) + ".nc"

    tp = xr.open_dataset(fn_tp)
    # slp = xr.open_dataset(fn_slp)
    # ta850 = xr.open_dataset(fn_ta850)

    # Load lat and lon
    lats = tp.variables['latitude'][:]
    lons = tp.variables['longitude'][:]

    # return tp, slp, ta850, lons, lats
    return tp, lons, lats
