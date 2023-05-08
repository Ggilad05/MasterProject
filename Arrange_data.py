import pandas as pd
import glob
import xarray as xr
from netCDF4 import Dataset
import time



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

    IMS_data = pd.read_csv('C:/Users/shrei/PycharmProjects/MasterProject/IMS_GILAD.csv')


    # IMS_new_data = pd.DataFrame(columns= ["Year"," Month", "Day","Mean MSLP"
    #     , "Accumulate precipitation", "# IMS > 0","% StationRain/ TotalStations"])

    # print(main_data.loc[(main_data["Index"] == 1) & (main_data["Year"] == 2000)].iloc[0][2])

    return main_data, IMS_data


def get_nc(year, month):
    number_of_days = pd.Timestamp(year=year, month=int(month), day=1).daysinmonth
    # sftp_client = client.open_sftp()
    # # ncfile = sftp_client.open("/data/iacdc/ECMWF/ERA5/pr_1hrPlev_reanalysis_ERA5_19790101_19790131.nc")
    # ncfile = sftp_client.open("/data/iacdc/ECMWF/ERA5/pr_1hrPlev_reanalysis_ERA5_"
    #                           +str(year)+month+"01_"+str(year)+month+str(number_of_days)+".nc")
    #
    # #
    # ncfile.prefetch()
    # b_ncfile = ncfile.read()
    #
    # # ****
    # nc = Dataset("nc", memory=b_ncfile, format="NETCDF4")
    # ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc), chunks='auto')

    # fn = "C:/Users/shrei/PycharmProjects/MasterProject/1979_2020/" + str(year) + ".nc"T
    fn = "D:/ERA5/pr_1hrPlev_reanalysis_ERA5_"+str(year)+month+"01_"+str(year)+month+str(number_of_days)+".nc"
    ds = xr.open_dataset(fn)

    # Load lat and lon
    lats = ds.variables['latitude'][:]
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    lons = ds.variables['longitude'][:]

    return ds, lons, lats
