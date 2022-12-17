import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import pandas as pd
import glob
import numpy as np
from numpy import deg2rad, cos, gradient, meshgrid
import xarray as xr
import os
import sys
import netCDF4 as nc

DATA_1 = '1980_141.nc'
DATA_2 = 'download.nc'
FILEPATH = 'C:\\Users\\shrei\\OneDrive\\Documents\\geo_assets\\{}'


def arrange_data():
    folder_path = 'C:/Users/shrei/PycharmProjects/MasterProject/tracks_ERA5_1979-2020_0.25deg_1hr'
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


def get_years_list(years, year, num):
    years.append(year)
    for i in range(num):
        year += 1
        years.append(year)
    return years


def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84

    Input
    ---------
    lat: vector or latitudes in degrees

    Output
    ----------
    r: vector of radius in meters

    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b ** 2 / a ** 2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan((1 - e2) * np.tan(lat))

    # radius equation
    # see equation 3-107 in WGS84
    r = (
            (a * (1 - e2) ** 0.5)
            / (1 - (e2 * np.cos(lat_gc) ** 2)) ** 0.5
    )

    return r


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters

    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees

    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]

    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = np.abs(dy * dx)

    xda = DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda


def calculate_total_precipitation_weighted(data):
    # area dataArray
    da_area = area_grid(data['latitude'], data['longitude'])
    # total_area = np.abs(da_area.sum(['latitude','longitude']))
    total_area = 5.1006948e+14
    # tp weighted by grid-cell area
    temp_weighted = (data["tp"] * da_area) / total_area
    return (data["tp"] * da_area) / total_area


def plot_data_on_map(data, m_data):
    # data - Trajectories data

    x = data.loc[(data["Index"] == 1) & (data["Year"] == 2000)]['Longitude']
    y = data.loc[(data["Index"] == 1) & (data["Year"] == 2000)]['Latitude']

    # years = get_years_list([], 1979, 41)
    years = get_years_list([], 1980, 0)

    # Take just Trajectories near Israel With polygon
    # filtered_data = data[(data["Longitude"] > 28) & (data["Longitude"] < 36) & (data["Latitude"] > 31) &
    #                      (data["Latitude"] < 36)]

    filtered_data = data[(data["Longitude"] > 20) & (data["Longitude"] < 40) & (data["Latitude"] > 28) &
                         (data["Latitude"] < 40)]

    # fn = "C:/GOG Games/download.nc"
    #fn = "C:/GOG Games/1980_141.nc"
    # fn = f"C:/GOG Games/{DATA_2}"
    fn = FILEPATH.format('1980_141.nc')
    # fn = "C:/Users/shrei/PycharmProjects/MasterProject/1980_141_all.nc"
    ds = xr.open_dataset(fn)

    # Load lat and lon
    lats = ds.variables['latitude'][:]
    lons = ds.variables['longitude'][:]

    """  Units of -->tp[m]: The depth of water on a p
     To get the total precipitation for an hour (mm) :  tp [mm]=tp [m]⋅1000"""
    # total_precipitation = ds["tp"]
    # total_precipitation.data = total_precipitation.data * 1000
    # total_precipitation.attrs['units'] = 'mm'
    # print(total_precipitation[0,0,1])
    # total_precipitation.isel(time=0).plot()
    # plt.show()

    tp_weighted = calculate_total_precipitation_weighted(ds)
    # print(tp_weighted)
    # tp_weighted.isel(time=0).plot()
    # plt.show()
    print("NO")
    print(tp_weighted)
    print("yes")
    print(tp_weighted.integrate(coord=["time"]))

    lon, lat = np.meshgrid(lons, lats)
    lama = 141 if DATA_1 in fn else 85

    for year in years:
        map = Basemap(projection='cass', lat_0=31.5, lon_0=32.851612, width=505000*2, height=505000*2,
                      resolution='l')

        x, y = map(lon, lat)
        list_j = []
        count = 0
        for j in filtered_data.loc[filtered_data["Year"] == year]["Index"]:
            if j in list_j:
                continue
            # print(j)
            if j == lama:
                sum_time = tp_weighted
                for i in range(0, len(tp_weighted)):
                    # norm = mpl.colors.Normalize(vmin=0, vmax=3)
                    map.pcolor(x, y, np.squeeze(tp_weighted[i, :, :]), cmap='jet')

                    #
                    lon_f = filtered_data.loc[(filtered_data["Index"] == j) & (filtered_data["Year"] == year)][
                        'Longitude']
                    lat_f = filtered_data.loc[(filtered_data["Index"] == j) & (filtered_data["Year"] == year)][
                        'Latitude']
                    map.plot(np.array(lon_f), np.array(lat_f), latlon=True, linewidth=5)
                    map.drawcoastlines()
                    # labels = [left,right,top,bottom]
                    map.drawmeridians(np.arange(28, 40, 1), labels=[True,False,False,True])
                    map.drawparallels(np.arange(20, 40, 1), labels=[True,False,False,False])
                    plt.colorbar(label='tp')





                    # lat_index = list(lats).index(lat_f.iloc[i])
                    # lon_index = list(lons).index(lon_f.iloc[i])
                    list_j.append(j)

                    # sum_int += tp_weighted[i, lat_index - 5:lat_index + 5,
                    #         lon_index - 5:lon_index + 5].sum(dim=['longitude', 'latitude'])
                    # print(sum_int)

                    # tp_weighted[:, lat_index - 5:lat_index + 5,
                    # lon_index - 5:lon_index + 5].isel(time=0).plot()

                    # print(sum_int)
                    count += 1
                    plt.savefig(
                        'C:/Users/shrei/PycharmProjects/MasterProject/Plots/1980_141/' + str(
                            count) + '.png')
                    print(" ")
                    plt.close()

                # list_sum.append(sum_int)

        map.plot([28, 28, 36, 36, 28], [36, 31, 31, 36, 36], latlon=True, c="r")
        map.drawcoastlines()


def project():
    cyclone_data, ims_data = arrange_data()
    plot_data_on_map(cyclone_data, ims_data)


def main():
    project()


if __name__ == '__main__':
    main()
