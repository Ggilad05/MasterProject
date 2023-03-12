import cursor as cursor
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import xarray as xr
import numpy as np
from Met_St import st_dic







def get_years_list(years, year, num):
    years.append(year)
    for i in range(num):
        year += 1
        years.append(year)
    return years

def plot_traj():
    year = input("Enter year for trajectories plot: ")
    index = input("Enter index for plot (enter all or num): ")

    years = get_years_list([], 1980, 0)

def plot_tp(data, lons, lats):
    # map = Basemap(projection='cass', lat_0=31.5, lon_0=32.851612, width=505000, height=790000/2,
    #               resolution='l')
    map = Basemap(projection='cass', lat_0=32.6, lon_0=33.851612, width=505000, height=790000 / 2,
                  resolution='l')
    lon, lat = np.meshgrid(lons, lats)
    x, y = map(lon, lat)
    # isreal = data.rio.clip(gdf[0])
    map.pcolor(x, y, np.squeeze(data), cmap='jet')
    map.colorbar()
    for st in st_dic:
        st_coor = st_dic[st]
        map.scatter(st_coor[0], st_coor[1], s=5, latlon=True, label=st)
    map.drawcoastlines()
    map.drawcountries(zorder=1, color="black", linewidth=1)
    # labels = [left,right,top,bottom]
    map.drawmeridians(np.arange(28, 40, 1), labels=[True, False, False, True])
    map.drawparallels(np.arange(20, 40, 1), labels=[True, False, False, False])

    # plt.legend()
    plt.show()






