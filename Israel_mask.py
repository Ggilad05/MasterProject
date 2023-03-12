import regionmask
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from shapely import Polygon



a = regionmask.defined_regions.natural_earth_v5_0_0.countries_50["Israel"].coords
# b = regionmask.defined_regions.natural_earth_v4_1_0.["West Bank"].coords
b = regionmask.defined_regions.natural_earth_v5_0_0.countries_50["Palestine"].coords



israel_coords = []
for p in a:
    israel_coords.append(p)
for p in b:
    israel_coords.append(p)

is_lon = []
is_lat = []
for p in israel_coords:
    is_lon.append(p[0])
    is_lat.append(p[1])
# map = Basemap(projection='cass', lat_0=31.5, lon_0=32.851612, width=505000 * 2, height=790000,
#               resolution='l')
# map.plot(x=is_lon, y=is_lat, latlon=True)
# plt.show()

is_polygon = Polygon(israel_coords)








