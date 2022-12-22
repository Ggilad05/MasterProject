import xarray as xr
import numpy as np



def weighted_area_grid(lat, lon):
    xlon, ylat = np.meshgrid(lon, lat)

    return np.cos(np.deg2rad(ylat))

def calculate_weighted(ds, lons, lats):
    xlon, ylat = np.meshgrid(lons, lats)
    weighted_area = weighted_area_grid(lats, lons)

    """  Units of -->tp[m]: The depth of water on a p
         To get the total precipitation for an hour (mm) :  tp [mm]=tp [m]⋅1000"""

    return ds["tp"] * weighted_area *1000