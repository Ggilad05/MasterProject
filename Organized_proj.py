import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from Arrange_data import arrange_data, get_nc
from Weighted_variable import calculate_weighted
from Plots_fun import plot_tp
from shapely.geometry import Polygon, Point
from Israel_mask import  is_polygon


def point_in_polygon(polygon, point):
    # Create a Shapely polygon from the list of coordinates
    poly = Polygon(polygon)

    # Create a Shapely point from the list of coordinates
    pt = Point(point)

    # Check if the point is inside the polygon using the .within() method
    return pt.within(poly)

def israel_mask(d, poly):
    mask = np.empty((len(d["latitude"]), len(d["longitude"])))
    mask[:] = np.NaN
    for i in range(len(d["latitude"])):
        for j in range(len(d["longitude"])):
            p = Point(d["longitude"][j], d["latitude"][i])
            if int(p.within(poly)) == 1:
                mask[i, j] = 1

    return d * mask


def main():
    cyclone_data, ims_data = arrange_data()
    data, lons, lats = get_nc()
    tp_weighted = calculate_weighted(data, lons, lats)
    tp_weighted.attrs['units'] = 'mm'
    tp_time_sum = tp_weighted.sum(dim="time")
    tp_time_sum = israel_mask(tp_time_sum, is_polygon)
    tp_tot = tp_time_sum.sum(dim=["longitude", "latitude"])
    print(tp_tot)

    plot_tp(tp_time_sum,lons, lats)
if __name__ == '__main__':
    main()