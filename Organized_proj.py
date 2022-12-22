from Arrange_data import arrange_data, get_nc
from Weighted_variable import calculate_weighted
from Plots_fun import plot_tp








def main():
    cyclone_data, ims_data = arrange_data()
    data, lons, lats = get_nc()
    tp_weighted = calculate_weighted(data, lons, lats)
    tp_weighted.attrs['units'] = 'mm'
    tp_time_sum = tp_weighted.sum(dim="time")
    plot_tp(tp_time_sum, lons, lats)


if __name__ == '__main__':
    main()