import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from Arrange_data import arrange_data, get_nc
from Weighted_variable import calculate_weighted
from Plots_fun import plot_tp
from shapely.geometry import Polygon, Point
# from Israel_mask import is_polygon, israel_coords
# from AccumulatePrepOnSt import hours, mean_reagion
import pandas as pd
import xarray as xr
from composite import composite_pr


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
            # if int(p.within(poly)) == 1:
            if poly.contains(p):
                mask[i, j] = 1
    israel = xr.DataArray(data=d * mask, attrs=dict(
        description="Total Precipitation",
        units="mm"))
    israel

    return israel


def region_prep_check(tp_weighted, lons, lats):
    region_1 = tp_weighted.sel(longitude=35.61, latitude=33.17, method='nearest')
    region_4 = tp_weighted.sel(longitude=35.22, latitude=31.78, method='nearest')
    region_5 = tp_weighted.sel(longitude=35.12, latitude=31.78, method='nearest')
    region_6 = tp_weighted.sel(longitude=34.918, latitude=31.2, method='nearest')
    region_7 = tp_weighted.sel(longitude=34.75, latitude=32.05, method='nearest')

    fig, axs = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    axs[0, 0].plot(hours, mean_reagion["1"][0], label="IMS 1 (North)")
    axs[0, 0].plot(hours, region_1, label="ERA5")
    axs[0, 0].legend()
    # plt.title("Mean IMS 1 vs ERA5")
    axs[0, 0].set_ylabel("Prep [mm]")

    axs[0, 1].plot(hours, mean_reagion["4"][0], label="IMS 4 (Jerusalem)")
    axs[0, 1].plot(hours, region_4, label="ERA5")
    axs[0, 1].legend()
    # plt.title("Mean IMS 4 vs ERA5")
    # plt.ylabel("Prep [mm]")

    axs[0, 2].plot(hours, mean_reagion["5"][0], label="IMS 5 (Soth to Jerusalem)")
    axs[0, 2].plot(hours, region_5, label="ERA5")
    axs[0, 2].legend()
    # plt.title("Mean IMS 5 vs ERA5")
    # plt.ylabel("Prep [mm]")

    axs[1, 0].plot(hours, mean_reagion["6"][0], label="IMS 6 (Negev)")
    axs[1, 0].plot(hours, region_6, label="ERA5")
    axs[1, 0].legend()
    # plt.title("Mean IMS 5 vs ERA5")
    axs[1, 0].set_ylabel("Prep [mm]")

    axs[1, 1].plot(hours, mean_reagion["7"][0], label="IMS 7 (Tel Aviv)")
    axs[1, 1].plot(hours, region_7, label="ERA5")
    axs[1, 1].legend()
    # plt.title("Mean IMS 7 vs ERA5")
    # plt.ylabel("Prep [mm]")
    # plt.show()

    pd_1 = pd.array(mean_reagion["1"][0])
    pd_4 = pd.array(mean_reagion["4"][0])
    pd_5 = pd.array(mean_reagion["5"][0])
    pd_6 = pd.array(mean_reagion["6"][0])
    pd_7 = pd.array(mean_reagion["7"][0])
    pd_all = pd.DataFrame({"Date": hours, "1": pd_1, "4": pd_4, "5": pd_5, "6": pd_6, "7": pd_7, "IMS 1": region_1,
                           "IMS 4": region_4, "IMS 5": region_5, "IMS 6": region_6, "IMS 7": region_7})
    pd_all.to_csv("pd_all.csv")

    tp_time_sum = tp_weighted.sum(dim="time")
    plot_tp(tp_time_sum, lons, lats)
    tp_time_sum = israel_mask(tp_time_sum, is_polygon)
    tp_tot = tp_time_sum.sum(dim=["longitude", "latitude"])
    tp_tot.plot()


def cyclones_tracks(data):

    years = [year for year in range(1979, 2021)]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    all = 0
    for year in years:
        print(year)
        for month in months:
            print(month)
            d, lons, lats = get_nc(year, month)
            # tp = calculate_weighted(d, lons, lats)
            index_list = []

            tp_all = np.zeros([40, 40])

            tp_i_acc = xr.DataArray(np.zeros(225).reshape(15, 15),
                                    dims=["latitude", "longitude"],
                                    coords=(
                                        np.arange(29.75, 33.5, 0.25),
                                        np.arange(33, 36.75, 0.25)))

            filterd_data = data.loc[(data["Year"] == year) & (data["Month"] == int(month))]
            last_index = filterd_data["Index"].iloc[-1]

            for index in range(1, last_index + 1):
                con = False
                filterd_data = data.loc[(data["Year"] == year) & (data["Index"] == index)]

                for j in range(0, np.shape(filterd_data)[0]):
                    if list(filterd_data["Longitude"])[j] > 35 and list(filterd_data["Longitude"])[j] < 40 and \
                            list(filterd_data["Latitude"])[j] > 30 and list(filterd_data["Latitude"])[j] < 40:
                        con = True
                if con:

                    track_progression = []
                    hours_all = []
                    tp_progress_all = []
                    tp_life_cycle_all = []
                    tp_life_cycle = 0
                    tp_progress = []
                    tp_israel_hour = []
                    print(index)
                    for track in filterd_data.loc[filterd_data["Index"] == index].iloc:
                        " lon index 15"
                        lon_index = list(lons).index(track["Longitude"])
                        lat_index = list(lats).index(track["Latitude"])

                        month_track = str(track["Month"])

                        # months str correction
                        if track["Month"] < 10:
                            month_track = "0" + month_track[0]
                        else:
                            month_track = month_track[0] + month_track[1]
                        day = str(track["Day"])
                        if track["Day"] < 10:
                            day = "0" + day[0]
                        else:
                            day = day[0] + day[1]
                        hour = str(track["Hour"])
                        if track["Hour"] < 10:
                            hour = "0" + hour[0]
                        else:
                            hour = hour[0] + hour[1]

                        np_time = np.datetime64(str(int(track["Year"])) + "-" + month + "-" + day + "T" + hour + ":00:00")

                        tp = calculate_weighted(d.isel(time=np.where(d["time"] == np_time)[0][0],
                                                                                          longitude=[i for i in range(lon_index - 20, lon_index + 20)],
                                                                                          latitude=[i for i in range(lat_index - 20, lat_index + 20)])
                                                                                   , lons[lon_index - 20:lon_index + 20],
                                                                                   lats[lat_index - 20:lat_index + 20])

                        tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc, tp_israel_hour, all = cyclone_track_calculations(
                                                               tp,
                                                               lat_index, lon_index, tp_life_cycle, tp_progress, track_progression, tp_all, tp_israel_hour,
                                                               lons,
                                                               lats, tp_i_acc, all)

                    # if lon_index > 19 and lon_index < 1421:
                    #     tp = calculate_weighted(d.isel(time=np.where(d["time"] == np_time)[0][0],
                    #                                    longitude=[i for i in range(lon_index - 20, lon_index + 20)],
                    #                                    latitude=[i for i in range(lat_index - 20, lat_index + 20)])
                    #                             , lons[lon_index - 20:lon_index + 20],
                    #                             lats[lat_index - 20:lat_index + 20])
                    #
                    #     tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc, tp_israel_hour = cyclone_track_calculations(
                    #         tp,
                    #         lat_index, lon_index, tp_life_cycle, tp_progress, track_progression, tp_all, tp_israel_hour,
                    #         lons,
                    #         lats, tp_i_acc)
                    #
                    # if (lon_index < 20):
                    #     lon = [i for i in range(0, lon_index + 20)]
                    #     rest = [-i for i in range((20 - lon_index), 0, -1)]
                    #     lon = rest + lon
                    #
                    #     lons_tp = list(lons[-(20 - lon_index):-1]) + list(lons[0:lon_index + 21])
                    #     tp = calculate_weighted(d.isel(time=np.where(d["time"] == np_time)[0][0],
                    #                                    longitude=lon,
                    #                                    latitude=[i for i in range(lat_index - 20, lat_index + 20)])
                    #                             , lons_tp, lats[lat_index - 20:lat_index + 20])
                    #
                    #     tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc, tp_israel_hour = cyclone_track_calculations(
                    #         tp,
                    #         lat_index, lon_index, tp_life_cycle, tp_progress, track_progression, tp_all, tp_israel_hour,
                    #         lons,
                    #         lats, tp_i_acc)
                    #
                    # if (lon_index > 1420):
                    #     lon = [i for i in range(lon_index - 20, 1440)]
                    #     rest = [i for i in range(0, 20 - (1440 - lon_index))]
                    #     lon = lon + rest
                    #
                    #     lons_tp = list(lons[lon_index - 20:1440]) + list(lons[0:20 - (1440 - lon_index)])
                    #     tp = calculate_weighted(d.isel(time=np.where(d["time"] == np_time)[0][0],
                    #                                    longitude=lon,
                    #                                    latitude=[i for i in range(lat_index - 20, lat_index + 20)])
                    #                             , lons_tp, lats[lat_index - 20:lat_index + 20])

                        # tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc, tp_israel_hour = cyclone_track_calculations(
                        #     tp,
                        #     lat_index, lon_index, tp_life_cycle, tp_progress, track_progression, tp_all, tp_israel_hour,
                        #     lons,
                        #     lats, tp_i_acc)

                index_list.append(index)
                hours_all.append(len(tp_progress))

                tp_life_cycle_all.append(tp_life_cycle)
                tp_progress_all.append(tp_progress)
                track_progression[0] = np.array(track_progression[0])
                for j in range(1, len(track_progression)):
                    track_progression[0] += np.array(track_progression[j])

                tp_i_acc = israel_mask(tp_i_acc, is_polygon)
                tp_i_acc.plot()
                plt.savefig(
                    'C:/Users/shrei/PycharmProjects/MasterProject/Plots/tp_israel_plot/' + str(
                        track["Year"]) + "_" + str(index) + '.png')
                plt.close()

                # tp_i_acc = xr.DataArray(np.zeros(225).reshape(15, 15),
                #                         dims=["latitude", "longitude"],
                #                         coords=(
                #                             np.arange(29.75, 33.5, 0.25),
                #                             np.arange(33, 36.75, 0.25)))

                # t = [i for i in range(0, len(tp_israel_hour))]
                # plt.plot(t, tp_israel_hour)
                # plt.savefig(
                #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/tp_israel_hour/' + str(
                #         track["Year"]) + "_" +
                #     "tp_israel_hour" + "_" + str(index) + '.png')
                # plt.close()
                #
                # tp_israel_hour = []
                # xr.DataArray(track_progression[0]).plot()
                # plt.savefig(
                #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/Merge/' + str(track["Year"]) + "_" +
                #     "MERGE" + "_" + str(index) + '.png')
                # plt.close()
                #
                # xr.DataArray(track_progression[0] / len(tp_progress)).plot()
                # plt.savefig(
                #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/Hour_mean/' + str(track["Year"]) + "_" +
                #     "HourMean" + "_" + str(index) + '.png')
                # plt.close()
                #
                # t = [i for i in range(0, len(tp_progress))]
                # plt.plot(t, tp_progress)
                # plt.savefig(
                #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/Tp_Progress/' + str(
                #         track["Year"]) + "_" +
                #     "TpProgress" + "_" + str(index) + '.png')
                # plt.close()
                #
            plt.contourf(tp_all/all)
            plt.colorbar()
            plt.savefig(
                    'C:/Users/shrei/PycharmProjects/MasterProject/Plots/tp_all/' + '.png')
            plt.close()
            #


def cyclone_track_calculations(tp, lat_index, lon_index, tp_life_cycle, tp_progress, track_progression,
                               tp_all, tp_israel_hour, lons, lats, tp_i_acc, all):
    precipitation = tp.sum(dim=["longitude", "latitude"])
    all += float(precipitation)
    tp_life_cycle += float(precipitation)

    tp_progress.append(float(precipitation.data))
    # print(tp_progress)
    track_progression.append(tp)
    tp_all += np.array(tp)

    x = lons[lon_index - 20: lon_index + 20]
    y = lats[lat_index - 20: lat_index + 20]

    prep_in_israel_sum = 0
    prep_in_israel_hour_sum = 0

    if 110 < lon_index < 160 and lat_index > 200 and lat_index < 240:
        for i in range(len(tp["latitude"])):
            for j in range(len(tp["longitude"])):
                p = Point(tp["longitude"][j], tp["latitude"][i])
                if p.within(is_polygon) and tp[np.where(tp["longitude"] == p.x)[0][0],
                np.where(tp["latitude"] == p.y)[0][0]].data > 0:
                    prep_in_israel_sum += tp[np.where(tp["longitude"] == p.x)[0][0],
                    np.where(tp["latitude"] == p.y)[0][0]].data

                    prep_in_israel_hour_sum += tp[np.where(tp["longitude"] == p.x)[0][0],
                    np.where(tp["latitude"] == p.y)[0][0]].data

                    tp_i_acc[np.where(tp_i_acc["latitude"] == p.y)[0][0],
                    np.where(tp_i_acc["longitude"] == p.x)[0][0]] += tp[np.where(tp["latitude"] == p.y)[0][0],
                    np.where(tp["longitude"] == p.x)[0][0]].data

    tp_israel_hour.append(prep_in_israel_hour_sum)

    return tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc, tp_israel_hour, all


def plot_cyclones(data):
    map = Basemap(projection='cass', lat_0=33.5, lon_0=32.851612, width=8000000, height=8000000,
                  resolution='l')
    years = [y for y in range(1979, 2021)]
    index_list = []
    f_y = [1987, 1989, 2004, 2005, 2015, 2016]
    f_index = [577, 624, 41, 575, 617, 514]
    for y in years:
        print(y)

        for index in range(1, data.loc[data["Year"] == y]["Index"].iloc[-1] + 1):
            con = False
            filterd_data = data.loc[(data["Year"] == y) & (data["Index"] == index)]

            for j in range(0, np.shape(filterd_data)[0]):
                '''(filterd_data.loc[filterd_data["Longitude"][j] > 20] or filterd_data.loc[
                    filterd_data["Longitude"][j] < 45]) and (
                            filterd_data.loc[filterd_data["Latitude"][j] > 25] or filterd_data.loc[
                        filterd_data["Latitude"][j] < 45])'''
                print(list(filterd_data["Longitude"])[j])
                if list(filterd_data["Longitude"])[j] > 30 and list(filterd_data["Longitude"])[j] < 40 and list(filterd_data["Latitude"])[j] > 28 and list(filterd_data["Latitude"])[j] < 40:
                    con = True
            if con:
                print(filterd_data)
                index_list.append(index)
                data_yi = data.loc[(data["Year"] == y) & (data["Index"] == index)]
                longitude_yi = data_yi["Longitude"]
                latitude_yi = data_yi["Latitude"]
                # for w in range(len(f_y)):
                #     if y == f_y[w] and index == f_index[w]:
                map.plot(np.array(longitude_yi), np.array(latitude_yi), latlon=True, linewidth=5)
    map.drawcoastlines()
    plt.savefig(
                            '/data/shreibshtein/plots/' + str(
                                y) + "_" + str(index) + '.png')
    plt.close()

    print(len(index_list))
    # map.drawcoastlines()
    # map.drawmeridians(np.arange(-90, 90, 5), labels=[True, False, False, True])
    # map.drawparallels(np.arange(-180, 180, 5), labels=[True, False, False, False])
    # plt.show()






if __name__ == '__main__':
    cyclone_data = arrange_data()  # Tracks of cyclones
    # plot_cyclones(cyclone_data)
    # cyclones_tracks(cyclone_data)  # For cyclone track...
    composite_pr(cyclone_data.copy())
    # data, lons, lats = get_nc(1979)
    # tp_time_sum=data["tp"].sum(dim="time")
    # plot_tp(tp_time_sum,lons, lats)
    # tp_time_sum.plot()
    # plt.show()
    # tp_weighted = calculate_weighted(data, lons, lats)
    # tp_weighted.attrs['units'] = 'mm'
    # region_prep_check(tp_weighted, lons, lats)
