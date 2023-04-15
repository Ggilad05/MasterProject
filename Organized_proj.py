import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from Arrange_data import arrange_data, get_nc
from Weighted_variable import calculate_weighted
from Plots_fun import plot_tp
from shapely.geometry import Polygon, Point
from Israel_mask import is_polygon, israel_coords
from AccumulatePrepOnSt import hours, mean_reagion
import pandas as pd
import xarray as xr


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
    year = 1979
    d, lons, lats = get_nc(year)
    tp = calculate_weighted(d, lons, lats)
    index_list = []
    index = 1
    track_progression = []

    hours_all = []
    tp_progress_all = []
    tp_life_cycle_all = []
    tp_life_cycle = 0
    tp_progress = []

    tp_all = np.zeros([40, 40])
    # tp_i_acc = xr.DataArray(np.zeros(119).reshape(17, 7),
    #                         dims=["latitude", "longitude"],
    #                         coords=(
    #                             np.arange(29.5, 33.75, 0.25),
    #                             np.arange(34.25, 36, 0.25)))

    tp_i_acc = xr.DataArray(np.zeros(225).reshape(15, 15),
                            dims=["latitude", "longitude"],
                            coords=(
                                np.arange(29.75, 33.5, 0.25),
                                np.arange(33, 36.75, 0.25)))

    continue_list = []
    tp_israel_hour = []
    for track in data.iloc:

        lon_index = list(lons).index(track["Longitude"])
        lat_index = list(lats).index(track["Latitude"])

        if lon_index < 20 or lon_index > len(lons) - 20 or lat_index < 20 or lat_index > len(lats) - 20:
            if [track["Year"], track["Index"]] not in continue_list:
                continue_list.append([track["Year"], track["Index"]])
                print(continue_list)
            continue
        month = str(track["Month"])
        if track["Month"] < 10:
            month = "0" + month[0]
        else:
            month = month[0] + month[1]
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

        if track["Year"] == year:
            np_time = np.datetime64(str(int(track["Year"])) + "-" + month + "-" + day + "T" + hour + ":00:00")
            time_index = np.where(tp["time"] == np_time)[0][0]
            print(year)
            if track["Index"] == index:
                print(index)
                tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc, tp_israel_hour = cyclone_track_calculations(
                    tp, time_index,
                    lat_index, lon_index, tp_life_cycle, tp_progress, track_progression, tp_all, tp_israel_hour, lons,
                    lats, tp_i_acc)
            if track["Index"] != index:
                index_list.append(index)
                hours_all.append(len(tp_progress))

                tp_life_cycle_all.append(tp_life_cycle)
                tp_life_cycle = 0
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

                # tp_i_acc = xr.DataArray(np.zeros(119).reshape(17, 7),
                #                         dims=["latitude", "longitude"],
                #                         coords=(
                #                             np.arange(29.5, 33.75, 0.25),
                #                             np.arange(34.25, 36, 0.25)))
                tp_i_acc = xr.DataArray(np.zeros(225).reshape(15, 15),
                                        dims=["latitude", "longitude"],
                                        coords=(
                                            np.arange(29.75, 33.5, 0.25),
                                            np.arange(33, 36.75, 0.25)))

                t = [i for i in range(0, len(tp_israel_hour))]
                plt.plot(t, tp_israel_hour)
                plt.savefig(
                    'C:/Users/shrei/PycharmProjects/MasterProject/Plots/tp_israel_hour/' + str(track["Year"]) + "_" +
                    "tp_israel_hour" + "_" + str(index) + '.png')
                plt.close()

                tp_israel_hour = []
                # xr.DataArray(track_progression[0]).plot()
                # plt.savefig(
                #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/Merge/' +str(track["Year"])+"_"+
                #     "MERGE" + "_" + str(index) + '.png')
                # plt.close()
                #
                # xr.DataArray(track_progression[0]/len(tp_progress)).plot()
                # plt.savefig(
                #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/Hour_mean/' + str(track["Year"]) + "_" +
                #     "HourMean" + "_" + str(index) + '.png')
                # plt.close()
                #
                # t = [i for i in range(0, len(tp_progress))]
                # plt.plot(t, tp_progress)
                # plt.savefig(
                #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/Tp_Progress/' + str(track["Year"]) + "_" +
                #     "TpProgress" + "_" + str(index) + '.png')
                # plt.close()
                #
                # plt.contourf(tp_all)
                # plt.colorbar()
                # plt.savefig(
                #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/tp_all/' + str(track["Year"]) + " " + str(
                #         index) + '.png')
                # plt.close()

                tp_progress = []
                track_progression = []

                index = track["Index"]
                tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc, tp_israel_hour = cyclone_track_calculations(
                    tp, time_index,
                    lat_index, lon_index, tp_life_cycle, tp_progress, track_progression, tp_all, tp_israel_hour, lons,
                    lats, tp_i_acc)

        if track["Year"] != year:
            year = int(track["Year"])
            data, lons, lats = get_nc(year)
            tp = calculate_weighted(data, lons, lats)
            index_list.append(index)
            hours_all.append(len(tp_progress))

            tp_life_cycle_all.append(tp_life_cycle)
            tp_life_cycle = 0
            tp_progress_all.append(tp_progress)
            track_progression[0] = np.array(track_progression[0])
            for j in range(1, len(track_progression)):
                track_progression[0] += np.array(track_progression[j])

            tp_i_acc = israel_mask(tp_i_acc, is_polygon)
            tp_i_acc.plot()
            plt.savefig(
                'C:/Users/shrei/PycharmProjects/MasterProject/Plots/tp_israel_plot/' + str(track["Year"]) + "_" + str(
                    index) + '.png')
            plt.close()

            # tp_i_acc = xr.DataArray(np.zeros(119).reshape(17, 7),
            #                         dims=["latitude", "longitude"],
            #                         coords=(
            #                             np.arange(29.5, 33.75, 0.25),
            #                             np.arange(34.25, 36, 0.25)))
            tp_i_acc = xr.DataArray(np.zeros(225).reshape(15, 15),
                                    dims=["latitude", "longitude"],
                                    coords=(
                                        np.arange(29.75, 33.5, 0.25),
                                        np.arange(33, 36.75, 0.25)))

            t = [i for i in range(0, len(tp_israel_hour))]
            plt.plot(t, tp_israel_hour)
            plt.savefig(
                'C:/Users/shrei/PycharmProjects/MasterProject/Plots/tp_israel_hour/' + str(track["Year"]) + "_" +
                "tp_israel_hour" + "_" + str(index) + '.png')
            plt.close()

            tp_israel_hour = []

            # xr.DataArray(track_progression[0]).plot()
            # plt.savefig(
            #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/Merge/' + str(track["Year"]) + "_" +
            #     "MERGE" + "_" + str(index) + '.png')
            # plt.close()
            #
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
            #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/Tp_Progress/' + str(track["Year"]) + "_" +
            #     "TpProgress" + "_" + str(index) + '.png')
            # plt.close()
            #
            # plt.contourf(tp_all)
            # plt.colorbar()
            # plt.savefig(
            #     'C:/Users/shrei/PycharmProjects/MasterProject/Plots/tp_all/'+ str(track["Year"])+" "+str(index)+'.png')
            # plt.close()

            tp_progress = []
            track_progression = []

            index = track["Index"]
            np_time = np.datetime64(str(int(track["Year"])) + "-" + month + "-" + day + "T" + hour + ":00:00")
            time_index = np.where(tp["time"] == np_time)[0][0]
            tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc = cyclone_track_calculations(tp, time_index,
                                                                                                         lat_index,
                                                                                                         lon_index,
                                                                                                         tp_life_cycle,
                                                                                                         tp_progress,
                                                                                                         track_progression,
                                                                                                         tp_all,
                                                                                                         tp_israel_hour,
                                                                                                         lons, lats,
                                                                                                         tp_i_acc)

    # pd.DataFrame(tp_all).to_csv("C:/Users/shrei/PycharmProjects/MasterProject/tp_all.csv")
    # pd.DataFrame(tp_life_cycle_all).to_csv("C:/Users/shrei/PycharmProjects/MasterProject/tp_life_cycle_all.csv")
    # pd.DataFrame(tp_progress_all).to_csv("C:/Users/shrei/PycharmProjects/MasterProject/tp_progress_all.csv")
    # pd.DataFrame({"Index": index_list,
    #               "Hours": hours_all,
    #               "tp_life_cycle": tp_life_cycle_all}).to_csv("C:/Users/shrei/PycharmProjects/MasterProject/All.csv")


def cyclone_track_calculations(tp, time_index, lat_index, lon_index, tp_life_cycle, tp_progress, track_progression,
                               tp_all, tp_israel_hour, lons, lats, tp_i_acc):
    precipitation = tp[time_index, lat_index - 20:lat_index + 20,
                    lon_index - 20:lon_index + 20].sum(dim=["longitude", "latitude"])

    tp_i = tp[time_index, lat_index - 20:lat_index + 20,
           lon_index - 20:lon_index + 20]

    # if precipitation == 0:
    #     print(tp[time_index, lat_index - 20:lat_index + 20, lon_index - 20:lon_index + 20])
    #     print("g")
    tp_life_cycle += float(precipitation)
    # print(float(precipitation))
    # print(" ")
    # print(tp_life_cycle)
    # print(" ")
    tp_progress.append(float(precipitation.data))
    # print(tp_progress)
    track_progression.append(tp[time_index, lat_index - 20:lat_index + 20, lon_index - 20:lon_index + 20])
    tp_all += np.array(tp[time_index, lat_index - 20:lat_index + 20, lon_index - 20:lon_index + 20])

    x = lons[lon_index - 20: lon_index + 20]
    y = lats[lat_index - 20: lat_index + 20]

    prep_in_israel_sum = 0
    prep_in_israel_hour_sum = 0

    for i in range(len(tp_i["latitude"])):
        for j in range(len(tp_i["longitude"])):
            p = Point(tp_i["longitude"][j], tp_i["latitude"][i])
            if p.within(is_polygon) and tp_i[np.where(tp_i["longitude"] == p.x)[0][0],
            np.where(tp_i["latitude"] == p.y)[0][0]].data > 0:
                prep_in_israel_sum += tp_i[np.where(tp_i["longitude"] == p.x)[0][0],
                np.where(tp_i["latitude"] == p.y)[0][0]].data

                prep_in_israel_hour_sum += tp_i[np.where(tp_i["longitude"] == p.x)[0][0],
                np.where(tp_i["latitude"] == p.y)[0][0]].data

                tp_i_acc[np.where(tp_i_acc["latitude"] == p.y)[0][0],
                np.where(tp_i_acc["longitude"] == p.x)[0][0]] += tp_i[np.where(tp_i["latitude"] == p.y)[0][0],
                np.where(tp_i["longitude"] == p.x)[0][0]].data

    tp_israel_hour.append(prep_in_israel_hour_sum)

    return tp_life_cycle, tp_progress, track_progression, tp_all, tp_i_acc, tp_israel_hour


if __name__ == '__main__':
    cyclone_data, ims_data = arrange_data()
    cyclones_tracks(cyclone_data)
    # data, lons, lats = get_nc(1979)
    # tp_time_sum=data["tp"].sum(dim="time")
    # plot_tp(tp_time_sum,lons, lats)
    # tp_time_sum.plot()
    # plt.show()
    # tp_weighted = calculate_weighted(data, lons, lats)
    # tp_weighted.attrs['units'] = 'mm'
    # region_prep_check(tp_weighted, lons, lats)
