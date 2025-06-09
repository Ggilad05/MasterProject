from Arrange_data import get_nc
import numpy as np
from weighted_area import area_grid
import matplotlib.pyplot as plt
import xarray as xr

def composite_pr(data):
    years = [year for year in range(1979, 2021)]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    tp_composite_all = np.zeros([40, 40])
    index_counter = 0
    for year in years:
        print(year)
    for month in months:
        print(month)
        d_pr, lons_pr, lats_pr = get_nc(year, month)

        filterd_data = data.loc[(data["Year"] == year) & (data["Month"] == int(month))]
        last_index = filterd_data["Index"].iloc[-1]

        tp_composite_index = np.zeros([40, 40])

        for index in range(1, last_index + 1):

            con = False
            filterd_data = data.loc[(data["Year"] == year) & (data["Index"] == index) & (data["Month"] == int(month))]

            for j in range(0, np.shape(filterd_data)[0]):
                if list(filterd_data["Longitude"])[j] > 30 and list(filterd_data["Longitude"])[j] < 40 and \
                        list(filterd_data["Latitude"])[j] > 28 and list(filterd_data["Latitude"])[j] < 40:
                    con = True
            if con:
                index_counter += 1
                h = 0

            for track in filterd_data.loc[filterd_data["Index"] == index].iloc:
                print(track)
                lon_index = list(lons_pr).index(track["Longitude"])
                lat_index = list(lats_pr).index(track["Latitude"])

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

                np_time = np.datetime64(str(int(track["Year"])) + "-" + month_track + "-" + day + "T" + hour + ":00:00")



                h += 1

            tp_composite_all += np.array(tp_composite_index / h)

            with open('index.txt', 'w') as f:
                f.write(index_counter)


print(index_counter)
print(tp_composite_all)
tp_composite_all = tp_composite_all / index_counter
plt.contourf(tp_composite_all)
plt.colorbar()
plt.savefig('/data/shreibshtein/plots/p/pic_4.png')
plt.close()
with open('readme.txt', 'w') as f:
    f.write(str(np.sum(tp_composite_all)))