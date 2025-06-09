import numpy as np
import matplotlib.pyplot as plt
from Arrange_data import arrange_data
import pandas as pd

if __name__ == '__main__':

    lon_t = []
    lat_t = []

    # Get trajectories data
    trajectories_data = arrange_data().sort_values(by=["Year", "Index"])
    filter_months = trajectories_data.where(
        (trajectories_data["Month"] == 12) | (trajectories_data["Month"] == 1) | (
                trajectories_data["Month"] == 2)).dropna()
    years = np.arange(1979, 2021, 1)

    # Count number of cyclones
    cyclone_counter = 0
    for year in years:
        # fig = plt.figure(figsize=(12, 6))
        # ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))

        list_year = filter_months["Year"] == year
        list_year = filter_months.where(list_year).dropna()
        group_year_index = list_year.groupby(by=["Index"])
        keys = list(group_year_index.groups.keys())

        # Run over each cyclone
        for k in keys:
            group = group_year_index.get_group(k)
            print(group)
            exit()

            # Check if inside box
            check_inside_box = False
            for i, j in zip((group["Longitude"].between(25, 40)), (group["Latitude"].between(25, 38))):
                if check_inside_box:
                    continue
                if i == True & j == True:
                    check_inside_box = True
                if check_inside_box:
                    x = group_year_index.get_group(k)["Longitude"]
                    y = group_year_index.get_group(k)["Latitude"]

                    lon_t.append(x.values)
                    lat_t.append(y.values)
    table = pd.DataFrame({'Longitude': lon_t, 'Latitude': lat_t})
    print(table)
    exit()
