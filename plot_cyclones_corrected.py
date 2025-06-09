import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from Arrange_data_corrected import read_trajectories
import pandas as pd

if __name__ == '__main__':

    years = np.arange(1979, 1990, 1)
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    count = 0
    for y in years:
        print(y)
        storms_data = read_trajectories(y)
        # count += len(storms_data)
        for s in range(len(storms_data)):
            group_data = storms_data.iloc[s]["Grouped data"]
            x = group_data["Longitude"]
            y = group_data["Latitude"]

            # Check if inside box
            check_inside_box = False
            for i, j in zip(x.between(25, 40), y.between(25, 38)):
                if check_inside_box:
                    continue
                if i == True & j == True:
                    check_inside_box = True
                if check_inside_box:
                    count += 1

                    plt.plot(x, y, transform=ccrs.Geodetic(), linewidth=3)
        ax.coastlines()
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle='-')
        ax.add_feature(cfeature.OCEAN)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                               linewidth=.6, color='gray', alpha=0.8, linestyle='-.')
        gl.xlabel_style = {"size": 15}
        gl.ylabel_style = {"size": 15}
        ax.set_extent((15, 45, 40, 23))
        # plt.savefig("/data/shreibshtein/plots/picc")
        plt.show()
    print(count)


