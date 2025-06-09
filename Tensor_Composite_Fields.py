import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pickle



def composite_all(dic, years):
    composite_tensor = []
    for y in years:
        print(y)
        file_list = glob.glob(dic + "/" + str(y) + "/*.npy")
        for storm in file_list:
            storm = np.load(storm)
            composite_tensor.append(storm)
    composite_tensor = np.array(composite_tensor)
    composite_tensor = composite_tensor.mean(axis=0)
    plt.imshow(composite_tensor)
    plt.colorbar(label="tp [m]")
    plt.savefig('/data/shreibshtein/Composite/composite_tensor')


if __name__ == '__main__':
    years = np.arange(1979, 2022, 1)
    files_dir = '/data/iacdc/ECMWF/ERA5/Tensors/'
    field_name_list = ['tp_1hr_0.25', 'msl_1hr_0.25', "sst_1hr_0.25"]
    # field_name_list = ['t_1hr_0.25/850']

    with open('/data/shreibshtein/Composite/classify_by_seasons.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    # composite_all(files_dir+field_name, years)
    # composite_tensor = []
    composite_winter_tp = []
    composite_spring_tp = []
    composite_summer_tp = []
    composite_autumn_tp = []

    composite_winter_msl = []
    composite_spring_msl = []
    composite_summer_msl = []
    composite_autumn_msl = []

    composite_winter_ta = []
    composite_spring_ta = []
    composite_summer_ta = []
    composite_autumn_ta = []
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    # seasons = ["Winter"]

    for season in seasons:
        with open('/data/shreibshtein/Composite/season.txt', 'w') as f:
            f.write(season)
        print(season)
        for y in years:
            with open('/data/shreibshtein/Composite/year.txt', 'w') as f:
                f.write(str(y))
            print(y)
            for n in field_name_list:
                print(n)
                file_list = glob.glob(files_dir + n + "/" + str(y) + "/*.npy")
                for storm in file_list:
                    name = Path(storm).stem
                    storm = np.load(storm)
                    composite = storm.mean(axis=0)
                    composite = np.flipud(composite)



                    if np.int64(name) in loaded_dict[str(y)]["Winter"]:
                        if n == "tp_1hr_0.25":
                            composite_winter_tp.append(composite)
                        if n == "msl_1hr_0.25":
                            composite_winter_msl.append(composite)
                        if n == "t_1hr_0.25/850":
                            composite_winter_ta.append(composite)

                    if int(name) in loaded_dict[str(y)]["Spring"]:
                        if n == "tp_1hr_0.25":
                            composite_spring_tp.append(composite)
                        if n == "msl_1hr_0.25":
                            composite_spring_msl.append(composite)
                        if n == "t_1hr_0.25/850":
                            composite_spring_ta.append(composite)

                    if int(name) in loaded_dict[str(y)]["Summer"]:
                        if n == "tp_1hr_0.25":
                            composite_summer_tp.append(composite)
                        if n == "msl_1hr_0.25":
                            composite_summer_msl.append(composite)
                        if n == "t_1hr_0.25/850":
                            composite_summer_ta.append(composite)

                    if int(name) in loaded_dict[str(y)]["Autumn"]:
                        if n == "tp_1hr_0.25":
                            composite_autumn_tp.append(composite)
                        if n == "msl_1hr_0.25":
                            composite_autumn_msl.append(composite)
                        if n == "t_1hr_0.25/850":
                            composite_autumn_ta.append(composite)

    # composite_tensor = np.array(composite_tensor)

    composite_winter_tp = np.array(composite_winter_tp).mean(axis=0)
    composite_spring_tp = np.array(composite_spring_tp).mean(axis=0)
    composite_summer_tp = np.array(composite_summer_tp).mean(axis=0)
    composite_autumn_tp = np.array(composite_autumn_tp).mean(axis=0)

    composite_winter_tp = np.flipud(composite_winter_tp)
    composite_spring_tp = np.flipud(composite_spring_tp)
    composite_summer_tp = np.flipud(composite_summer_tp)
    composite_autumn_tp = np.flipud(composite_autumn_tp)

    composite_winter_msl = np.array(composite_winter_msl).mean(axis=0)
    composite_spring_msl = np.array(composite_spring_msl).mean(axis=0)
    composite_summer_msl = np.array(composite_summer_msl).mean(axis=0)
    composite_autumn_msl = np.array(composite_autumn_msl).mean(axis=0)

    composite_winter_msl = np.flipud(composite_winter_msl)
    composite_spring_msl = np.flipud(composite_spring_msl)
    composite_summer_msl = np.flipud(composite_summer_msl)
    composite_autumn_msl = np.flipud(composite_autumn_msl)

    composite_winter_ta = np.array(composite_winter_ta).mean(axis=0)
    composite_spring_ta = np.array(composite_spring_ta).mean(axis=0)
    composite_summer_ta = np.array(composite_summer_ta).mean(axis=0)
    composite_autumn_ta = np.array(composite_autumn_ta).mean(axis=0)

    composite_winter_ta = np.flipud(composite_winter_ta)
    composite_spring_ta = np.flipud(composite_spring_ta)
    composite_summer_ta = np.flipud(composite_summer_ta)
    composite_autumn_ta = np.flipud(composite_autumn_ta)



    im = plt.imshow(composite_winter_tp)
    cs = plt.contour(composite_winter_msl, colors='r')
    # plt.colorbar(cs, orientation='horizontal', pad=0.05)
    plt.clabel(cs, inline=1, fontsize=10)
    cs = plt.contour(composite_winter_ta, 7, colors='y')
    # plt.colorbar(cs, orientation='horizontal')
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(im, label="tp [m]")
    plt.title("Composite over all Winters")
    plt.savefig('/data/shreibshtein/Composite/composite_winter')
    plt.close()

    im = plt.imshow(composite_spring_tp)
    cs = plt.contour(composite_spring_msl, colors='r')
    # plt.colorbar(cs, orientation='horizontal', pad=0.05)
    plt.clabel(cs, inline=1, fontsize=10)
    cs = plt.contour(composite_spring_ta, 7, colors='y')
    # plt.colorbar(cs, orientation='horizontal')
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(im, label="tp [m]")
    plt.title("Composite over all Springs")
    plt.savefig('/data/shreibshtein/Composite/composite_spring')
    plt.close()
    #

    im = plt.imshow(composite_summer_tp)
    cs = plt.contour(composite_summer_msl, colors='r')
    # plt.colorbar(cs, orientation='horizontal', pad=0.05)
    plt.clabel(cs, inline=1, fontsize=10)
    cs = plt.contour(composite_summer_ta, 7, colors='y')
    # plt.colorbar(cs, orientation='horizontal')
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(im, label="tp [m]")
    plt.title("Composite over all Summers")
    plt.savefig('/data/shreibshtein/Composite/composite_summers')
    plt.close()
    #

    im = plt.imshow(composite_autumn_tp)
    cs = plt.contour(composite_autumn_msl, colors='r')
    # plt.colorbar(cs, orientation='horizontal', pad=0.05)
    plt.clabel(cs, inline=1, fontsize=10)
    cs = plt.contour(composite_autumn_ta, 7, colors='y')
    # plt.colorbar(cs, orientation='horizontal')
    plt.clabel(cs, inline=1, fontsize=10)
    plt.colorbar(im, label="tp [m]")
    plt.title("Composite over all Autumn")
    plt.savefig('/data/shreibshtein/Composite/composite_autumn')
    plt.close()

    exit()
