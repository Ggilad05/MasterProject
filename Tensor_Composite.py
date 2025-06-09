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
    years = np.arange(1979, 2021, 1)
    files_dir = '/data/iacdc/ECMWF/ERA5/Tensors/'
    field_name = 'tp_1hr_0.25'

    with open('/data/shreibshtein/Composite/classify_by_seasons.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    # composite_all(files_dir+field_name, years)

    # composite_tensor = []
    composite_winter = []
    composite_spring = []
    composite_summer = []
    composite_autumn = []
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    for season in seasons:
        print(season)
        for y in years:
            print(y)
            file_list = glob.glob(files_dir + field_name + "/" + str(y) + "/*.npy")
            for storm in file_list:
                name = Path(storm).stem
                storm = np.load(storm)
                composite = storm.mean(axis=0)
                plt.imshow(composite)
                plt.show()
                if int(name) in loaded_dict[str(y)]["Winter"]:
                    composite_winter.append(composite)
                if int(name) in loaded_dict[str(y)]["Spring"]:
                    composite_spring.append(composite)
                if int(name) in loaded_dict[str(y)]["Summer"]:
                    composite_summer.append(composite)
                if int(name) in loaded_dict[str(y)]["Autumn"]:
                    composite_autumn.append(composite)

    # composite_tensor = np.array(composite_tensor)



    composite_winter = np.array(composite_winter)
    composite_spring = np.array(composite_spring)
    composite_summer = np.array(composite_summer)
    composite_autumn = np.array(composite_autumn)

    composite_winter = composite_winter.mean(axis=0)
    composite_spring = composite_spring.mean(axis=0)
    composite_summer = composite_summer.mean(axis=0)
    composite_autumn = composite_autumn.mean(axis=0)
    # composite_tensor = composite_tensor.mean(axis=0)

    plt.imshow(composite_winter)
    plt.colorbar(label="tp [m]")
    plt.savefig('/data/shreibshtein/Composite/composite_winter')
    plt.close()

    plt.imshow(composite_spring)
    plt.colorbar(label="tp [m]")
    plt.savefig('/data/shreibshtein/Composite/composite_spring')
    plt.close()

    plt.imshow(composite_summer)
    plt.colorbar(label="tp [m]")
    plt.savefig('/data/shreibshtein/Composite/composite_summer')
    plt.close()

    plt.imshow(composite_autumn)
    plt.colorbar(label="tp [m]")
    plt.savefig('/data/shreibshtein/Composite/composite_autumn')
    plt.close()


            # plt.imshow(composite)
            # plt.colorbar(label="tp [m]")
            # plt.savefig('/data/shreibshtein/Composite/'+field_name+'/' +str(y)+'/'+name)
            # plt.close()
