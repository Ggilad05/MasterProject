import numpy as np
import paramiko
import time
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
client.connect("132.66.102.172", username="shreibshtein", password="gilad051295")


sftp_client = client.open_sftp()
ncfile = sftp_client.open("/data/iacdc/ECMWF/ERA5/pr_1hrPlev_reanalysis_ERA5_19790101_19790131.nc")
ncfile.prefetch()
b_ncfile = ncfile.read()    # ****
nc = Dataset("nc", memory=b_ncfile, format="NETCDF4")
dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))


tp = dataset["tp"]



tp[0].plot()
plt.show()

# sftp_client.open("/data/iacdc/ECMWF/ERA5/ERA5_DL_year")





client.close()






