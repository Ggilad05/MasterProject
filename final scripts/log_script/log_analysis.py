import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import torch
import glob
import pandas as pd
from pathlib import Path








if __name__ == "__main__":

    # folders_names = ['33_0','33_1']
    # mean_list = [11.518224716186523, 11.518182754516602]
    # std_list = [0.01461584959179163, 0.01746928133070469]
    folders_names = ['33_3']
    mean_list = [11.518209457397461]
    std_list = [0.015625]
    base_path = r'C:\Users\shrei\PycharmProjects\MasterProject\final scripts\log_script'
    n = len(folders_names)
    errors = []
    for i in range(n):
        path = os.path.join(base_path, folders_names[i])
        prediction_path = Path(path)
        matching_files = list(prediction_path.rglob("*predictions.pt"))
        print(matching_files[0])
        predictions = torch.load(matching_files[0], map_location='cpu')
        #
        for key, value in predictions.items():
            print(key)
            y_hat, y_true = value[0].cpu(), value[1].cpu()

            denorm_y_hat = torch.exp((y_hat[:, 0] * std_list[i] + mean_list[i])) / 100.0
            denorm_y_true = torch.exp((y_true[:, 0] * std_list[i]  + mean_list[i])) / 100.0
            error = (denorm_y_hat - denorm_y_true).detach().numpy()
            errors.append(error)

    # 2. Calculate Mean Absolute Error (MAE)
    final_errors = np.concatenate(errors)
    mae = np.mean(np.abs(final_errors))
    print(f"\nMean Absolute Error (MAE): {mae:.4f}")

    # 3. Plot the histogram of the errors
    plt.figure(figsize=(10, 6))
    # Use density=True to normalize the histogram
    plt.hist(errors, bins=50, density=True, alpha=0.7, color='g', label='Error Distribution')
