import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# main

data_path = 'C:/Users/shrei/PycharmProjects/MasterProject/2.2019.csv'
data = pd.read_csv(data_path)
st = {"Jerusalem Givat Ram": [],
            "Jerusalem Centre": [],
            "Netiv Halamed He": [],
            "Shani": [],
            "Arad": [],
            "Bet Dagan": [],
            "Beit Jimal": [],
            "Rosh Zurim": [],
            "Tel Aviv Coast": [],
            "Zova": [],
            "Maale Adummim": [],
            "Kefar Blum": [],
            "Kefar Giladi": [],
            "Nevatim": []}

mean_reagion = {"1": [], "4": [], "5": [], "6": [], "7": []}



curr_h = 0
s = 0

for n in st:
    for i in data.index:
        if curr_h == 5:
            s += data[n][i]
            curr_h = 0
            st[n].append(s)
            s = 0.0
        else:
            s += data[n][i]
            curr_h += 1

mean_reagion["1"].append(np.array([sum(i) for i in zip(st["Kefar Blum"], st["Kefar Giladi"])])/2)
mean_reagion["4"].append(np.array([sum(i) for i in zip(st["Jerusalem Centre"], st["Jerusalem Givat Ram"]
                                                       ,st["Maale Adummim"])])/3)
mean_reagion["5"].append(np.array([sum(i) for i in zip(st["Zova"], st["Rosh Zurim"],st["Beit Jimal"],
st["Netiv Halamed He"])])/4)
mean_reagion["6"].append(np.array([sum(i) for i in zip(st["Nevatim"], st["Shani"],st["Arad"])])/3)
mean_reagion["7"].append(np.array([sum(i) for i in zip(st["Tel Aviv Coast"], st["Bet Dagan"])])/2)

# print(len(mean_reagion["1"][0]))
# print(len(mean_reagion["4"][0]))
# print(len(mean_reagion["5"][0]))
# print(len(mean_reagion["6"][0]))
# print(len(mean_reagion["7"][0]))
#
# print(mean_reagion)

hours = pd.date_range("2019-02-01", "2019-03-1",freq="H")
hours= hours[0:-1]









# plt.plot(hours, mean_reagion["1"][0], label = "1")
# plt.plot(hours, mean_reagion["4"][0], label = "4")
# plt.plot(hours, mean_reagion["5"][0], label = "5")
# plt.plot(hours, mean_reagion["6"][0], label = "6")
# plt.plot(hours, mean_reagion["7"][0], label = "7")
#
# plt.legend()
#
# plt.show()







