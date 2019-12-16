import pandas as pd
import numpy as np
from influxdb import DataFrameClient
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

scanners = [
    'C400A2E19293', # R1824
    'CD4533FFC0E1', # R1836
    'D2B6503554D7', # R1826
    'DB8B36A69C56', # R1840
    'DD697EA75B68', # R1835
    'DF231643E227', # R1830
    'E13B805C6CB0', # R1825
    'E43355CA8B96', # R1833
    'E6D9D20DD197', # R1831
    'E8FD0B453DC4', # R1837
    'E96AF2C858BA', # R1827
    'EC72840D9AD3', # R1823
    'F1307ECB3B90', # R1834
    'F1EDAF28E08A', # R1821
    'F69A86823B96', # R1828
    'FB2EE01C18CE', # R1829
    'FDAE5980F28C'  # R1832
]

def read_data(filename, beacon=None):
    data = pd.read_csv(filename)
    
    print("All beacons:", data["beacon"].unique())
    # If beacon is None, select all
    if beacon is not None:
        data = data[data["beacon"] == beacon]
        print("Selecting", beacon)
    else:
        print("Selecting all")

    data = data.pivot_table(columns="scanner", values="rssi", index=["time", "location"])
    data.rename_axis(None, axis=1, inplace=True)
    data.reset_index(inplace=True)
    
    if data["time"].dtypes != "datetime64[ns]":
        data["time"] = pd.to_datetime(data["time"])
    
    return data

def read_pin(filename):
    pin = pd.read_csv(filename)
    pinMap = pin.set_index("Id").transpose()
    return pinMap.to_dict()

def read_ble(filename):
    pin = pd.read_csv(filename)[['BLE_MAC', 'X', 'Y']]
    pinMap = pin.set_index("BLE_MAC").transpose()
    return pinMap.to_dict()

def train_validation_test_split(df, train_portion=0.6, validation_portion=0.2, random_state=123456):
    first = train_portion
    second = train_portion + validation_portion
    return np.split(df.sample(frac=1, random_state=random_state), [int(first * len(df)), int(second * len(df))])

def train_validation_split(df, train_portion=0.8, random_state=123456):
    return np.split(df.sample(frac=1, random_state=random_state), [int(train_portion * len(df))])

def min_max_scaling(df, min=-100, max=-40):
    return (df - min) / (max - min)

def format_data(data):
    data = data.pivot(columns="scanner", values="rssi").rename_axis(None, axis=1)
    data = data.reindex(columns=scanners)
    return data

class InfluxDBClient:
    def __init__(self):
        self.client = DataFrameClient(host='localhost', port=8086)
        self.client.switch_database('Megallen2')

    def retrieveData(self, seconds=5, beacon="0117C55D14E4"):
        res = self.client.query("SELECT * FROM OnlineRSSI WHERE time > now() - {}s and beacon = \'{}\'".format(seconds, beacon))
        if res == {}:
            raise Exception("No data incoming")
        return res["OnlineRSSI"]

class AnimatedScatter(object):
    def __init__(self, predict):
        self.predict = predict
        self.stream = self.data_stream()
        self.fig, self.ax = plt.subplots(figsize=(15,15))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=100, init_func=self.setup_plot, blit=True)

        img =  mpimg.imread('../Map/main.png')
        self.ax.imshow(img)

    def setup_plot(self):
        point = next(self.stream)
        self.scat = self.ax.scatter(point[:, 0] * 44, point[:, 1] * 44, edgecolor='k', s=100)
        return self.scat,

    def data_stream(self):
        while True:
            yield self.predict()

    def update(self, i):
        data = next(self.stream)
        self.scat.set_offsets(data[:, :2] * 44)
        return self.scat,
