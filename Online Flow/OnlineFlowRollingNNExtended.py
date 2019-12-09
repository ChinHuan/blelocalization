import joblib
import pandas as pd
import numpy as np
from influxdb import DataFrameClient
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

scanners = [
    'C400A2E19293', 'CD4533FFC0E1', 'D2B6503554D7', 'DB8B36A69C56', 'DD697EA75B68', 'DF231643E227', 
    'E13B805C6CB0', 'E43355CA8B96', 'E6D9D20DD197', 'E8FD0B453DC4', 'E96AF2C858BA', 'EC72840D9AD3', 
    'F1307ECB3B90', 'F1EDAF28E08A', 'F69A86823B96', 'FB2EE01C18CE', 'FDAE5980F28C'
]

scaler = joblib.load("../Models/rolling_nn3/scaler")
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[17]),
    layers.Dense(2, kernel_regularizer=regularizers.l2(1), activity_regularizer=regularizers.l1(1))
])

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
model.load_weights("../Models/rolling_nn3/cp.ckpt")

client = DataFrameClient(host='localhost', port=8086)
client.get_list_database()
client.switch_database('Megallen2')

class AnimatedScatter(object):
    def __init__(self, numpoints=50):
        self.stream = self.data_stream()
        self.fig, self.ax = plt.subplots(figsize=(15,15))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=500, init_func=self.setup_plot, blit=True)

        img =  mpimg.imread('../Map/main.png')
        self.ax.imshow(img)

    def setup_plot(self):
        point = next(self.stream)
        self.scat = self.ax.scatter(point[:, 0] * 44, point[:, 1] * 44, edgecolor='k', s=100)
        return self.scat,

    def data_stream(self):
        while True:
            res = client.query("SELECT * FROM OnlineRSSI WHERE time > now() - 10s")
            if res == {}:
                raise Exception("No data incoming")
            res = res["OnlineRSSI"]

            B1 = res[res["beacon"] == "0117C55D14E4"]
            B1 = B1.pivot(columns="scanner", values="rssi").rename_axis(None, axis=1)
            B1 = B1.reindex(columns=scanners)

            data = B1.rolling(10, min_periods=1).mean()
            data = data.ffill().fillna(-100)
            data = scaler.transform(data)
            pred = model.predict(data)
            loc = np.expand_dims(pred.mean(axis=0), axis=0)
            yield loc

    def update(self, i):
        data = next(self.stream)
        self.scat.set_offsets(data[:, :2] * 44)
        return self.scat,

ani = AnimatedScatter()
plt.show()