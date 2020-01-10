from utilities import scanners, format_data, min_max_scaling, read_ble
from utilities import InfluxDBClient, AnimatedScatter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

ble = read_ble('../Data/ble.csv')
coor = np.array([0, 0])
scanners.remove('D2B6503554D7')

def create_window(dataset, win_size, start_index=0, end_index=None):
    data = []

    start_index = start_index + win_size
    if end_index is None:
        end_index = len(dataset)

    for i in range(start_index, end_index):
        indices = range(i-win_size, i)
        data.append(dataset[scanners].iloc[indices].values)
    return np.array(data)

def impute(data):
    data.update(min_max_scaling(data))
    data.update(data.rolling('5s').mean())
    data.ffill(inplace=True)

def reg_impute(data):
    for s in scanners:
        dx = coor[0] - ble[s]['X']
        dy = coor[1] - ble[s]['Y']
        d = np.sqrt(np.square(dy) + np.square(dx))
        reg = joblib.load('../Models/linear_distance_models_2/{}.joblib'.format(s))
        ss = pd.Series(reg.predict(d.reshape((-1, 1)))).repeat(data.shape[0])
        ss.index = data.index
        data[s].fillna(ss, inplace=True)

def inv_scale(y):
    s = np.array([33.5, 16.8])
    return y * s

def predict():
    global coor
    data = client.retrieveData(seconds=10, beacon="0117C55D14E4")
    data = format_data(data)
    impute(data)
    reg_impute(data)
    data = create_window(data, 10)
    coor = model.predict(data).mean(axis=0)
    coor = coor.mean(axis=0)
    coor = inv_scale(coor)
    print(coor)
    return np.expand_dims(coor, axis=0)

model = tf.keras.models.load_model("../Models/LSTM_W10.h5")
client = InfluxDBClient()

ani = AnimatedScatter(predict)
plt.show()


