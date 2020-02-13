from utilities import scanners, format_data, min_max_scaling, read_ble
from utilities import InfluxDBClient, AnimatedScatter

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def linear_impute(data):
    for s in scanners:
        dx = coor[0] - ble[s]['X']
        dy = coor[1] - ble[s]['Y']
        d = np.sqrt(np.square(dy) + np.square(dx))
        reg = joblib.load('../Models/linear_distance_models_2/{}.joblib'.format(s))
        ss = pd.Series(reg.predict(np.repeat(d, data.shape[0]).reshape((-1, 1))))
        ss.index = data.index
        data[s].fillna(ss, inplace=True)

def impute(data):
    rolling_win = 60

    data_rolled = data.rolling(rolling_win, min_periods=1).mean()
    data_imputed = data_rolled.ffill()
    linear_impute(data_imputed)
    return data_imputed

ble = read_ble('../Data/ble.csv')
coor = np.array([0, 0])

def predict():
    global coor
    data = client.retrieveData(seconds=15, beacon="0117C55D14E4")
    data = format_data(data)
    data[scanners] = impute(data[scanners])
    data[scanners] = normalize(data[scanners] + 100)
    coor = model.predict(data.values).mean(axis=0)
    print(coor)
    return np.expand_dims(coor, axis=0)

model = tf.keras.models.load_model("../Models/MLP_CombineData.h5")
client = InfluxDBClient()

ani = AnimatedScatter(predict)
plt.show()
