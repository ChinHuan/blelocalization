from utilities import scanners, read_pin, formatData, minMaxScaling, InfluxDBClient, AnimatedScatter

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict():
    data = client.retrieveData(seconds=10, beacon="0117C55D14E4")
    data = formatData(data)
    data[scanners] = minMaxScaling(data[scanners])
    data = data.rolling(15, min_periods=1).mean().reset_index()
    data = data.ffill().fillna(0)
    pred = model.predict(data[scanners].values)
    pred = enc.inverse_transform(pred)
    x = np.vectorize(lambda x: pin[x]["X"])(pred).reshape((-1, 1))
    y = np.vectorize(lambda x: pin[x]["Y"])(pred).reshape((-1, 1))
    return np.expand_dims(np.hstack((x, y)).mean(axis=0), axis=0)

pin_file = "../Data/pin.csv"
pin = read_pin(pin_file)
enc = joblib.load("../Models/MLP_Classification_Rolling_FFill_MinMax_Encoder.joblib")
model = joblib.load('../Models/RF_Classification_Rolling_FFill_MinMax.joblib')
client = InfluxDBClient()

ani = AnimatedScatter(predict)
plt.show()