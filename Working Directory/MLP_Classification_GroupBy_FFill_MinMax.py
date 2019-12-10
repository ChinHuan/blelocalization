from utilities import scanners, read_pin, formatData, minMaxScaling, InfluxDBClient, AnimatedScatter
from models import MLPClassifier

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pin_file = "../Data/pin.csv"
pin = read_pin(pin_file)
enc = joblib.load("../Models/MLP_Classification_GroupBy_FFill_MinMax_Encoder.joblib")
model = MLPClassifier()
model.load('../Models/MLP_Classification_GroupBy_FFill_MinMax.h5')
client = InfluxDBClient()

def predict():
    data = client.retrieveData(seconds=20, beacon="0117C55D14E4")
    data = formatData(data)
    data[scanners] = minMaxScaling(data[scanners])
    data = data.groupby(pd.Grouper(freq="1s")).mean().reset_index()
    data = data.ffill().fillna(0)
    pred = model.predict(data[scanners].values)
    pred = np.argmax(pred, axis=1)
    pred = enc.inverse_transform(pred)
    x = np.vectorize(lambda x: pin[x]["X"])(pred).reshape((-1, 1))
    y = np.vectorize(lambda x: pin[x]["Y"])(pred).reshape((-1, 1))
    return np.expand_dims(np.hstack((x, y)).mean(axis=0), axis=0)

ani = AnimatedScatter(predict)
plt.show()