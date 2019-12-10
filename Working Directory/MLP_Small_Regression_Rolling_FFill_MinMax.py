from utilities import scanners, formatData, minMaxScaling
from utilities import InfluxDBClient, AnimatedScatter
from models import MLP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict():
    data = client.retrieveData(seconds=20, beacon="0117C55D14E4")
    data = formatData(data)
    data[scanners] = minMaxScaling(data[scanners])
    data = data.rolling(15, min_periods=1).mean().reset_index()
    data = data.ffill().fillna(0)
    pred = model.predict(data[scanners].values)
    return np.expand_dims(pred.mean(axis=0), axis=0)

model = MLPRegressor(size='small')
model.load('../Models/Small_MLP_Regression_Rolling_FFill_MinMax.h5')
client = InfluxDBClient()

ani = AnimatedScatter()
ani.setPrediction(predict)
plt.show()