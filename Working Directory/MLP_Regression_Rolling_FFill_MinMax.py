from utilities import scanners, format_data, min_max_scaling
from utilities import InfluxDBClient, AnimatedScatter
from models import MLPRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict():
    data = client.retrieveData(seconds=15, beacon="0117C55D14E4")
    data = format_data(data)
    data[scanners] = min_max_scaling(data[scanners])
    data = data.rolling(30, min_periods=1).mean().reset_index()
    data = data.ffill().fillna(0)
    pred = model.predict(data[scanners].values)
    return np.expand_dims(np.mean(pred, axis=0), axis=0)

model = MLPRegressor(model_type="small")
model.load('../Models/Small_MLP_Regression_Rolling_FFill_MinMax.h5')
client = InfluxDBClient()

ani = AnimatedScatter(predict)
plt.show()