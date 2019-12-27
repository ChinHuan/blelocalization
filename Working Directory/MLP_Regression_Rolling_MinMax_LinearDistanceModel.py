from utilities import scanners, format_data, min_max_scaling, read_ble
from utilities import InfluxDBClient, AnimatedScatter
from models import MLPRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

ble = read_ble('../Data/ble.csv')
coor = np.array([-1, -1])

def reg_impute(data):
    for s in scanners:
        dx = coor[0] - ble[s]['X']
        dy = coor[1] - ble[s]['Y']
        d = np.sqrt(np.square(dy) + np.square(dx))
        reg = joblib.load('../Models/linear_distance_models/{}.joblib'.format(s))
        ss = pd.Series(np.repeat(reg.predict(d.reshape((-1, 1))), data.shape[0]))
        data[s].fillna(ss, inplace=True)

def predict():
    global coor
    data = client.retrieveData(seconds=5, beacon="0117C55D14E4")
    data = format_data(data)
    data[scanners] = min_max_scaling(data[scanners])
    data = data.rolling(15, min_periods=1).mean().reset_index()
    if (coor == np.array([-1, -1])).all():
        data = data.fillna(0)
    else:
        reg_impute(data)
    coor = model.predict(data[scanners].values).mean(axis=0)
    return np.expand_dims(coor, axis=0)

model = MLPRegressor(model_type='special')
model.model.fit(np.array([[0] * 17]), np.array([[0, 0]]))
model.model.load_weights('../Models/MLP_Regression_Rolling_MinMax_LinearDistanceModel.h5')
client = InfluxDBClient()

ani = AnimatedScatter(predict)
plt.show()
