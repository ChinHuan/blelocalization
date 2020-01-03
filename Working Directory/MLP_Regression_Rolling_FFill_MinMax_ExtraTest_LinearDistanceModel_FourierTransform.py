from utilities import scanners, format_data, min_max_scaling, read_ble
from utilities import InfluxDBClient, AnimatedScatter
from models import MLPRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy import signal

def reg_impute(data):
    for s in scanners:
        dx = coor[0] - ble[s]['X']
        dy = coor[1] - ble[s]['Y']
        d = np.sqrt(np.square(dy) + np.square(dx))
        reg = joblib.load('../Models/linear_distance_models/{}.joblib'.format(s))
        ss = pd.Series(reg.predict(d.reshape((-1, 1)))).repeat(data.shape[0]).reset_index(drop=True)
        data[s].fillna(ss, inplace=True)

def lfilter(xn):
    if isinstance(xn, pd.Series):
        xn = xn.values
    b, a = signal.butter(5, 0.1)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
    return z

def fn(sub_df):
    # for s in scanners:
    #     sub_df[s] = lfilter(sub_df[s])
    sub_df.update(sub_df.mean())
    return sub_df

ble = read_ble('../Data/ble.csv')
coor = np.array([20, 10])

def predict():
    global coor
    data = client.retrieveData(seconds=5, beacon="0117C55D14E4")
    data = format_data(data)
    data[scanners] = min_max_scaling(data[scanners])
    data.reset_index(drop=True, inplace=True)
    reg_impute(data)
    fn(data)
    coor = model.predict(data[scanners].values).mean(axis=0)
    return np.expand_dims(coor, axis=0)

model = MLPRegressor(model_type='special')
model.model.fit(np.array([[0] * 17]), np.array([[0, 0]]))
model.model.load_weights('../Models/MLP_Regression_Rolling_FFill_MinMax_ExtraTest_LinearDistanceModel_FourierTransform.h5')
client = InfluxDBClient()

ani = AnimatedScatter(predict)
plt.show()
