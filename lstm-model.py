import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
LSTM model libraries
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping

"""
Data Preprocessing
"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit


stock_data = pd.read_csv("stock_data/TD.csv", index_col="Date")

target_y = stock_data["Close"]
x_feat = stock_data.iloc[:, 0:3]

#scaling the data
sc = StandardScaler()
x_ft = sc.fit_transform(x_feat.values)
stock_data_scaled = pd.DataFrame(columns=x_feat.columns,
                    data = x_ft,
                    index = x_feat.index)

def lstm_split(data, n_steps):
    x,y = [],[]
    for i in range (len(data) - n_steps+1):
        x.append(data[i:i+n_steps, :-1])
        y.append(data[i + n_steps-1 , -1])
    return np.array(x), np.array(y)

"""
Creating Training and Testing splits for the data
"""

x1,y1 = lstm_split(stock_data_scaled.values, n_steps=2)

train_split_ratio = 0.8
split_idx = int(np.ceil(len(x1)*train_split_ratio))
date_index = stock_data_scaled.index

x_train, x_test = x1[:split_idx], x1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
x_train_date, x_test_date = date_index[:split_idx], date_index[split_idx:]

print(x1.shape, x_train.shape, x_test.shape, y_test.shape)

