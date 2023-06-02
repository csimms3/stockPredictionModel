import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as dates
import datetime as dt

"""
June 1, 2023
Following tutorial at:
https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571
"""




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


"""
LSTM model on TD.csv
"""
stock_data = pd.read_csv("../stock_data/TD.csv", index_col="Date")

target_y = stock_data["Close"]
x_feat = stock_data.iloc[:, 0:3]



"""
Scaling
"""
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

"""
Building LSTM Model
"""

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2]),
              activation="relu",
              return_sequences=True))
lstm.add(Dense(1))
lstm.compile(loss="mean_squared_error", optimizer="adam")
lstm.summary()


"""
Fit model to data
"""
history = lstm.fit(x_train, y_train,
                   epochs=10, batch_size=4,
                   verbose=2, shuffle=False)



"""
Compare against actual value
"""

y_pred = lstm.predict(x_test)

y_pred = [a[1] for a in y_pred]
print(y_pred)
print(y_test)
print(x_test_date)

plt.figure(figsize=(12, 8))
plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(dates.DayLocator(interval=90))
x_dates = x_test_date[1:]

plt.plot(x_dates, y_test, label="Actual")
plt.plot(x_dates, y_pred, label="LSTM Prediction")
plt.xlabel("Time")
plt.ylabel("CAD")
plt.title("LSTM Prediction of TD")
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()