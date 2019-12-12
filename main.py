import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

days_before = 180
days_ahead = 10
num_period = 20
lstm_model = 'model/lstm_model.h5'
gru_model = 'model/gru_model.h5'

df = pd.read_csv('UNVRfix.csv')
#set date as index
#df['date'] = pd.to_datetime(df['date'])
x_as_date = df['date']
df.set_index('date', inplace = True)

#we're using only close price
df = df['close']

#Data normalization
array = df.values.reshape(df.shape[0],1)
scl = MinMaxScaler()
array = scl.fit_transform(array)

#split in train and test

division = len(array) - num_period*days_ahead
leftover = division%days_ahead+1

test = array[division-days_before:]
train = array[leftover:division]

def processData(data, days_before, days_ahead, jump=1):
   x,y = [],[]
   for i in range(0,len(data)-days_before-days_ahead+1, jump):
      x.append(data[i:(i+days_before)])
      y.append(data[(i+days_before):(i+days_before+days_ahead)])
   return np.array(x), np.array(y)

def predict_lstm(lstm_model, xtest, ytest):
   lstm_model = load_model(lstm_model)
   xtest_lstm = lstm_model.predict(xtest)
   xtest_lstm = xtest_lstm.ravel()
   ytest = np.reshape(ytest, (200,1))
   xtest_lstm = np.reshape(xtest_lstm, (200,1))
   lstm_mse = metrics.mean_squared_error(ytest, xtest_lstm)
   return xtest_lstm, lstm_mse

def predict_gru(gru_model, xtest, ytest):
   gru_model = load_model(gru_model)
   xtest_gru = gru_model.predict(xtest)
   xtest_gru = xtest_gru.ravel()
   ytest = np.reshape(ytest, (200,1))
   xtest_lstm = np.reshape(xtest_gru, (200,1))
   gru_mse = metrics.mean_squared_error(ytest, xtest_gru)
   return xtest_gru, gru_mse
   
xtrain, ytrain = processData(train, days_before, days_ahead, days_ahead)
xtest, ytest = processData(test, days_before, days_ahead, days_ahead)

xtest_lstm, lstm_mse = predict_lstm(lstm_model, xtest, ytest)
xtest_gru, gru_mse = predict_gru(gru_model, xtest, ytest)
print('LSTM\nMSE: %f \nGRU\nMSE: %f' %(lstm_mse, gru_mse))

y = np.concatenate((ytrain, ytest), axis=0)

#Data Tested
plt.plot([x for x in range(1641,1841)], scl.inverse_transform(xtest_lstm.reshape(-1,1)), color='r', label='LSTM')
   
#Data Tested
plt.plot([x for x in range(1641,1841)], scl.inverse_transform(xtest_gru.reshape(-1,1)), color='g', label='GRU')  

#Data used
plt.plot(scl.inverse_transform(array), color='b', label='Target')

plt.title('Stock Prediction with LSTM and GRU')
plt.xlabel('Date')
plt.ylabel('Close Price')

plt.legend(loc='best')
plt.savefig('model/FigurePredicted.png')
plt.show()
