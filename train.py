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
from model import lstm_model,gru_model

days_before = 180
days_ahead = 10
num_period = 20
EPOCHS = 50

df = pd.read_csv('UNVRfix.csv')
#set date as index
df['date'] = pd.to_datetime(df['date'])
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

x,Y = processData(train, days_before, days_ahead)
y = np.array([list(a.ravel()) for a in Y])

print('Creating LSTM model')
lstm_model = lstm_model(days_before, days_ahead)

print('Creating GRU model')
gru_model = gru_model(days_before, days_ahead)

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size = 0.2, random_state=42)

def train_lstm(lstm_model,x_train, x_validate, y_train, y_validate, EPOCHS):
   print('Train on LSTM')
   history = lstm_model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_validate, y_validate), shuffle=True, batch_size=512, verbose=2)
   save_model_lstm = 'model/lstm_model_3.h5'
   lstm_model.save(save_model_lstm)
   print("Model saved to {}".format(save_model_lstm))
   plt.plot(history.history['loss'], label='loss')
   plt.plot(history.history['val_loss'], label='val_loss')
   plt.legend(loc='best')
   plt.savefig('model/lossfig_lstm.png')
   state =  'Training LSTM Done'
   plt.close()
   return state
   
def train_gru(gru_model,x_train, x_validate, y_train, y_validate, EPOCHS):
   print('Train on LSTM')
   history = gru_model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_validate, y_validate), shuffle=True, batch_size=512, verbose=2)
   save_model_gru = 'model/gru_model.h5'
   gru_model.save(save_model_gru)
   print("Model saved to {}".format(save_model_gru))
   plt.plot(history.history['loss'], label='loss')
   plt.plot(history.history['val_loss'], label='val_loss')
   plt.legend(loc='best')
   plt.savefig('model/lossfig_gru.png')
   state = 'Training GRU Done'
   return state
   
state = train_lstm(lstm_model,x_train, x_validate, y_train, y_validate, EPOCHS)
print(state)
state = train_gru(lstm_model,x_train, x_validate, y_train, y_validate, EPOCHS)
print(state)