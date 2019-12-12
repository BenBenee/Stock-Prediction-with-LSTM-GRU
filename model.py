from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras import optimizers

def lstm_model(days_before, days_ahead):
   lstm_model = Sequential()
   lstm_model.add(LSTM(256, input_shape=(days_before,1), return_sequences=True))
   lstm_model.add(Dropout(0.4))
   lstm_model.add(LSTM(128, input_shape=(256,1),return_sequences=True))
   lstm_model.add(Dropout(0.4))
   lstm_model.add(LSTM(64, input_shape=(128,1), return_sequences=True))
   lstm_model.add(LSTM(64, input_shape=(64,1)))
   lstm_model.add(Dropout(0.2))
   lstm_model.add(Dense(days_ahead, activation='relu'))
   lstm_model.compile(loss='mean_squared_error', optimizer='adam')
   return lstm_model
   
def gru_model(days_before, days_ahead):
   gru_model = Sequential()
   gru_model.add(GRU(256, input_shape=(days_before,1), return_sequences=True))
   gru_model.add(Dropout(0.4))
   gru_model.add(GRU(128, input_shape=(256,1), return_sequences=True))
   gru_model.add(Dropout(0.4))
   gru_model.add(GRU(64, input_shape=(128,1), return_sequences=True))
   gru_model.add(GRU(64, input_shape=(64,1)))
   gru_model.add(Dropout(0.2))
   gru_model.add(Dense(days_ahead, activation='relu'))
   gru_model.compile(loss='mean_squared_error', optimizer='adam')
   return gru_model