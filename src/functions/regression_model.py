
from keras.models import Sequential
from keras.layers import Dense

def regression_model(n_cols):
    #create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))

    model.add(Dense(1))

    #compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model