from keras.models import Sequential
from keras.layers import Dense

# define classification model
def classification_model(n_cols):
    # create model
    model = Sequential()
    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model