from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


def get_model():
    model = Sequential()

    model.add(Convolution2D(20, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'))
    model.add(Dropout(0.25))
    model.add(Convolution2D(20, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model