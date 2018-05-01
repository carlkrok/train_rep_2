from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dropout, TimeDistributed
from keras.layers import Dense, BatchNormalization
from keras.layers import Input
from keras.layers import Conv2D, Conv3D, LSTM
from keras.layers import MaxPooling2D, MaxPooling3D
from keras.utils import layer_utils
from keras.models import load_model
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras import backend as K
import h5py


def cnn_lstm():

    img_input = Input(shape=(64, 64, 3))

    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(img_input)

    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)


    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)

    # Block 3

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    x = MaxPooling2D((2, 2), strides=(2,2))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)


    top_model = Flatten()(x)
    
    lstm_layers = LSTM(kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), activity_regularizer=l2(0.01), dropout=0.25, recurrent_dropout=0.25 return_sequences=True)(top_model)
    lstm_layers = LSTM(kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01), activity_regularizer=l2(0.01), dropout=0.25, recurrent_dropout=0.25 return_sequences=False)(lstm_layers)

    
    # Regression part
    fc1 = Dense(50, activation='tanh')(lstm_layers)
    fc1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(fc1)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(10, activation='tanh')(fc1)
    
    prediction = Dense(1)(fc2)

    model = Model(inputs=img_input, outputs=prediction)

    slow_adam = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=slow_adam, metrics=['mse'])
    model.summary()

    model.save("cnn_lstm.h5")

    return 0;
