from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dropout, TimeDistributed
from keras.layers import Dense, BatchNormalization
from keras.layers import Input
from keras.layers import Conv2D, Conv3D
from keras.layers import MaxPooling2D, MaxPooling3D
from keras.utils import layer_utils
from keras.models import load_model
from keras.utils.data_utils import get_file
from keras import backend as K
import h5py


def our_own_cnn_2():

    img_input = Input(shape=(3, 64, 64, 3))

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))(img_input)

    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2,2)))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)


    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))(x)

    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2,2)))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)

    # Block 3

    x = TimeDistributed(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))(x)

    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2,2)))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))(x)

    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2,2)))(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)



    top_model = Flatten()(x)
    top_model = Dropout(0.5)(top_model)

    # Regression part
    fc1 = Dense(256, activation='tanh')(top_model)
    prediction = Dense(1, activation='elu')(fc1)

    model = Model(inputs=img_input, outputs=prediction)

    slow_adam = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=slow_adam, metrics=['accuracy'])
    model.summary()

    model.save("our_own_cnn_2")

    return 0;
