import our_own_cnn
import our_own_cnn_2
import our_own_cnn_3
import our_own_cnn_4
import load_dataset_simulator
import load_balanced_dataset
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import CSVLogger

import numpy as np
import math
import matplotlib.pyplot as plt



def main():

    print("Creating model...")
    our_own_cnn_4.our_own_cnn_4()

    print("Loading model...")
    model = load_model("our_own_cnn_4_monday.h5")
    #model = load_model("saturday_best_model.h5")
    model.summary()

    csv_logger = CSVLogger('log_monday.csv', append=True, separator=';')
    checkpoint = ModelCheckpoint('curr_best_model_monday.h5', monitor='val_loss',verbose=0,save_best_only=True, mode='auto') #Saved_models

    print("Loading training dataset...")
    np_images_train = np.load("np_images_balanced.npy")
    np_steering_train = np.load("np_steering_angles_log.npy")
    
    print("Loading validation dataset...")
    np_val_images_train = np.load("np_val_images_balanced.npy")
    np_val_steering_train = np.load("np_val_steering_angles_log.npy")
    
    history = model.fit(x=np_images_train, y=np_steering_train, epochs=200, batch_size=10, callbacks=[checkpoint, csv_logger], validation_data=(np_val_images_train, np_val_steering_train)) 

    print("Saving the model...")
    model.save("our_own_cnn_4_trained_monday.h5")


    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
