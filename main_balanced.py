import our_own_cnn
import our_own_cnn_2
import our_own_cnn_3
import load_balanced_data
import load_dataset_simulator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import CSVLogger

import numpy as np
import math
import matplotlib.pyplot as plt



def main():

    print("Creating model...")
    our_own_cnn_3.our_own_cnn_3()

    print("Loading model...")
    model = load_model("our_own_cnn_3")

    csv_logger = CSVLogger('log_friday.csv', append=True, separator=';')
    checkpoint = ModelCheckpoint('curr_best_model.h5', monitor='val_loss',verbose=0,save_best_only=True, mode='auto') #Saved_models

    np_images = np.load("np_images_balanced.npy")
    np_steering = np.load("np_steering_angles_log.npy")
    
    np_val_images = np.load("np_val_images_balanced.npy")
    np_val_steering = np.load("np_val_steering_angles_log.npy")
    
    history = model.fit(x=np_images, y=np_steering, epochs=200, batch_size=10, callbacks=[checkpoint, csv_logger], validation_data=(np_val_images, np_val_steering)) 

    print("Saving the model...")
    model.save("our_own_cnn_3_trained.h5")


    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
