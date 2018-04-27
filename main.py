import our_own_cnn
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
    our_own_cnn.our_own_cnn_1()

    print("Loading model...")
    model = load_model("our_own_cnn_1")

    csv_logger = CSVLogger('log_friday.csv', append=True, separator=';')
    checkpoint = ModelCheckpoint('curr_best_model.h5', monitor='val_loss',verbose=0,save_best_only=True, mode='auto') #Saved_models

    np_steering_tot = np.zeros((1))

    np_val_images, np_val_steering = load_dataset_simulator.load_dataset("center","test")
    np_val_images_new, np_val_steering_new = load_dataset_simulator.load_dataset("right","test")
    np_val_images = np.concatenate((np_val_images, np_val_images_new))
    np_val_steering = np.concatenate((np_val_steering, np_val_steering_new))
    np_val_images_new, np_val_steering_new = load_dataset_simulator.load_dataset("left","test")
    np_val_images = np.concatenate((np_val_images, np_val_images_new))
    np_val_steering = np.concatenate((np_val_steering, np_val_steering_new))

    print("Length of val images: ", len(np_val_images))

    first_iter = True

    print("Loading datasets...")
    for dataset in ["LEFT", "RIGHT", "mond", "mond2", "mond3", "mond4", "track1_rewind", "track2"]:
        for camera_angle in ["center", "right", "left"]:

            print("Currently loading dataset: ", dataset, ", angle: ", camera_angle, ".")
            np_images, np_steering = load_dataset_simulator.load_dataset(camera_angle,dataset)

            if first_iter:
                first_iter = False
            else:
                model = load_model('curr_best_model.h5')

            print("Training the model...")
            history = model.fit(x=np_images, y=np_steering, epochs=40, batch_size=3, callbacks=[checkpoint, csv_logger], validation_data=(np_val_images, np_val_steering))

            np_steering_tot = np.concatenate((np_steering_tot, np_steering))


    print("Saving the model...")
    model.save("our_own_cnn_1_trained.h5")

    print("Saving the steering angles...")
    np.save("np_steering_tot", np_steering_tot)


    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
