import our_own_cnn
import our_own_cnn_2
import our_own_cnn_3
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

            np_steering_tot = np.concatenate((np_steering_tot, np_steering))

            #np_images_time = np.zeros((int(len(np_images)/3), 3,64,64,3), dtype=float, order='C')
            #np_steering_time = np.zeros((int(len(np_steering)/3),3,1), dtype=float, order='C')

            #for step_index in range(0,len(np_images),3):
            #    for time_index in range(3):
            #       np_images_time[int(step_index/3)][time_index] = np_images[step_index+time_index]
            #        np_steering_time[int(step_index/3)][time_index] = np_steering[step_index+time_index]

            if first_iter:
                first_iter = False
            else:
                model = load_model('curr_best_model.h5')

            #print("--- SHAPE OF np_images: ", np_images.shape)
            #print("--- SHAPE OF np_steering: ", np_steering.shape)

            #print("--- SHAPE OF np_images_time: ", np_images_time.shape)
            #print("--- SHAPE OF np_steering_time: ", np_steering_time.shape)

            print("Training the model...")
            history = model.fit(x=np_images, y=np_steering, epochs=40, batch_size=3, callbacks=[checkpoint, csv_logger], validation_data=(np_val_images, np_val_steering)) #




    print("Saving the model...")
    model.save("our_own_cnn_3_trained.h5")

    print("Saving the steering angles...")
    np.save("np_steering_tot", np_steering_tot)


    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
