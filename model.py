
import numpy as np
import cv2
import matplotlib.pyplot as plt

import csv

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)



from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
import cv2
import numpy as np
import sklearn

#generator to generate normal,flipped,grayscale images

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                batch_sample[0] = batch_sample[0].replace("\\", "/")
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                
                #center
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                #left
                batch_sample[1] = batch_sample[1].replace("\\", "/")
                lname = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(lname)
                left_angle = float(batch_sample[3])+0.09
                images.append(left_image)
                angles.append(left_angle)

                #right
                batch_sample[2] = batch_sample[2].replace("\\", "/")
                rname = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(rname)
                right_angle = float(batch_sample[3])-0.09
                images.append(right_image)
                angles.append(right_angle)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
               	augmented_measurements.append(measurement)
		
                #grayscale
                tempimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                augmented_images.append(np.stack((tempimg,)*3, 2))
                augmented_measurements.append(measurement)

                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

		#grayscale
                tempimgflipped = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_RGB2GRAY)
                augmented_images.append(np.stack((tempimgflipped,)*3, 2))
                augmented_measurements.append(measurement*-1.0)

            
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# print("aug:", len(X_train), " " , X_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#normalize the image
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,3,3,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(Flatten())
model.add(Dense(160))
model.add(Dense(60))
model.add(Dense(1))

#use adam optimizer
model.compile(loss='mse',optimizer='adam' )

model.fit_generator(train_generator,samples_per_epoch=len(train_samples)*12, validation_data=validation_generator,nb_val_samples=len(validation_samples)*12,nb_epoch=7)

#save model
model.save('model.h5')

