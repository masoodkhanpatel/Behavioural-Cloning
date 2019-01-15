import csv
import cv2
import numpy as np

# Empty list for storing the data
data = [] 

# Reading the log file, of the recorded data
with open('./data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None) # Ignoring a row, so as to ignore header incase it is present
    for line in reader:
        data.append(line) 

import sklearn        
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Splitting the training and validation set using scikit-learn train_test_split function
train_samples, validation_samples = train_test_split(data, test_size=0.20)

# Generator function
def generator(data, batch_size=32):
    num_samples = len(data)
    while 1: 
        shuffle(data)
        for offset in range(0, num_samples, batch_size):
            samples = data[offset: offset + batch_size]

            images = []
            measurements = []
            correction = 0.2 # Parameter to tune to adjust steering measurements for the side camera images
            for sample in samples:
                    for i in range(0,3): #images corresponding to each camera
                        
                        name = './data/IMG/' + sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) #BGR to RGB conversion
                        measurement = float(sample[3]) # Fetching the steering measurement
                        images.append(center_image) # Adding the image
                        images.append(cv2.flip(center_image, 1)) # Adding the flipped image, flipping using opencv flip()

                        # Taking the steering measurements
                        # i=0 --> Center camera image
                        # i=1 --> Left camera image
                        # i=2 --> Right camera image

                        # Adding the steering measurements of image
                        if(i==0):
                            measurements.append(measurement)
                        elif(i==1):
                            measurements.append(measurement + correction) 
                        elif(i==2):
                            measurements.append(measurement - correction)
                        
                        # Adding the steering measurements of flipped image
                        if(i==0):
                            measurements.append(-(measurement))
                        elif(i==1):
                            measurements.append(-(measurement + correction))
                        elif(i==2):
                            measurements.append(-(measurement - correction)) 

            # Converting to numpy arrays
            X_train = np.array(images)
            y_train = np.array(measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train) # yield instead of return for the generator

training_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))   # Setting up Lambda layer and normalizing the images
model.add(Cropping2D(cropping = ((70,25),(0,0))))                           # Cropping the image using Keras Cropping2D Layer
model.add(Conv2D(24, (5, 5), strides=(2, 2)))                               # Convolutional feature map with 24 filters, kernel(5x5) and stride(2x2) 
model.add(Activation('elu'))                                                # Exponential Linear Unit(elu) Activation layer
model.add(Conv2D(36, (5, 5), strides=(2, 2)))                               # Convolutional feature map with 36 filters, kernel(5x5) and stride(2x2) 
model.add(Activation('elu'))                                                # Exponential Linear Unit(elu) Activation layer
model.add(Conv2D(48, (5, 5), strides=(2, 2)))                               # Convolutional feature map with 48 filters, kernel(5x5) and stride(2x2) 
model.add(Activation('elu'))                                                # Exponential Linear Unit(elu) Activation layer
model.add(Conv2D(64, (3, 3)))                                               # Convolutional feature map with 64 filters, kernel(3x3) and stride(1x1)
model.add(Activation('elu'))                                                # Exponential Linear Unit(elu) Activation layer
model.add(Conv2D(64, (3, 3)))                                               # Convolutional feature map with 64 filters, kernel(3x3) and stride(1x1)
model.add(Activation('elu'))                                                # Exponential Linear Unit(elu) Activation layer
model.add(Flatten())                                                        # Flattening layer
model.add(Dense(100))                                                       # Fully connected layer
model.add(Activation('elu'))                                                # Exponential Linear Unit(elu) Activation layer
model.add(Dropout(0.25))                                                    # Dropout layer
model.add(Dense(50))                                                        # Fully connected layer
model.add(Activation('elu'))                                                # Exponential Linear Unit(elu) Activation layer
model.add(Dense(10))                                                        # Fully connected layer
model.add(Activation('elu'))                                                # Exponential Linear Unit(elu) Activation layer
model.add(Dense(1))                                                         # Output

model.compile(loss='mse',optimizer='adam')      # Mean squared error loss function to minize the loss and Adam optimizer

history_object = model.fit_generator(training_generator, 
                steps_per_epoch = int(len(train_samples) / 32), 
                validation_data = validation_generator, 
                validation_steps = int(len(validation_samples) / 32),
                epochs=3)

# Saving the model
model.save('model.h5')

# Summary display
model.summary()

import matplotlib.pyplot as plt

# Plotting the training and validation loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()