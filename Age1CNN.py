## -*- coding: utf-8 -*-
#"""
#Created on Tue Sep 12 15:18:25 2017
#
#@author: pshrivas
#"""
#
#
#import os
#import random
#
#import pandas as pd
#from scipy.misc import imread
#import numpy as np
#import matplotlib.pyplot as plt
#train = pd.read_csv( 'train.csv')
#test = pd.read_csv('test.csv')
#
#import keras
#from sklearn.preprocessing import LabelEncoder
#
#lb = LabelEncoder()
#train_y = lb.fit_transform(train.Class)
#train_y = keras.utils.np_utils.to_categorical(train_y)
#
#from scipy.misc import imresize
#temp = []
#for img_name in train.ID:
#    img_path = os.path.join( 'Train', img_name)
#    img = imread(img_path)
#    img = imresize(img, (32, 32))
#    img = img.astype('float32') # this will help us in later stage
#    temp.append(img)
#
#train_x = np.stack(temp)
#
#
#temp = []
#for img_name in test.ID:
#    img_path = os.path.join('Test', img_name)
#    img = imread(img_path)
#    img = imresize(img, (32, 32))
#    temp.append(img.astype('float32'))
#
#test_x = np.stack(temp)
#
#
#
#from keras.models import Sequential
#from keras.layers import Convolution2D
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#from keras.layers import Dense
#
## Initialising the CNN
#classifier = Sequential()
#
## Step 1 - Convolution
#classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
#
## Step 2 - Pooling
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
## Adding a second convolutional layer
#classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
## Step 3 - Flattening
#classifier.add(Flatten())
#
## Step 4 - Full connection
#classifier.add(Dense(output_dim = 128, activation = 'relu'))
#classifier.add(Dense(output_dim = 3, activation = 'softmax'))
#
## Compiling the CNN
#classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#classifier.summary()
#
#from keras.preprocessing.image import ImageDataGenerator
#traingen = ImageDataGenerator(rescale=1./255)
#testgen = ImageDataGenerator(rescale=1./255)
#
#from sklearn.model_selection import train_test_split
#X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.10, random_state=42)
#batches = traingen.flow(X_train, y_train, batch_size=64)
#val_batches=testgen.flow(X_val, y_val, batch_size=64)
#history = classifier.fit_generator(batches,batches.n,epochs=5,validation_data=val_batches,validation_steps=val_batches.n)
#
#
#
#classifier.fit_generator(batches,batches.n, epochs=25)
##77.58 with batch_size = 500 and epochs =25
#
##86.71 with batch_size = 128 epochs = 25
#pred = classifier.predict_classes(test_x)
#pred = lb.inverse_transform(pred)
#
#test['Class'] = pred
#test.to_csv("sub03.csv",index=False)
#
#
#i = random.choice(train.index)
#img_name = train.ID[i]
#
#img = imread(os.path.join('Train', img_name)).astype('float32')
#plt.imshow(imresize(img, (128, 128)))
#pred = classifier.predict_classes(train_x)
#print('Original:', train.Class[i], 'Predicted:', lb.inverse_transform(pred[i]))