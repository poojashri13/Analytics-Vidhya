# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:40:03 2017

@author: pshrivas
"""

#98.71 % accuracy
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load dataset
train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")
train_X = (train_set.iloc[:,1:].values).astype("float32")
train_y = train_set.iloc[:,0].values
test_X = test_set.values.astype("float32")


#Reshape data

train_X = train_X.reshape(train_X.shape[0],28,28,1)
test_X = test_X.reshape(test_X.shape[0],28,28,1)


#Preprocessing
mean = train_X.mean().astype(np.float32)
sd = train_X.std().astype(np.float32)

def standardize(x):
    return (x-mean)/sd

#Encoding 
from keras.utils.np_utils import to_categorical
train_y = to_categorical(train_y)
num_class = train_y.shape[1]


#CNN
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Lambda
classifier = Sequential()
classifier.add(Lambda(standardize,input_shape = (28,28,1)))
classifier.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))

classifier.summary()
classifier.compile(optimizer = "adam",loss= "categorical_crossentropy",metrics =["accuracy"])



classifier.fit(train_X,train_y,batch_size=128,epochs=25,validation_split=0.2)
pred=classifier.predict_classes(test_X)
digit = pd.DataFrame({"ImageId":list(range(1,len(test_set)+1)),"Label":pred})

digit.to_csv("digitCNN.csv",index=False)