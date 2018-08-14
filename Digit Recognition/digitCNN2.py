# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:51:33 2017

@author: pshrivas
"""
#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Load Data
train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")

train_X = train_set.iloc[:,1:].values.astype("float32")
test_X = test_set.values.astype("float32")
train_Y =  train_set.iloc[:,0].values
                         
#Reshape data
train_X = train_X.reshape(train_X.shape[0],28,28,1)
test_X = test_X.reshape(test_X.shape[0],28,28,1)

from keras.utils.np_utils import to_categorical
train_Y = to_categorical(train_Y)
num_class = train_Y.shape[1]

#CNN
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape = (28,28,1),activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=128,activation="relu"))

classifier.add(Dense(units=10,activation="softmax"))

classifier.compile(optimizer="sgd",loss ="categorical_crossentropy", metrics=["accuracy"])
classifier.summary()

from keras.preprocessing.image import ImageDataGenerator
traingen = ImageDataGenerator(rescale=1./255)
testgen = ImageDataGenerator(rescale=1./255)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_X, train_Y, test_size=0.10, random_state=42)
batches = traingen.flow(X_train, y_train, batch_size=64)
val_batches=testgen.flow(X_val, y_val, batch_size=64)
history = classifier.fit_generator(batches,batches.n,epochs=5,validation_data=val_batches,validation_steps=val_batches.n)
 

pred=classifier.predict_classes(test_X)

digit = pd.DataFrame({"ImageId":list(range(1,len(test_set)+1)),"Label":pred})

digit.to_csv("digitCNN3Layer.csv",index=False)#Accuracy 98.985
 
                         
             #Accuracy with one CNN layer     97.785       
                         