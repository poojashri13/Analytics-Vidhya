# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:00:08 2017

@author: pshrivas
"""

import os
import random

import pandas as pd
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv( 'train.csv')
test = pd.read_csv('test.csv')


from scipy.misc import imresize
temp = []
for img_name in train.ID:
    img_path = os.path.join( 'Train', img_name)
    img = imread(img_path)
    img = imresize(img, (32, 32))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

train_x = np.stack(temp)


temp = []
for img_name in test.ID:
    img_path = os.path.join('Test', img_name)
    img = imread(img_path)
    img = imresize(img, (32, 32))
    temp.append(img.astype('float32'))

test_x = np.stack(temp)


train_x = train_x / 255.
test_x = test_x / 255.

import keras
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)

input_num_units = (32, 32, 3)
hidden_num_units = 500
output_num_units = 3

epochs = 5
batch_size = 128

from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer

model = Sequential([
  InputLayer(input_shape=input_num_units),
  Flatten(),
  Dense(units=hidden_num_units, activation='relu'),
  Dense(units=output_num_units, activation='softmax'),
])

model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1, validation_split=0.2)


pred = model.predict_classes(test_x)
pred = lb.inverse_transform(pred)

test['Class'] = pred
test.to_csv("sub01.csv",index=False)
type(test)

i = random.choice(train.index)
img_name = train.ID[i]

img = imread(os.path.join('Train', img_name)).astype('float32')
plt.imshow(imresize(img, (128, 128)))
pred = model.predict_classes(train_x)
print('Original:', train.Class[i], 'Predicted:', lb.inverse_transform(pred[i]))
