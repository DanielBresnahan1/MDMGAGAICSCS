# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:14:11 2022

@author: Danie
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:15:17 2022

@author: Danie
"""

import numpy as np
import os
from batches_iterator import BatchesIterator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.layers.convolutional import Conv2D as Convolution2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics

model = Sequential()

valid_lesion_dir = "Z:\\Coding\\Dataset\\Validation\\Positive"
valid_no_lesion_dir = "Z:\\Coding\\Dataset\\Validation\\Negative"
train_lesion_dir = "Z:\\Coding\\Dataset\\Train\\Positive"
train_no_lesion_dir = "Z:\\Coding\\Dataset\\Train\\Negative"

train_batch_size = 40000
valid_batch_size = 50000

train_batches = BatchesIterator(train_batch_size,train_no_lesion_dir,train_lesion_dir)
valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir)
X_valid, Y_valid = next(valid_batches)



model.add(BatchNormalization(axis=1, input_shape=(3,224,224)))
model.add(Convolution2D(16,3,3, data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Convolution2D(8,3,3, data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,3,3, data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.summary()

file_name = 'network_T_1'
saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

cllbcks= [saveWeigts]



for mini_epoch in range(20):
	print("Epoch ", mini_epoch)

	X_train, Y_train = next(train_batches)

	model.fit(X_train, Y_train, batch_size=32, callbacks = cllbcks,
			validation_data=(X_valid,Y_valid), verbose=1)