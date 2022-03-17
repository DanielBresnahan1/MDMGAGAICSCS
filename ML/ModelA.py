# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:38:25 2022

@author: Danie
"""


import numpy as np
import os
from batches_iterator_half_no_regular_no_lesion import BatchesIterator
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


from keras.callbacks import ModelCheckpoint

model = Sequential()

valid_lesion_dir = "Z:\\Coding\\Dataset\\Validation\\Positive"
valid_no_lesion_dir = "Z:\\Coding\\Dataset\\Validation\\Negative"
train_lesion_dir = "Z:\\Coding\\Dataset\\Train\\Positive"
train_no_lesion_dir = "Z:\\Coding\\Dataset\\Train\\Negative"


train_batch_size = 40000
valid_batch_size = 50000

train_batches = BatchesIterator(train_batch_size,train_no_lesion_dir,train_lesion_dir)
valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir)
X_valid, Y_valid = valid_batches.next()

model.add(BatchNormalization(mode=0, axis=1, input_shape=(3,224,224)))
model.add(Convolution2D(64,3,3,
			 init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3,init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(256,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(256,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(1,init='glorot_normal'))

model.add(Activation('sigmoid'))
opti=Adam(lr=0.000001)
model.compile(loss='binary_crossentropy', optimizer=opti, class_mode='binary')
model.summary()

file_name = 'network_A_1'
saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='val_acc', verbose=1, save_best_only=True)

cllbcks= [saveWeigts]



for mini_epoch in range(1000):
	print("Epoch ", mini_epoch)

	X_train, Y_train = train_batches.next()

	model.fit(X_train, Y_train, batch_size=64, nb_epoch=1, callbacks = cllbcks,
			validation_data=(X_valid,Y_valid),show_accuracy=True, verbose=1)