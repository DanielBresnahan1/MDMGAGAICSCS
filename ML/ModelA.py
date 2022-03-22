# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:38:25 2022

@author: Danie
"""


import numpy as np
import os
from batches_iterator import BatchesIterator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras


def generate_model():
    model = Sequential()
    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, 3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(256,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(256,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    
    model.add(BatchNormalization())
    model.add(Dense(1))
    
    model.add(Activation('sigmoid'))
    opti=Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=[
                      tf.keras.metrics.binary_crossentropy,
                      'accuracy'])
    
    model.build((None, 3, 224, 224))
    model.summary()
    
    return model


if __name__=="__main__":
    
    valid_lesion_dir = "E:\\Coding\\Dataset\\Validation\\Positive"
    valid_no_lesion_dir = "E:\\Coding\\Dataset\\Validation\\Negative"
    train_lesion_dir = "E:\\Coding\\Dataset\\Train\\Positive"
    train_no_lesion_dir = "E:\\Coding\\Dataset\\Train\\Negative"
    
    train_batch_size = 1000
    valid_batch_size = 1100
    
    train_batches = BatchesIterator(train_batch_size,train_no_lesion_dir,train_lesion_dir)
    valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir)
    
    file_name = 'network_A_1'
    saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)
    
    cllbcks= [saveWeigts]
    
    if os.path.exists(os.path.join(os.getcwd(),'network_A_1_best_weights.h5')):
        model = keras.models.load_model('network_A_1_best_weights.h5')
    else:
        model = generate_model()
        
    for mini_epoch in range(1000):
        print("Epoch ", mini_epoch)
    
        X_train, Y_train, _ = next(train_batches)
        X_valid, Y_valid, _ = next(valid_batches)
    
        model.fit(X_train, Y_train, callbacks = cllbcks,
                  validation_data=(X_valid,Y_valid), verbose=1)