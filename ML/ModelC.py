# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:17:15 2022

@author: Danie
"""

import numpy as np
import os
from batches_iterator import BatchesIterator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


def generate_model():
    
    model = Sequential()

    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(32,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.01))
    
    model.add(Conv2D(64,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.01))
    
    model.add(Conv2D(128,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(128,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.01))
    
    model.add(Conv2D(256,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(256,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.01))
    
    model.add(Conv2D(16,3, kernel_regularizer=l2(.0001)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    
    model.add(Flatten())
    
    model.add(Dense(1))
    
    model.add(Activation('sigmoid'))
    
    opti = Adam(lr=0.0001)
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
    
    train_batch_size = 32
    valid_batch_size = 32
    
    train_batches = BatchesIterator(train_batch_size,train_no_lesion_dir,train_lesion_dir)
    valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir)
    
    file_name = 'network_C_1'
    saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)
    cllbcks= [saveWeigts]
    
    if os.path.exists(os.path.join(os.getcwd(),file_name+'_best_weights.h5')):
        model = tf.keras.models.load_model(file_name+'_best_weights.h5')
    else:
        model = generate_model()
    
    model.fit(x=train_batches, epochs=12, steps_per_epoch=len(train_batches), callbacks = cllbcks,
              validation_data=valid_batches, verbose=1)