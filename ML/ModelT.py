# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:07:30 2022

@author: Danie
"""

import numpy as np
import os
from test_iterator import TestIterator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

def generate_model():
    
    model = Sequential()
    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(8,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(16,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    
    model.add(Conv2D(8,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(8,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    
    model.add(Conv2D(8,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(8,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(8,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    model.add(Conv2D(8,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    
    model.add(BatchNormalization())
    model.add(Dense(1))
    
    opti=Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=[
                      tf.keras.metrics.binary_crossentropy,
                      'accuracy'])
    
    model.build((None, 1, 193, 126))
    model.summary()
    
    return model

if __name__=="__main__":
    
    
    base_dir = "E:\\Coding\\Dataset"
    train_image_dir = os.path.join(base_dir, "Test_Map_Heat")
    val_image_dir = os.path.join(base_dir, "Train_Map_Heat")
    batch_size = 32
    train_csvf = "test_labels.csv"
    val_csvf = "train_map.csv"
    
    epochsN = 128
    
    train_csv_dir = os.path.join(base_dir, train_csvf)
    val_csv_dir = os.path.join(base_dir, val_csvf)
    
    train_batches = TestIterator(batch_size, train_csv_dir, train_image_dir)
    val_batches = TestIterator(batch_size, val_csv_dir, val_image_dir)
    
    file_name = 'network_T_1'
    saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "log"), histogram_freq=1)
    
    cllbcks= [saveWeigts, tensorboard_callback]
    
    model_path = os.path.join(os.getcwd(),file_name+'_best_weights.h5')
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(file_name+'_best_weights.h5')
    else:
        model = generate_model()
    
    model.fit(x=train_batches, epochs=epochsN, steps_per_epoch=len(train_batches), callbacks = cllbcks,
              validation_data=val_batches, verbose=1)
    
    model.save(model_path, include_optimizer=False)
