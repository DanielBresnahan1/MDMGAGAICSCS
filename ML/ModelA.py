# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:38:25 2022

@author: Danie

First of 3 CNN's for classification, takes the form of a keras Sequential CNN. 
Also contains functionality to train the model using a train val split specificied in the relevant csv files

To train the model please set the following variables in __main__
valid_lesion_dir = "Path to folder containing positive patches for validation"
valid_no_lesion_dir = "Path to folder containing negative patches for validation"
train_lesion_dir = "Path to folder containing positive patches for training"
train_no_lesion_dir = "Path to folder containing negative patches for training"

Model Structure: 
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     batch_normalization (BatchN  (None, 3, 224, 224)      12        
     ormalization)                                                   
                                                                     
     conv2d (Conv2D)             (None, 64, 222, 222)      1792      
                                                                     
     batch_normalization_1 (Batc  (None, 64, 222, 222)     256       
     hNormalization)                                                 
                                                                     
     activation (Activation)     (None, 64, 222, 222)      0         
                                                                     
     conv2d_1 (Conv2D)           (None, 64, 220, 220)      36928     
                                                                     
     batch_normalization_2 (Batc  (None, 64, 220, 220)     256       
     hNormalization)                                                 
                                                                     
     activation_1 (Activation)   (None, 64, 220, 220)      0         
                                                                     
     max_pooling2d (MaxPooling2D  (None, 64, 110, 110)     0         
     )                                                               
                                                                     
     conv2d_2 (Conv2D)           (None, 64, 108, 108)      36928     
                                                                     
     batch_normalization_3 (Batc  (None, 64, 108, 108)     256       
     hNormalization)                                                 
                                                                     
     activation_2 (Activation)   (None, 64, 108, 108)      0         
                                                                     
     conv2d_3 (Conv2D)           (None, 64, 106, 106)      36928     
                                                                     
     batch_normalization_4 (Batc  (None, 64, 106, 106)     256       
     hNormalization)                                                 
                                                                     
     activation_3 (Activation)   (None, 64, 106, 106)      0         
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 64, 53, 53)       0         
     2D)                                                             
                                                                     
     conv2d_4 (Conv2D)           (None, 128, 51, 51)       73856     
                                                                     
     batch_normalization_5 (Batc  (None, 128, 51, 51)      512       
     hNormalization)                                                 
                                                                     
     activation_4 (Activation)   (None, 128, 51, 51)       0         
                                                                     
     conv2d_5 (Conv2D)           (None, 128, 49, 49)       147584    
                                                                     
     batch_normalization_6 (Batc  (None, 128, 49, 49)      512       
     hNormalization)                                                 
                                                                     
     activation_5 (Activation)   (None, 128, 49, 49)       0         
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 128, 24, 24)      0         
     2D)                                                             
                                                                     
     conv2d_6 (Conv2D)           (None, 256, 22, 22)       295168    
                                                                     
     batch_normalization_7 (Batc  (None, 256, 22, 22)      1024      
     hNormalization)                                                 
                                                                     
     activation_6 (Activation)   (None, 256, 22, 22)       0         
                                                                     
     conv2d_7 (Conv2D)           (None, 256, 20, 20)       590080    
                                                                     
     batch_normalization_8 (Batc  (None, 256, 20, 20)      1024      
     hNormalization)                                                 
                                                                     
     activation_7 (Activation)   (None, 256, 20, 20)       0         
                                                                     
     conv2d_8 (Conv2D)           (None, 128, 18, 18)       295040    
                                                                     
     batch_normalization_9 (Batc  (None, 128, 18, 18)      512       
     hNormalization)                                                 
                                                                     
     activation_8 (Activation)   (None, 128, 18, 18)       0         
                                                                     
     flatten (Flatten)           (None, 41472)             0         
                                                                     
     batch_normalization_10 (Bat  (None, 41472)            165888    
     chNormalization)                                                
                                                                     
     dense (Dense)               (None, 1)                 41473     
                                                                     
     activation_9 (Activation)   (None, 1)                 0         
                                                                     
    =================================================================
    Total params: 1,726,285
    Trainable params: 1,641,031
    Non-trainable params: 85,254
    _________________________________________________________________
"""


import numpy as np
import os
from batches_iterator import BatchesIterator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

def generate_model() -> tf.keras.Sequential:
    """
    Function that generates sequential CNN, see module doc for more details

    Returns
    -------
    model : tf.keras.Sequential
        CNNA for classification.

    """
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
    
    train_batch_size = 32
    valid_batch_size = 32
    
    train_batches = BatchesIterator(train_batch_size,train_no_lesion_dir,train_lesion_dir)
    valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir)
    
    file_name = 'network_A_1'
    saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "log"), histogram_freq=1)
    
    cllbcks= [saveWeigts, tensorboard_callback]
    
    if os.path.exists(os.path.join(os.getcwd(),'network_A_1_best_weights.h5')):
        model = tf.keras.models.load_model('network_A_1_best_weights.h5')
    else:
        model = generate_model()
    
    model.fit(x=train_batches, epochs=12, steps_per_epoch=len(train_batches), callbacks = cllbcks,
              validation_data=valid_batches, verbose=1)
        
    
        