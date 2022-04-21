# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:17:15 2022

@author: Daniel

third of 3 CNN's for classification, takes the form of a keras Sequential CNN. 
Also contains functionality to train the model using a train val split specificied in the relevant csv files

To train the model please set the following variables in __main__
valid_lesion_dir = "Path to folder containing positive patches for validation"
valid_no_lesion_dir = "Path to folder containing negative patches for validation"
train_lesion_dir = "Path to folder containing positive patches for training"
train_no_lesion_dir = "Path to folder containing negative patches for training"

model Structure:
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     batch_normalization_22 (Bat  (None, 3, 224, 224)      12        
     chNormalization)                                                
                                                                     
     conv2d_18 (Conv2D)          (None, 32, 222, 222)      896       
                                                                     
     batch_normalization_23 (Bat  (None, 32, 222, 222)     128       
     chNormalization)                                                
                                                                     
     activation_20 (Activation)  (None, 32, 222, 222)      0         
                                                                     
     conv2d_19 (Conv2D)          (None, 32, 220, 220)      9248      
                                                                     
     batch_normalization_24 (Bat  (None, 32, 220, 220)     128       
     chNormalization)                                                
                                                                     
     activation_21 (Activation)  (None, 32, 220, 220)      0         
                                                                     
     max_pooling2d_7 (MaxPooling  (None, 32, 110, 110)     0         
     2D)                                                             
                                                                     
     conv2d_20 (Conv2D)          (None, 64, 108, 108)      18496     
                                                                     
     batch_normalization_25 (Bat  (None, 64, 108, 108)     256       
     chNormalization)                                                
                                                                     
     activation_22 (Activation)  (None, 64, 108, 108)      0         
                                                                     
     conv2d_21 (Conv2D)          (None, 64, 106, 106)      36928     
                                                                     
     batch_normalization_26 (Bat  (None, 64, 106, 106)     256       
     chNormalization)                                                
                                                                     
     activation_23 (Activation)  (None, 64, 106, 106)      0         
                                                                     
     max_pooling2d_8 (MaxPooling  (None, 64, 53, 53)       0         
     2D)                                                             
                                                                     
     dropout_8 (Dropout)         (None, 64, 53, 53)        0         
                                                                     
     conv2d_22 (Conv2D)          (None, 64, 51, 51)        36928     
                                                                     
     batch_normalization_27 (Bat  (None, 64, 51, 51)       256       
     chNormalization)                                                
                                                                     
     activation_24 (Activation)  (None, 64, 51, 51)        0         
                                                                     
     conv2d_23 (Conv2D)          (None, 64, 49, 49)        36928     
                                                                     
     batch_normalization_28 (Bat  (None, 64, 49, 49)       256       
     chNormalization)                                                
                                                                     
     activation_25 (Activation)  (None, 64, 49, 49)        0         
                                                                     
     max_pooling2d_9 (MaxPooling  (None, 64, 24, 24)       0         
     2D)                                                             
                                                                     
     dropout_9 (Dropout)         (None, 64, 24, 24)        0         
                                                                     
     conv2d_24 (Conv2D)          (None, 128, 22, 22)       73856     
                                                                     
     batch_normalization_29 (Bat  (None, 128, 22, 22)      512       
     chNormalization)                                                
                                                                     
     activation_26 (Activation)  (None, 128, 22, 22)       0         
                                                                     
     conv2d_25 (Conv2D)          (None, 128, 20, 20)       147584    
                                                                     
     batch_normalization_30 (Bat  (None, 128, 20, 20)      512       
     chNormalization)                                                
                                                                     
     activation_27 (Activation)  (None, 128, 20, 20)       0         
                                                                     
     max_pooling2d_10 (MaxPoolin  (None, 128, 10, 10)      0         
     g2D)                                                            
                                                                     
     dropout_10 (Dropout)        (None, 128, 10, 10)       0         
                                                                     
     conv2d_26 (Conv2D)          (None, 256, 8, 8)         295168    
                                                                     
     batch_normalization_31 (Bat  (None, 256, 8, 8)        1024      
     chNormalization)                                                
                                                                     
     activation_28 (Activation)  (None, 256, 8, 8)         0         
                                                                     
     conv2d_27 (Conv2D)          (None, 256, 6, 6)         590080    
                                                                     
     batch_normalization_32 (Bat  (None, 256, 6, 6)        1024      
     chNormalization)                                                
                                                                     
     activation_29 (Activation)  (None, 256, 6, 6)         0         
                                                                     
     max_pooling2d_11 (MaxPoolin  (None, 256, 3, 3)        0         
     g2D)                                                            
                                                                     
     dropout_11 (Dropout)        (None, 256, 3, 3)         0         
                                                                     
     conv2d_28 (Conv2D)          (None, 16, 1, 1)          36880     
                                                                     
     batch_normalization_33 (Bat  (None, 16, 1, 1)         64        
     chNormalization)                                                
                                                                     
     activation_30 (Activation)  (None, 16, 1, 1)          0         
                                                                     
     dropout_12 (Dropout)        (None, 16, 1, 1)          0         
                                                                     
     flatten_2 (Flatten)         (None, 16)                0         
                                                                     
     dense_2 (Dense)             (None, 1)                 17        
                                                                     
     activation_31 (Activation)  (None, 1)                 0         
                                                                     
    =================================================================
    Total params: 1,287,437
    Trainable params: 1,285,223
    Non-trainable params: 2,214
    _________________________________________________________________
    
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


def generate_model() -> tf.keras.Sequential:
    """
    Function that generates sequential CNN, see module doc for more details

    Returns
    -------
    model : tf.keras.Sequential
        CNNC for classification.

    """
    
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