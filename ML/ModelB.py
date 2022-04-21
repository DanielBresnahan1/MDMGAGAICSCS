# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:15:17 2022

@author: Danie

Second of 3 CNN's for classification, takes the form of a keras Sequential CNN. 
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
     batch_normalization_11 (Bat  (None, 3, 224, 224)      12        
     chNormalization)                                                
                                                                     
     conv2d_9 (Conv2D)           (None, 64, 222, 222)      1792      
                                                                     
     batch_normalization_12 (Bat  (None, 64, 222, 222)     256       
     chNormalization)                                                
                                                                     
     activation_10 (Activation)  (None, 64, 222, 222)      0         
                                                                     
     conv2d_10 (Conv2D)          (None, 64, 220, 220)      36928     
                                                                     
     batch_normalization_13 (Bat  (None, 64, 220, 220)     256       
     chNormalization)                                                
                                                                     
     activation_11 (Activation)  (None, 64, 220, 220)      0         
                                                                     
     dropout (Dropout)           (None, 64, 220, 220)      0         
                                                                     
     max_pooling2d_3 (MaxPooling  (None, 64, 110, 110)     0         
     2D)                                                             
                                                                     
     conv2d_11 (Conv2D)          (None, 128, 108, 108)     73856     
                                                                     
     batch_normalization_14 (Bat  (None, 128, 108, 108)    512       
     chNormalization)                                                
                                                                     
     activation_12 (Activation)  (None, 128, 108, 108)     0         
                                                                     
     dropout_1 (Dropout)         (None, 128, 108, 108)     0         
                                                                     
     conv2d_12 (Conv2D)          (None, 128, 106, 106)     147584    
                                                                     
     batch_normalization_15 (Bat  (None, 128, 106, 106)    512       
     chNormalization)                                                
                                                                     
     activation_13 (Activation)  (None, 128, 106, 106)     0         
                                                                     
     dropout_2 (Dropout)         (None, 128, 106, 106)     0         
                                                                     
     max_pooling2d_4 (MaxPooling  (None, 128, 53, 53)      0         
     2D)                                                             
                                                                     
     conv2d_13 (Conv2D)          (None, 128, 51, 51)       147584    
                                                                     
     batch_normalization_16 (Bat  (None, 128, 51, 51)      512       
     chNormalization)                                                
                                                                     
     activation_14 (Activation)  (None, 128, 51, 51)       0         
                                                                     
     dropout_3 (Dropout)         (None, 128, 51, 51)       0         
                                                                     
     max_pooling2d_5 (MaxPooling  (None, 128, 25, 25)      0         
     2D)                                                             
                                                                     
     conv2d_14 (Conv2D)          (None, 256, 23, 23)       295168    
                                                                     
     batch_normalization_17 (Bat  (None, 256, 23, 23)      1024      
     chNormalization)                                                
                                                                     
     activation_15 (Activation)  (None, 256, 23, 23)       0         
                                                                     
     dropout_4 (Dropout)         (None, 256, 23, 23)       0         
                                                                     
     conv2d_15 (Conv2D)          (None, 256, 21, 21)       590080    
                                                                     
     batch_normalization_18 (Bat  (None, 256, 21, 21)      1024      
     chNormalization)                                                
                                                                     
     activation_16 (Activation)  (None, 256, 21, 21)       0         
                                                                     
     dropout_5 (Dropout)         (None, 256, 21, 21)       0         
                                                                     
     max_pooling2d_6 (MaxPooling  (None, 256, 10, 10)      0         
     2D)                                                             
                                                                     
     conv2d_16 (Conv2D)          (None, 512, 8, 8)         1180160   
                                                                     
     batch_normalization_19 (Bat  (None, 512, 8, 8)        2048      
     chNormalization)                                                
                                                                     
     activation_17 (Activation)  (None, 512, 8, 8)         0         
                                                                     
     dropout_6 (Dropout)         (None, 512, 8, 8)         0         
                                                                     
     conv2d_17 (Conv2D)          (None, 32, 6, 6)          147488    
                                                                     
     batch_normalization_20 (Bat  (None, 32, 6, 6)         128       
     chNormalization)                                                
                                                                     
     activation_18 (Activation)  (None, 32, 6, 6)          0         
                                                                     
     dropout_7 (Dropout)         (None, 32, 6, 6)          0         
                                                                     
     flatten_1 (Flatten)         (None, 1152)              0         
                                                                     
     batch_normalization_21 (Bat  (None, 1152)             4608      
     chNormalization)                                                
                                                                     
     dense_1 (Dense)             (None, 1)                 1153      
                                                                     
     activation_19 (Activation)  (None, 1)                 0         
                                                                     
    =================================================================
    Total params: 2,632,685
    Trainable params: 2,627,239
    Non-trainable params: 5,446
    _________________________________________________________________
"""

import numpy as np
import os
from batches_iterator import BatchesIterator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint



def generate_model() -> tf.keras.Sequential:
    """
    Function that generates sequential CNN, see module doc for more details

    Returns
    -------
    model : tf.keras.Sequential
        CNNB for classification.

    """
    
    model = Sequential()
    
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    
    model.add(Conv2D(64,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    
    
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(128,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(256,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    
    model.add(Conv2D(256,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    
    model.add(Conv2D(512,3))
    model.add(BatchNormalization(axis=1))
    
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    
    model.add(Conv2D(32,3))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    
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
    
    generate_model()
    
    # valid_lesion_dir = "E:\\Coding\\Dataset\\Validation\\Positive"
    # valid_no_lesion_dir = "E:\\Coding\\Dataset\\Validation\\Negative"
    # train_lesion_dir = "E:\\Coding\\Dataset\\Train\\Positive"
    # train_no_lesion_dir = "E:\\Coding\\Dataset\\Train\\Negative"
    
    # train_batch_size = 32
    # valid_batch_size = 32
    
    # train_batches = BatchesIterator(train_batch_size,train_no_lesion_dir,train_lesion_dir)
    # valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir)
    
    # file_name = 'network_B_1'
    # saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "log"), histogram_freq=1)
    
    # cllbcks= [saveWeigts, tensorboard_callback]
    
    # if os.path.exists(os.path.join(os.getcwd(),'network_B_1_best_weights.h5')):
    #     model = tf.keras.models.load_model('network_B_1_best_weights.h5')
    # else:
    #     model = generate_model()
    
    # model.fit(x=train_batches, epochs=12, steps_per_epoch=len(train_batches), callbacks = cllbcks,
    #           validation_data=valid_batches, verbose=1)