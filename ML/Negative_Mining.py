# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:25:07 2022

@author: Danie
"""
import numpy as np
import tensorflow as tf
from batches_iterator import BatchesIterator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
import os

def generate_model():
    base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,weights=None,input_shape=(3, 224, 224))    
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    
    predictions = Dense(1, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # for layer in base_model.layers:
    #     layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=[
                      tf.keras.metrics.binary_crossentropy, 'accuracy'])
    
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
    
    model = generate_model()
    
    file_name = 'network_M_1'
    saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "log"), histogram_freq=1)
    
    cllbcks= [saveWeigts, tensorboard_callback]
    
    model.fit(x=train_batches, epochs=12, steps_per_epoch=len(train_batches), validation_data=valid_batches, verbose=1)
        