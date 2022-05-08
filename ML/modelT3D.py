# -*- coding: utf-8 -*-
"""
Created on Tue May  3 01:47:53 2022

@author: Danie
"""

from testIterator3D import TestIterator3D
from ModelT import create_res_net
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

if __name__=="__main__":
    
    base_dir = "E:\\Coding\\Dataset"
    train_image_dir = os.path.join(base_dir, "Train_Maps")
    val_image_dir = os.path.join(base_dir, "Test_Maps")
    batch_size = 32
    train_csvf = "annotations_val.csv"
    val_csvf = "annotations_test.csv"
    
    epochsN = 64
    
    train_csv_dir = os.path.join(base_dir, train_csvf)
    val_csv_dir = os.path.join(base_dir, val_csvf)
    
    train_batches = TestIterator3D(batch_size, train_csv_dir, train_image_dir)
    val_batches = TestIterator3D(batch_size, val_csv_dir, val_image_dir)
    
    file_name = 'network_T3D_1'
    
    saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "log"), histogram_freq=1)
    
    cllbcks= [saveWeigts, tensorboard_callback]
    
    model_path = os.path.join(os.getcwd(),file_name+'_best_weights.h5')
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(file_name+'_best_weights.h5')
    else:
        model = create_res_net((3, 193, 126), 64, 10)
    
    model.summary()
    
    model.fit(x=train_batches, epochs=epochsN, steps_per_epoch=len(train_batches), callbacks = cllbcks,
              validation_data=val_batches, verbose=1)
    
    model.save(model_path, include_optimizer=False)
