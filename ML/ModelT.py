# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:07:30 2022

@author: Danie

ResNet equivelent code, built from medium article found here: https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
"""

import os
from tensorflow import Tensor
from test_iterator import TestIterator
from tensorflow.keras.layers import Conv2D, Input, AveragePooling2D, ReLU, BatchNormalization, Dense, Flatten, Add
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = Input(shape=(1, 193, 126))
    num_filters = 64
    
    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = AveragePooling2D(4)(t)
    t = Flatten()(t)
    outputs = Dense(10, activation='softmax')(t)
    
    model = Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__=="__main__":
    
    
    base_dir = "E:\\Coding\\Dataset"
    train_image_dir = os.path.join(base_dir, "Test_Map_Heat")
    val_image_dir = os.path.join(base_dir, "Train_Map_Heat")
    batch_size = 32
    train_csvf = "test_labels.csv"
    val_csvf = "train_map.csv"
    
    epochsN = 32
    
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
        model = create_res_net()
    
    model.summary()
    
    model.fit(x=train_batches, epochs=epochsN, steps_per_epoch=len(train_batches), callbacks = cllbcks,
              validation_data=val_batches, verbose=1)
    
    model.save(model_path, include_optimizer=False)
