# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:14:11 2022

@author: Danie
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:15:17 2022

@author: Danie
"""


import tensorflow as tf
from batches_iterator import BatchesIterator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Conv2D as Convolution2D
from tensorflow.keras.callbacks import ModelCheckpoint


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

model = Sequential()

valid_lesion_dir = "E:\\Coding\\Dataset\\Validation\\Positive"
valid_no_lesion_dir = "E:\\Coding\\Dataset\\Validation\\Negative"
train_lesion_dir = "E:\\Coding\\Dataset\\Train\\Positive"
train_no_lesion_dir = "E:\\Coding\\Dataset\\Train\\Negative"

train_batch_size = 200
valid_batch_size = 500

train_batches = BatchesIterator(train_batch_size,train_no_lesion_dir,train_lesion_dir)
valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir)


model.add(BatchNormalization(axis=1, input_shape=(3,224,224)))
model.add(Convolution2D(16,3,3, data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Convolution2D(8,3,3, data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,3,3, data_format="channels_first"))
model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(1))

model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=[
                  tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy'),
                  'accuracy'])
model.summary()  

file_name = 'network_T_1'
saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='accuracy', verbose=1, save_best_only=True)

cllbcks= [saveWeigts]


if __name__=="__main__":
    
    for mini_epoch in range(1000):
        print("Epoch ", mini_epoch)
    
        X_train, Y_train, _ = next(train_batches)
        X_valid, Y_valid, _ = next(valid_batches)
    
        model.fit(X_train, Y_train, callbacks = cllbcks,
    			validation_data=(X_valid,Y_valid), verbose=0)