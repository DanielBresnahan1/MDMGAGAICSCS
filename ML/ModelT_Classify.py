# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:39:38 2022

@author: Danie
"""

from tensorflow import keras
from batches_iterator import BatchesIterator

model = keras.models.load_model('network_T_1_best_weights.h5', compile=False)

valid_lesion_dir = "E:\\Coding\\Dataset\\Validation\\Positive"
valid_no_lesion_dir = "E:\\Coding\\Dataset\\Validation\\Negative"
valid_batch_size = 1

valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir, no_lesion=False)

for i in range(3):
    x, y, file = next(valid_batches)
    label = model(x, training=False)
    
    print("File: {} , Label: {}".format(file, label))
    

