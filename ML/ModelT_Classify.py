# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:39:38 2022

@author: Danie
"""

from tensorflow import keras
from batches_iterator import BatchesIterator
import shutil
import os
from tqdm import tqdm

model = keras.models.load_model('network_A_1_best_weights.h5', compile=False)

test_lesion_dir = "E:\\Coding\\Dataset\\Test\\Positive"
test_no_lesion_dir = "E:\\Coding\\Dataset\\Test\\Negative"
valid_batch_size = 1

train_dest = "E:\\Coding\\Dataset\\Test\\Copy"

valid_batches = BatchesIterator(valid_batch_size,test_no_lesion_dir,test_lesion_dir,
                                no_lesion=False, return_dir=True)

print("Files before hard mining: {}".format(len(os.listdir(train_dest))))
for i in tqdm(range(len(valid_batches))):
    x, y, path = valid_batches.__getitem__(i)
    predict = model(x, training=False)
    
    
    if y[0] == 1 and predict[0][0] < 0.5:
        # print(type(path[0]))
        shutil.copyfile(path[0], os.path.join(train_dest, os.path.basename(path[0])))

print("Files after hard mining: {}".format(len(os.listdir(train_dest))))
        

