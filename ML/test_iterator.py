# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:13:56 2022

@author: Danie
"""

from tensorflow.keras.utils import Sequence
import numpy as np
import os
import PIL
import math
import csv
import random

"""@package
Keras Sequence Iterator for heat map images, for training the final model
"""
class TestIterator(Sequence):
    def __init__(self, batch_size, csv_dir,
              image_folder_path, return_name=False, preserve_image=False):

        self.batch_size = batch_size
        self.files = []
        self.labels = {}
        self.csv_dir = csv_dir
        self.return_name = return_name
        self.preserve_image = preserve_image
        self.get_labels()
        for folder in os.listdir(image_folder_path):
            for file in os.listdir(os.path.join(image_folder_path, folder)):
                self.files.append((os.path.join(image_folder_path, folder, file), self.labels[folder]))
   
    
    def __len__(self):
        return math.ceil(len(self.files)/self.batch_size)
    
    def get_labels(self):
        
        with open(self.csv_dir, "r") as f:
            reader = csv.reader(f)
            
            for row in reader:
                self.labels.update({row[0].split(".")[0]:int(row[1])})
        
    def on_epoch_end(self):
        if not self.preserve_image:
            random.shuffle(self.files)

    def __getitem__(self, index):

        if len(self.files)-(index*self.batch_size) <= self.batch_size:
            this_batch_size = len(self.files) - (index*self.batch_size)
        else:
            this_batch_size = self.batch_size

        batch_x = np.zeros((this_batch_size,1,193,126))
        batch_y = np.zeros((this_batch_size), dtype='uint8')

        batch_file_names = []
        for x in range(this_batch_size):
            this_file_name, this_file_label = self.files[(index*self.batch_size)+x]
            # print(this_file_name)
            pic = PIL.Image.open(this_file_name)
            # print("Image_Shape: {}".format(pic.size))
            this_pic = np.zeros((193, 126))
            pic = np.array(pic)
            # print("Numpy array shape: {}".format(pic.shape))
            
            this_pic[:,:] = pic
            
            batch_x[x]=this_pic
            batch_y[x]=this_file_label

            batch_file_names.append(this_file_name) 
            
        if self.return_name:
            return batch_x, batch_y, batch_file_names
        else:
            return batch_x, batch_y


if __name__=="__main__":
    
    base_dir = "E:\\Coding\\Dataset"
    image_dir = os.path.join(base_dir, "Train_Map_Heat")
    batch_size = 8
    csvf = "train_map.csv"
    csv_dir = os.path.join(base_dir, csvf)
    
    testIter = TestIterator(batch_size, csv_dir, image_dir, return_name=True)
    
    for i in range(2):
        x, y, name = testIter.__getitem__(i)
        print(x)
    
    