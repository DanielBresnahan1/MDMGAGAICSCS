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

class TestIterator(Sequence):
    def __init__(self, batch_size, label,
              image_folder_path, return_name=True):

        self.batch_size = batch_size
        self.files = []
        self.label = label
        self.return_name = return_name
        for file in os.listdir(image_folder_path):
            self.files.append((os.path.join(image_folder_path, file), label))

            
    
    def __len__(self):
        return math.ceil(len(self.files)/self.batch_size)
    

    def __getitem__(self, index):

        if len(self.files)-(index*self.batch_size) <= self.batch_size:
            this_batch_size = len(self.files) - (index*self.batch_size)
        else:
            this_batch_size = self.batch_size

        batch_x = np.zeros((this_batch_size,3,224,224), dtype='uint8')
		
        batch_y = np.zeros((this_batch_size), dtype='uint8')

        batch_file_names = []
        for x in range(this_batch_size):
            this_file_name = self.files[(index*self.batch_size)+x][0]
            pic = PIL.Image.open(this_file_name)
            pic = np.array(pic)
            batch_x[x]=pic.reshape(3,224,224)
            batch_y[x]=self.files[(index*self.batch_size)+x][1]
            batch_file_names.append(this_file_name) 
            
        if self.return_name:
            return batch_x, batch_y, batch_file_names
        else:
            return batch_x, batch_y
    


if __name__=="__main__":
    
    import csv
    
    test_image = "DSC00030"
    base_dir = "E:\\Coding\\Dataset"
    image_dir = os.path.join(base_dir, "Test", test_image)
    batch_size = 8
    csvf = "test_labels.csv"
    label = 0
    
    with open(os.path.join(base_dir, csvf), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == test_image+".jpg" or row[0] == test_image+".jpeg":
                label = row[1]
                break
    
    testIter = TestIterator(batch_size, label, image_dir, return_name=True)
    
    for i in range(len(testIter)):
        x, y, name = testIter.__getitem__(i)
        print("File: {}, Label: {}".format(name, y))
    
    