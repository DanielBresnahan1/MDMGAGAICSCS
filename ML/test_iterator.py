# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:13:56 2022

@author: Danie

keras Sequence Iterator responsible for iterating through the folders that contain the heat maps, generated
by heatMapper
"""

from tensorflow.keras.utils import Sequence
import numpy as np
import os
import PIL
import math
import csv
import random


class TestIterator(Sequence):
    def __init__(self, batch_size: int, csv_dir: str,
              image_folder_path: str, return_name=False, preserve_image=False):
        """
        Constructor for TestIterator

        Parameters
        ----------
        batch_size : int
            size of batch, aka number of files per batch.
        csv_dir : str
            Path to csv file.
        image_folder_path : str
            Path to folder containing subdirs, which themselves contain the heatmaps.
        return_name : BOOLEAN, optional
            FOR TESTING, return the names of the files in each batch. The default is False.
        preserve_image : BOOL, optional
            If shuffling should occur at the end of epoch
            Should be True for training, false for prediction. The default is False.

        Returns
        -------
        None.

        """

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
   
    
    def __len__(self) -> int:
        """
        Return the number of batches in the dataset, the number of steps per epoch

        Returns
        -------
        int
            number of batches in the dataset.

        """
        return math.ceil(len(self.files)/self.batch_size)
    
    def get_labels(self):
        """
        Open the csv file and extract the labels, add them to a dictioinary with the image name as key

        Returns
        -------
        None.

        """
        
        with open(self.csv_dir, "r") as f:
            reader = csv.reader(f)
            
            for row in reader:
                self.labels.update({row[0].split(".")[0]:int(row[1])})
        
    def on_epoch_end(self):
        """
        Function that runs conditionally at the end of each epoch, shuffles the data

        Returns
        -------
        None.

        """
        if not self.preserve_image:
            random.shuffle(self.files)

    def __getitem__(self, index: int) -> [np.ndarray, np.ndarray]:
        """
        The meat of the iterator, will return numpy arrays contianing the batch X, and Batch Y 
        associated with the current batch index.

        Parameters
        ----------
        index : int
            The batch index, 0 < index < len.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing batch X and batch Y.
            X is of the form (batch_size, 1, 193, 126)
            Y is the form (batch_size, )

        """

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
    
    