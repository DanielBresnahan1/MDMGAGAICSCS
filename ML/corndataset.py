# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 23:30:09 2022

@author: Daniel Bresnahan
"""
import os

import pandas as pd


class CornDataset():
    """ Corn with and without NCLB Dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """

        Parameters
        ----------
        csv_file : String
            Path to CSV File with Annotations.
        root_dir : String
            Directory with all the images.
        transform : callable, optional
            Optional Transformation to apply to an image

        """
        
        self.img_labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)
    
    
    def __getitem__(self, idx):
        """
        Will Return the item at idx, with label 1 as healthy 0 as unhealthy

        Parameters
        ----------
        idx : list or tensor
        Indexes to select

        Returns
        -------
        None.

        """
        
        
        label = sum(self.img_labels.iloc[idx, 1:5])
        
        if label:
            label = 1
            
        
        return label
        
        
if __name__=="__main__":
    csv = "E:\\Coding\\Dataset\\annotations_test.csv"
    root_dir = "E:\\Coding\\Dataset"
    data = CornDataset(csv, root_dir)
    
    num_pos = 0
    num_neg = 0
    
    for i in range(len(data)):
        label = data.__getitem__(i)
        
        if label == 0:
            num_neg += 1
        else:
            num_pos += 1
            
    print("Number Positive Samples: {} \n Number Negative Samples: {}".format(num_pos, num_neg))
        
        
        
        