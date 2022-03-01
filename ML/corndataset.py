# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 23:30:09 2022

@author: Daniel Bresnahan
"""
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms, io
import pandas as pd
import torch


class CornDataset(Dataset):
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
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = sum(self.img_labels.iloc[idx, 1:5])
        
        if label:
            label = 1
            
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        image = io.read_image(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
        
        
        
        
        
        