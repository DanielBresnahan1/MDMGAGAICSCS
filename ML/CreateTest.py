# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:26:30 2022

@author: Danie

DEPRICATED FILE: DO NOT RUN
This file was responsible for generating the grid patched images of the test directory. This was used 
when the final classification method involved sampling a set of representative points from
a grid classification of the test images by networks A, B, C

CreateTest will take in a directory of images, and the path for the label_csv 
And it will perform a grid patch on each image; A grid patch is when an image is split into even chunks of 
224X224. After grid patching, it will generate a csv file containg the image name and its class label
"""
import os
import shutil
from createSplits import find_unique
from imagepatching import ImagePatcher
import PIL
from tqdm import tqdm
import csv

def patch_test(base_dir: str, test_dir: str, label_csv: str, annotations_csv: str):
    """
    

    Parameters
    ----------
    base_dir : str
        Path to base directory that contains the csv and the images.
    test_dir : str
        Path to dir that contains test images.
    label_csv : str
        Name of csv to save class labels too.
    annotations_csv : str
        Name of csv which contains images and lesion coords.

    Returns
    -------
    None.

    """
    
    vert_patcher = ImagePatcher(None, (224, 224), imageSize=(6000, 4000), majorAxisDif=0, rotBoosting=False)
    hor_patcher = ImagePatcher(None, (224, 224), imageSize=(4000, 6000), majorAxisDif=0, rotBoosting=False)
    weird_patcher = ImagePatcher(None, (224, 224), imageSize=(5184, 3456), majorAxisDif=0, rotBoosting=False)
    
    image_list = []
    
    uniques = find_unique(os.path.join(base_dir, annotations_csv))
    
    for key in tqdm(uniques.keys()):
        image_dir = os.path.join(base_dir, "test", key.split(".")[0])
        os.mkdir(image_dir)
        
        cur_image = os.path.join(base_dir, test_dir, key)
        
        im = PIL.Image.open(cur_image)
        
        
        if im.size == (6000, 4000):
            vert_patcher.set_save_dir(image_dir)
            vert_patcher.patch(cur_image, (0, 0, 0, 0))
        elif im.size == (4000, 6000):
            hor_patcher.set_save_dir(image_dir)
            hor_patcher.patch(cur_image, (0, 0, 0, 0))
        else:
            weird_patcher.set_save_dir(image_dir)
            weird_patcher.patch(cur_image, (0, 0, 0, 0))
        
        coord_list = [int(c) for c in uniques[key][0][1:5]]
        label = 0
        if all(coord_list):
            label = 1
        
        image_list.append([key, label])
        
    
    with open(os.path.join(base_dir, label_csv), "w", newline='') as f:
        writer = csv.writer(f)
        
        writer.writerows(image_list)
        
        
        


if __name__=="__main__":
    base_dir = "E:\\Coding\\Dataset"
    test_dir = "images_test"
    label_csv = "test_labels.csv"
    annotations_csv = "annotations_test.csv"
    
    patch_test(base_dir, test_dir, label_csv, annotations_csv)
    
    