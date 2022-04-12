# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:26:01 2022

@author: Daniel
"""

from imagepatching import ImagePatcher
import os
import csv
import PIL


def patch_all(base_dir, annotations_csv, pic_folder, save_folder):
    """
    Patch_all will go through all the files listed in the data csv. 
    At this point, it assumes that the header has been removed from the csv.

    Parameters
    ----------
    base_dir : TYPE
        DESCRIPTION.
    annotations_csv : TYPE
        DESCRIPTION.
    pic_folder : TYPE
        DESCRIPTION.
    save_folder : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    annotations = os.path.join(base_dir, annotations_csv)
    
    pic_locations = os.path.join(base_dir, pic_folder)
    
    save_dir = os.path.join(base_dir, save_folder)
    
    vert_patcher = ImagePatcher(save_dir, (224, 224), imageSize=(6000, 4000), rotBoosting=True)
    hor_patcher = ImagePatcher(save_dir, (224, 224), imageSize=(4000, 6000), rotBoosting=True)
    weird_patcher = ImagePatcher(save_dir, (224, 224), imageSize=(5184, 3456), rotBoosting=True)
    
    with open(annotations, "r") as f:
        reader = csv.reader(f)
        
        
        for index, line in enumerate(reader):
            
            print("~~~~~Image Number: {}~~~~~~".format(index+1))
            
            cur_image = os.path.join(pic_locations,line[0])
            
            im = PIL.Image.open(cur_image)
            
            if im.size == (6000, 4000):
                vert_patcher.patch(cur_image, (int(line[1]), int(line[2]), int(line[3]), int(line[4])), sub_folder=True)
            elif im.size == (4000, 6000):
                hor_patcher.patch(cur_image, (int(line[1]), int(line[2]), int(line[3]), int(line[4])), sub_folder=True)
            else:
                weird_patcher.patch(cur_image, (int(line[1]), int(line[2]), int(line[3]), int(line[4])), sub_folder=True)


if __name__=="__main__":
    base_dir = "E:\Coding\Dataset"
    annotations_csv="annotations_handheld.csv"
    pic_folder="images_handheld"
    save_folder="Train"
    patch_all(base_dir, annotations_csv, pic_folder, save_folder)
        
