# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:26:01 2022

@author: Daniel

This script defines the method patch_all, and runs it given current running context
The purpose is to utilize ImagePatcher to patch all images (both positive and negative) in a given dir. 

Should be ran after createSplits. 
"""

from imagepatching import ImagePatcher
import os
import csv
import PIL

def patch_all(base_dir: str, annotations_csv: str, pic_folder: str, save_folder: str):
    """
    patch_all is a testing script, that will iterate through each file described 
    at base_dir/annotations_csv. And perform patching with an ImagePatcher. 
    

    Parameters
    ----------
    base_dir : str
        path of base dir, which every file and folder reside in.
    annotations_csv : str
        Name of the annotations csv, which contains all file labels and coords.
    pic_folder : str
        Name of folder which contains images to patch.
    save_folder : str
        Name of folder to save patched images to.

    Returns
    -------
    None.

    """
    
    annotations = os.path.join(base_dir, annotations_csv)
    
    pic_locations = os.path.join(base_dir, pic_folder)
    
    save_dir = os.path.join(base_dir, save_folder)
    
    vert_patcher = ImagePatcher(save_dir, (224, 224), imageSize=(6000, 4000), rotBoosting=True)
    hor_patcher = ImagePatcher(save_dir, (224, 224), imageSize=(4000, 6000), rotBoosting=True)
    #Some of the iamges have a super weird resolution of 5184 X 3456??
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
        
