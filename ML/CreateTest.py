# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:26:30 2022

@author: Danie
"""
import os
import shutil
from createSplits import find_unique
from imagepatching import ImagePatcher
import PIL
from tqdm import tqdm
import csv

def patch_test(base_dir, test_dir, label_csv, annotations_csv):
    
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
    
    