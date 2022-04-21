# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:32:20 2022

@author: Danie

File that utilizes imagePatcher to geenrate patches for heat_map_generation. This works slightly differnt
than stage 1 patching, because all iamges should be patched using a stride, regardless of label.
"""
import os
from createSplits import find_unique
from imagepatching import ImagePatcher
import PIL
import csv

def map_split(directory: str, annotations_csv: str, save_dir: str, save_csv: str):
    """
    Iterators over every image in the directory associated with annotations_csv and generates patches 
    by striding some stride length, set to 30 as per paper specifications.
    In total for an image of size 6000x4000 this will generate 25,634 patches. 

    Parameters
    ----------
    directory : str
        directory path that points to base directory containing every folder and file.
    annotations_csv : str
        Name of annotations_csv, containing images and lession coordiantes.
    save_dir : str
        Name of directory to save folders containing patches to.
    save_csv : str
        Name of new csv to create that will contain the images and their associated class labels.

    Returns
    -------
    None.

    """
    unique_images = find_unique(annotations_csv)
    
    vert_patcher = ImagePatcher("", (224, 224), stride=30, imageSize=(6000, 4000), majorAxisDif=0, rotBoosting=False)
    
    hor_patcher = ImagePatcher("", (224, 224), stride=30, imageSize=(4000, 6000), majorAxisDif=0, rotBoosting=False)
    
    weird_patcher = ImagePatcher("", (224, 224), stride=30, imageSize=(5184, 3456), majorAxisDif=0, rotBoosting=False)
    
    #it took actually forever to figure out that there are only 2 images with this dimension :()
    singular_weird_patcher = ImagePatcher("", (224, 224), stride=30, imageSize=(3456, 5184), majorAxisDif=0, rotBoosting=False)
    
    new_csv = []
    
    for index, image in enumerate(unique_images.keys()):
        print("~~~~Image Number: {}~~~~~".format(index))
        # print(image)
        cur_image = os.path.join(directory,image)

        if sum([int(c) for c in unique_images[image][0][1:5]]):
            new_csv.append([image, 1])
        else:
            new_csv.append([image, 0])
        
        new_dir = os.path.join(save_dir, image.split(".")[0])
        if os.path.exists(new_dir):
            continue
        
        os.mkdir(new_dir)
        
        im = PIL.Image.open(cur_image)
        
        
        if im.size == (6000, 4000):
            vert_patcher.set_save_dir(new_dir)
            vert_patcher.patch(cur_image, (0, 0, 0, 0), sub_folder=False, Map=True)
        elif im.size == (4000, 6000):
            hor_patcher.set_save_dir(new_dir)
            hor_patcher.patch(cur_image, (0, 0, 0, 0), sub_folder=False, Map=True)
        elif im.size == (5184, 3456):
            weird_patcher.set_save_dir(new_dir)
            weird_patcher.patch(cur_image, (0, 0, 0, 0), sub_folder=False, Map=True)
        else:
            singular_weird_patcher.set_save_dir(new_dir)
            singular_weird_patcher.patch(cur_image, (0, 0, 0, 0), sub_folder=False, Map=True)
        
        
        
    if save_csv:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(new_csv)

if __name__=="__main__":
    base_dir = "E:\\Coding\\Dataset"
    annotations_csv="annotations_test.csv"
    pic_folder="images_test"
    save_folder="Test_Map"
    
    image_dir = os.path.join(base_dir, pic_folder) 
    acsv_dir = os.path.join(base_dir, annotations_csv)
    save_dir = os.path.join(base_dir, save_folder)
    
    save_csv = "test_map.csv"

    scsv_dir = os.path.join(base_dir, save_csv)
    
    
    map_split(image_dir, acsv_dir, save_dir, "")
    
    