# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:26:01 2022

@author: Daniel
"""

from imagepatching import ImagePatcher
import os
import csv
import PIL
from tqdm import tqdm

base_dir = "Z:\Coding\Dataset"

annotations = os.path.join(base_dir, "annotations_handheld.csv")

pic_locations = os.path.join(base_dir, "images_handheld")

save_dir = os.path.join(base_dir, "patched")

vert_patcher = ImagePatcher("Z:\Coding\Dataset\patched", (224, 224), imageSize=(6000, 4000))
hor_patcher = ImagePatcher("Z:\Coding\Dataset\patched", (224, 224), imageSize=(4000, 6000))

with open(annotations, "r") as f:
    reader = csv.reader(f)
    
    next(reader)
    
    
    for line in tqdm(reader):
        cur_image = os.path.join(pic_locations,line[0])
        
        im = PIL.Image.open(cur_image)
        
        if im.size == (6000, 4000):
            vert_patcher.patch(cur_image, (int(line[1]), int(line[2]), int(line[3]), int(line[4])))
        else:
            hor_patcher.patch(cur_image, (line[1], line[2], line[3], line[4]))

        
