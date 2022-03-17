# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:18:08 2022

@author: Danie
"""

import PIL
import os
from tqdm import tqdm

train_dir = "Z:\Coding\Dataset\Train\Positive"

for file in tqdm(os.listdir(train_dir)):
    name_path = os.path.join(train_dir,  file)
    im = PIL.Image.open(name_path)
    
    if not (im.size[0] == 224 and im.size[1] == 224):
        im = im.resize((224, 224))
        im.save(name_path)
        