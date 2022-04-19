# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:36:20 2022

@author: Danie
"""

import numpy as np
import os
import tensorflow as tf
import PIL
from PIL import Image
from tqdm import tqdm
from mapIterator import MapIterator
import re

class TileMapper:
    
    def __init__(self, images_dir, save_dir, original_dir, modelA, modelB, modelC, map_dimensions=(193, 126), patch_size=(224, 224)):
        self.images_dir = images_dir
        self.save_dir = save_dir
        self.original_dir = original_dir
        self.image_size = (0,0)
        self.patch_size = patch_size
        self.current_image = ""
        self.modelA = self.load_model(modelA)
        self.modelB = self.load_model(modelB)
        self.modelC = self.load_model(modelC)
        self.average_mapA = None
        self.count_mapA = None
        self.average_mapB = None
        self.count_mapB = None
        self.average_mapC = None
        self.count_mapC = None
        self.map_dimensions = map_dimensions
        
    def load_model(self, model_dir):
        model = tf.keras.models.load_model(model_dir)
        model.trainable = False
        
        return model
    
    def new_image(self, image_name):
        self.current_image = image_name
        self.current_image_dir = os.path.join(self.images_dir, self.current_image)
        imagePath = os.path.join(self.original_dir, self.current_image+".JPG")
        print(imagePath)
        if os.path.exists(imagePath):
            im = PIL.Image.open(imagePath)
        elif os.path.exists(os.path.join(self.original_dir, self.current_image+".Jpeg")):
            im = PIL.Image.open(os.path.join(self.original_dir, image_name+".Jpeg"))
        
        self.image_size = im.size
        
    
    def update_matrices(self, coords, predictionsA, predictionsB, predictionsC):
        
        
        for coord, predA, predB, predC in zip(coords, predictionsA, predictionsB, predictionsC):
            
            new_x = int(coord[0]/30)
            new_y = int(coord[1]/30)
            
            self.average_mapA[new_x][new_y]+=predA[0]
            
            self.average_mapB[new_x][new_y]+=predB[0]
            
            self.average_mapC[new_x][new_y]+=predC[0]
        
    
    def create_matrices(self):
        self.average_mapA = np.zeros(self.map_dimensions)
        
        self.average_mapB = np.zeros(self.map_dimensions)
        
        self.average_mapC = np.zeros(self.map_dimensions)
        
    
    
    def create_map(self, image_name):
        self.new_image(image_name)
        self.create_matrices()
        
        batch_size = 32
        
        mapIter = MapIterator(batch_size, self.current_image_dir, return_name=False)
        
        #Do prediction with predict outside of loop to avoid initializing tensorflow several times
        predictionsA = self.modelA.predict(mapIter, batch_size, steps=len(mapIter), verbose=1)
        predictionsB = self.modelB.predict(mapIter, batch_size, steps=len(mapIter), verbose=1)
        predictionsC = self.modelC.predict(mapIter, batch_size, steps=len(mapIter), verbose=1)
            
        file_names = mapIter.files
            
        coords = [tuple(map(int, re.findall('\_[0-9]+_[0-9]+_[0-9]+\.', file)[0].split("_")[1:3])) for file in file_names]
            
        self.update_matrices(coords, predictionsA, predictionsB, predictionsC)
        
        self.save_matrices()
        
    
    def save_matrices(self):
        
        save_folder = os.path.join(self.save_dir, self.current_image)
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        save_matrixA = self.average_mapA*255
        save_matrixB = self.average_mapB*255
        save_matrixC = self.average_mapC*255
        
        
        imgA = Image.fromarray(save_matrixA)
        imgB = Image.fromarray(save_matrixB)
        imgC = Image.fromarray(save_matrixC)
        
        imgA = imgA.convert("L")
        imgB = imgB.convert("L")
        imgC = imgC.convert("L")
        
        imgA.save(os.path.join(save_folder, "predictionA.png"))
        imgB.save(os.path.join(save_folder, "predictionB.png"))
        imgC.save(os.path.join(save_folder, "predictionC.png"))
        
            

if __name__=="__main__":
    patch_size = (224, 224)
    base_dir = "E:\\Coding\\Dataset"
    original_dir = "images_test"
    map_folder = "Test_Map"
    map_save_folder = "Test_Map_Heat"
    model_dir = "E:\\Coding\\MDMGAGAICSCS\\ML"
    
    images_dir = os.path.join(base_dir, map_folder)
    save_dir = os.path.join(base_dir, map_save_folder)
    
    networkA = 'network_A_1_best_weights.h5'
    networkB = 'network_B_1_best_weights.h5'
    networkC = 'network_C_1_best_weights.h5'
    
    networkA_p = os.path.join(model_dir, networkA)
    networkB_p = os.path.join(model_dir, networkB)
    networkC_p = os.path.join(model_dir, networkC)
    
    mapper = TileMapper(images_dir, save_dir, os.path.join(base_dir, original_dir), networkA_p, networkB_p, networkC_p)
    
    for folder in os.listdir(images_dir):
        if os.path.exists(os.path.join(save_dir, folder)):
            continue
        
        
        mapper.create_map(folder)