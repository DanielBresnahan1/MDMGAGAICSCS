# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:07:07 2022

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


"""@package docstring
This class represents

"""

class HeatMaper:
    
    def __init__(self, images_dir, save_dir, original_dir, modelA, modelB, modelC, patch_size=(224, 224)):
        self.images_dir = images_dir
        self.save_dir = save_dir
        self.patch_size = patch_size
        self.original_dir = original_dir
        self.image_size = (0,0)
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
        
    def create_matrices(self):
        self.average_mapA = np.zeros(self.image_size)
        self.count_mapA = np.zeros(self.image_size)
        
        self.average_mapB = np.zeros(self.image_size)
        self.count_mapB = np.zeros(self.image_size)
        
        self.average_mapC = np.zeros(self.image_size)
        self.count_mapC = np.zeros(self.image_size)
    
    def update_matrices(self, coords, predictionsA, predictionsB, predictionsC):
        
        for coord, predA, predB, predC in zip(coords, predictionsA, predictionsB, predictionsC):
        
            self.average_mapA[coord[0]:coord[0]+self.patch_size[0]+1, coord[1]:coord[1]+self.patch_size[1]+1]+=predA
            self.count_mapA[coord[0]:coord[0]+self.patch_size[0]+1, coord[1]:coord[1]+self.patch_size[1]+1]+=1
            
            self.average_mapB[coord[0]:coord[0]+self.patch_size[0]+1, coord[1]:coord[1]+self.patch_size[1]+1]+=predB
            self.count_mapB[coord[0]:coord[0]+self.patch_size[0]+1, coord[1]:coord[1]+self.patch_size[1]+1]+=1
            
            self.average_mapC[coord[0]:coord[0]+self.patch_size[0]+1, coord[1]:coord[1]+self.patch_size[1]+1]+=predC
            self.count_mapC[coord[0]:coord[0]+self.patch_size[0]+1, coord[1]:coord[1]+self.patch_size[1]+1]+=1
        
    
    def create_map(self, image_name):
        self.new_image(image_name)
        self.create_matrices()
        
        batch_size = 32
        
        mapIter = MapIterator(batch_size, self.current_image_dir, return_name=False)
            
        coords = [tuple(map(int, re.findall('\_[0-9]+_[0-9]+_[0-9]+\.', file)[0].split("_")[1:3])) for file in mapIter.files]
        
        predictionsA = self.modelA.predict(mapIter, steps=len(mapIter), batch_size=batch_size, verbose=1)
        predictionsB = self.modelB.predict(mapIter, steps=len(mapIter), batch_size=batch_size, verbose=1)
        predictionsC = self.modelC.predict(mapIter, steps=len(mapIter), batch_size=batch_size, verbose=1)
            
        self.update_matrices(coords, predictionsA, predictionsB, predictionsC)
        
        exit()
        
        self.save_matrices()
        
    
    def save_matrices(self):
        
        save_folder = os.path.join(self.save_dir, self.current_image)
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        #if part of image wasnt touches do to rounding, add 1 to base
        self.count_mapA[self.count_mapA == 0] += 1
        self.count_mapB[self.count_mapB == 0] += 1
        self.count_mapC[self.count_mapC == 0] += 1
        
        
        save_matrixA = np.divide(self.average_mapA, self.count_mapA)
        save_matrixB = np.divide(self.average_mapB, self.count_mapB)
        save_matrixC = np.divide(self.average_mapC, self.count_mapC)
        
        save_matrixA = save_matrixA*255
        save_matrixB = save_matrixB*255
        save_matrixC = save_matrixC*255
        
        
        imgA = Image.fromarray(save_matrixA)
        imgB = Image.fromarray(save_matrixB)
        imgC = Image.fromarray(save_matrixC)
        
        imgA = imgA.convert("L")
        imgB = imgB.convert("L")
        imgC = imgC.convert("L")
        
        imgA.save(os.path.join(save_folder, "modelA.png"))
        imgB.save(os.path.join(save_folder, "modelB.png"))
        imgC.save(os.path.join(save_folder, "modelC.png"))
        
            
            

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
    
    mapper = HeatMaper(images_dir, save_dir, os.path.join(base_dir, original_dir), networkA_p, networkB_p, networkC_p)
    
    
    
    for folder in os.listdir(images_dir):     
        
        mapper.create_map(folder)
        
    
    # new_arrA = np.load("savea.npy")
    # new_arrA = new_arrA*255
    
    # new_arrB = np.load("saveb.npy")
    # new_arrB = new_arrB*255
    
    # new_arrC = np.load("savec.npy")
    # new_arrC = new_arrB*255
    
    
    # imgA = Image.fromarray(new_arrA)
    # imgB = Image.fromarray(new_arrB)
    # imgC = Image.fromarray(new_arrC)
    
    # imgA = imgA.convert("L")
    # imgB = imgB.convert("L")
    # imgC = imgC.convert("L")
    
    # imgA.save(os.path.join("E:\\Coding\\Dataset\\Train_Map_Classifications\\DSC00027", "modelA.png"))
    # imgB.save(os.path.join("E:\\Coding\\Dataset\\Train_Map_Classifications\\DSC00027", "modelB.png"))
    # imgC.save(os.path.join("E:\\Coding\\Dataset\\Train_Map_Classifications\\DSC00027", "modelC.png"))
    
    
    
    