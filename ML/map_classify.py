# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:07:07 2022

@author: Danie
"""

import numpy as np
import os
import tensorflow as tf
import PIL
import Image

class HeatMaper:
    
    def __init__(self, images_dir, save_dir, patch_size, original_dir, modelA, modelB, modelC):
        self.images_dir = images_dir
        self.save_dir = save_dir
        self.patch_size = patch_size
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
        if os.path.exists(model_dir):
            model = tf.keras.models.load_model(model_dir)
        
        return model
    
    def new_image(self, image_name):
        self.current_image = image_name
        imagePath = os.path.join(self.images_dir, self.current_image+".JPG")
        if os.path.exists(imagePath):
            im = PIL.Image.open(imagePath)
        elif os.path.exists(os.path.join(self.images_dir, self.current_image+".JPEG")):
            im = PIL.Image.open(os.path.join(self.images_dir, image_name+".JPEG"))
        
        self.image_size = im.size
        
        self.average_mapA = np.zeros(self.image_size)
        self.count_mapA = np.zeros(self.image_size)
        
        self.average_mapB = np.zeros(self.image_size)
        self.count_mapB = np.zeros(self.image_size)
        
        self.average_mapC = np.zeros(self.image_size)
        self.count_mapC = np.zeros(self.image_size)
    
    def update_matrices(self, x, y, predA, predB, predC):
        self.average_mapA[x:self.patch_size[0]+1, y:self.patch_size[1]+1]+=predA
        self.count_mapA[x:self.patch_size[0]+1, y:self.patch_size[1]+1]+=1
        
        self.average_mapB[x:self.patch_size[0]+1, y:self.patch_size[1]+1]+=predB
        self.count_mapB[x:self.patch_size[0]+1, y:self.patch_size[1]+1]+=1
        
        self.average_mapC[x:self.patch_size[0]+1, y:self.patch_size[1]+1]+=predC
        self.count_mapC[x:self.patch_size[0]+1, y:self.patch_size[1]+1]+=1
        
    
    def create_map(self, image_name):
        self.new_image(image_name)
    
        for file in os.listdir(self.image_dir):
            file_parts = file.split("_")
            x_coord = int(file_parts[1])
            y_coord = int(file_parts[2])
            
            pic = PIL.Image.open(os.path.join(self.images_dir, self.current_image, file))
            
            pic = np.array(pic)
            x=pic.reshape(3,224,224)
            
            predictionA = np.array(self.modelA(x, training=False))
            predictionB = np.array(self.modelB(x, training=False))
            predictionC = np.array(self.modelC(x, training=False))
            
            self.update_matrices(x_coord, y_coord, predictionA, predictionB, predictionC)
        
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
        
        imgA = Image.fromarray(save_matrixA)
        imgB = Image.fromarray(save_matrixB)
        imgC = Image.fromarray(save_matrixC)
        
        imgA.save(os.path.join(save_folder, "modelA.png"))
        imgB.save(os.path.join(save_folder, "modelB.png"))
        imgC.save(os.path.join(save_folder, "modelC.png"))
        
            
            

if __name__=="__main__":
    patch_size = (224, 244)
    base_dir = "E:\\Coding\\Dataset"
    test_image = "DSC00027"
    original_dir = "images_validation"
    map_folder = "Train_Map"
    map_save_folder = "Train_Map_Classifications"
    original_dir = os.path.join()
    
    images_dir = os.path.join(base_dir, map_folder)
    save_dir = os.path.join(base_dir, map_save_folder)
    
    networkA = 'network_A_1'
    networkB = 'network_B_1'
    networkC = 'network_C_1'
    
    mapper = HeatMapper(images_dir, save_dir, patch_size, )
    
    
    
    