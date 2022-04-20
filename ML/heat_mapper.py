# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:07:07 2022

@author: Danie

This file contains definition for the HeatMapper class, which is responsible for generating
the heat maps by utilizing pretrained models
"""

import numpy as np
import os
import tensorflow as tf
import PIL
from PIL import Image
from tqdm import tqdm
from mapIterator import MapIterator
import re
from tensorflow.keras import Sequential


class HeatMaper:
    
    def __init__(self, images_dir: str, save_dir: str, original_dir: str, modelA: str, 
                 modelB: str, modelC: str, dimensions=(193,126)):
        """
        The constructor for HeatMapper, also runs self.load_model

        Parameters
        ----------
        images_dir : str
            path to directory containing patched images to generate heat maps of.
        save_dir : str
            path to directory to save heat maps too.
        original_dir : str
            path to directory containing original images, prioir to patching.
        modelA : str
            Path to pretrained CNN model A.
        modelB : str
            Path to pretrained CNN model B.
        modelC : str
            Path to pretrained CNN model C.
        patch_size : TUPLE, optional
            The size of the patches, Default is (224, 224).

        Returns
        -------
        None.

        """
        self.images_dir = images_dir
        self.save_dir = save_dir
        self.original_dir = original_dir
        self.image_size = (0,0)
        self.current_image = ""
        self.modelA = self.load_model(modelA)
        self.modelB = self.load_model(modelB)
        self.modelC = self.load_model(modelC)
        self.prediction_mapA = None
        self.prediction_mapB = None
        self.prediction_mapC = None
        self.dimensions = dimensions
        self.rotate = False
    
    def load_model(self, model_dir: str) -> Sequential:
        """
        function to load CNN model into memory,
        Heat Mapper expects the model to a Sequential CNN accepting 
        images of size self.patch_size, and product an output
        of size 1.

        Parameters
        ----------
        model_dir : str
            path to CNN model.

        Returns
        -------
        Sequential
            Keras sequential model for prediction.

        """
        model = tf.keras.models.load_model(model_dir)
        model.trainable = False
        
        return model
    
    def new_image(self, image_name:str):
        """
        Takes in the name of a new image to process. Sets relavant variables 
        like cur_path, and current_image

        Parameters
        ----------
        image_name : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.current_image = image_name
        self.current_image_dir = os.path.join(self.images_dir, self.current_image)
        imagePath = os.path.join(self.original_dir, self.current_image+".JPG")
        if os.path.exists(imagePath):
            im = PIL.Image.open(imagePath)
        elif os.path.exists(os.path.join(self.original_dir, self.current_image+".Jpeg")):
            im = PIL.Image.open(os.path.join(self.original_dir, image_name+".Jpeg"))
        
        self.image_size = im.size
        
        
        
    def create_matrices(self):
        """
        A function to initialize the matrices in size image_size. called after new_image

        Returns
        -------
        None.

        """
        
        
        self.prediction_mapA = np.zeros(self.dimensions)
        
        self.prediction_mapB = np.zeros(self.dimensions)
        
        self.prediction_mapC = np.zeros(self.dimensions)
    
    def update_matrices(self, coords, predictionsA, predictionsB, predictionsC):
        
        for coord, predA, predB, predC in zip(coords, predictionsA, predictionsB, predictionsC):
            
            x = int(coord[0]/30)
            y = int(coord[1]/30)
            
            # print("X: {}, Y: {}".format(x, y))
            # print("prediction: {}".format(predA))
            
            self.prediction_mapA[x][y] = predA[0]
            
            self.prediction_mapB[x][y] = predB[0]
            
            self.prediction_mapC[x][y] = predC[0]
        
    
    def create_map(self, image_name):
        self.new_image(image_name)
        self.create_matrices()
        
        batch_size = 32
        
        mapIter = MapIterator(batch_size, self.current_image_dir, return_name=False)
            
        coords = [tuple(map(int, re.findall('\_[0-9]+_[0-9]+_[0-9]+\.', file)[0].split("_")[1:3])) for file in mapIter.files]
        
        # coords = sorted(coords , key=lambda k: [k[1], k[0]])
        
        predictionsA = self.modelA.predict(mapIter, steps=len(mapIter), batch_size=batch_size, verbose=1)
        predictionsB = self.modelB.predict(mapIter, steps=len(mapIter), batch_size=batch_size, verbose=1)
        predictionsC = self.modelC.predict(mapIter, steps=len(mapIter), batch_size=batch_size, verbose=1)
        
        self.update_matrices(coords, predictionsA, predictionsB, predictionsC)
        
        self.save_matrices()
        
    
    def save_matrices(self):
        
        save_folder = os.path.join(self.save_dir, self.current_image)
        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
            
        #if part of image wasnt touches do to rounding, add 1 to base
        
        save_matrixA = self.prediction_mapA*255
        save_matrixB = self.prediction_mapB*255
        save_matrixC = self.prediction_mapC*255
        
        imgA = Image.fromarray(save_matrixA)
        imgB = Image.fromarray(save_matrixB)
        imgC = Image.fromarray(save_matrixC)
        
        if self.rotate:
            imgA = imgA.rotate(90, resample=PIL.Image.BICUBIC)
            imgB = imgB.rotate(90, resample=PIL.Image.BICUBIC)
            imgC = imgC.rotate(90, resample=PIL.Image.BICUBIC)
        
        imgA = imgA.convert("L")
        imgB = imgB.convert("L")
        imgC = imgC.convert("L")
        
        imgA.save(os.path.join(save_folder, "modelA.png"))
        imgB.save(os.path.join(save_folder, "modelB.png"))
        imgC.save(os.path.join(save_folder, "modelC.png"))
        
            
            

if __name__=="__main__":
    patch_size = (224, 224)
    base_dir = "E:\\Coding\\Dataset\\new_heat_mapper"
    map_folder = "images"
    map_save_folder = "images_heat"
    original_dir = "E:\\Coding\\Dataset\\new_heat_mapper\\images_original"
    model_dir = "E:\\Coding\\MDMGAGAICSCS\\ML"
    
    
    images_dir = os.path.join(base_dir, map_folder)
    save_dir = os.path.join(base_dir, map_save_folder)
    
    networkA = 'network_A_1_best_weights.h5'
    networkB = 'network_B_1_best_weights.h5'
    networkC = 'network_C_1_best_weights.h5'
    
    networkA_p = os.path.join(model_dir, networkA)
    networkB_p = os.path.join(model_dir, networkB)
    networkC_p = os.path.join(model_dir, networkC)
    
    mapper = HeatMaper(images_dir, save_dir, original_dir, networkA_p, networkB_p, networkC_p)
    
    
    
    for folder in os.listdir(images_dir):
        if os.path.exists(os.path.join(save_dir, folder)):
            continue
        
        
        mapper.create_map(folder)
        
    
    
    