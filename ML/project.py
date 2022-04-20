# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:44:45 2022

@author: Danie

DEPRICATED FILE DO NOT USE

Old biolerplate work for when stage 3 classification was just going to be selecting a set of representative
points create by ModelsA B and C. Has terrible accuracy and should absolutley not be used. 
"""

import tensorflow as tf
import numpy as np
import matplotlib as plt
import os
import csv
from test_iterator import TestIterator
import pandas as pd
import plotly.express as px
import plotly
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import svm
import random


def getLabel(base_dir, csvf, test_image):
    
    label = 0
    
    with open(os.path.join(base_dir, csvf), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == test_image+".JPG" or row[0] == test_image+".JPEG":
                label = row[1]
                break
    
    return label

if __name__=="__main__":
    
    base_dir = "E:\\Coding\\Dataset"
    test_dir = "images_test"
    label_csv = "test_labels.csv"
    
    networkA = 'network_A_1'
    networkB = 'network_B_1'
    networkC = 'network_C_1'
    
    base_dir = "E:\\Coding\\Dataset"
    
    batch_size = 32
    csvf = "test_labels.csv"
    
    
    svc = svm.SVC()
    
    totdf = pd.DataFrame()
    
    
    if os.path.exists(os.path.join(os.getcwd(),networkA+'_best_weights.h5')):
        modelA = tf.keras.models.load_model(networkA+'_best_weights.h5')
    
    if os.path.exists(os.path.join(os.getcwd(),networkB+'_best_weights.h5')):
        modelB = tf.keras.models.load_model(networkB+'_best_weights.h5')
    
    if os.path.exists(os.path.join(os.getcwd(),networkC+'_best_weights.h5')):
        modelC = tf.keras.models.load_model(networkC+'_best_weights.h5')
    
    file_list = os.listdir(os.path.join(base_dir, "Test"))
    random.shuffle(file_list)
    
    for index, folder in tqdm(enumerate(file_list)):
        
        if index > 10:
            break
        
        image_dir = os.path.join(base_dir, "Test", folder)
        
        predictionsDict = {"File": [], "x": [], "y": [], "z": [], "label": []}
        
        label = getLabel(base_dir, csvf, folder)
        
        testIter = TestIterator(batch_size, label, image_dir, return_name=True)
        
        for i in range(len(testIter)):
            
            x, y, name = testIter.__getitem__(0)
            
            predictionsA = np.array(modelA(x, training=False))
            predictionsB = np.array(modelB(x, training=False))
            predictionsC = np.array(modelC(x, training=False))
            
            for idx, pred in enumerate(zip(predictionsA, predictionsB, predictionsC, y)):
                # if pred[0] >= 0.3 or pred[1] >= 0.3 or pred[2] >= 0.3:
                predictionsDict["File"].append(name[idx])
                predictionsDict["x"].append(pred[0])
                predictionsDict["y"].append(pred[1])
                predictionsDict["z"].append(pred[2])
                predictionsDict["label"].append(pred[3])
                
                
            
        
        df=pd.DataFrame(data=predictionsDict)
        
        
        if((df.max().drop('File') > 0.5).any()):
            df.drop(df[(df.x < 0.5) & (df.y < 0.5) & (df.z < 0.5)].index, inplace=True)
            df = df.head(n=4)
        else:
            df.drop(df[(df.x > 0.5) & (df.y > 0.5) & (df.z > 0.5)].index, inplace=True)
            df = df.head(n=4)

        totdf = totdf.append(df)

    fig = plt.figure(figsize=(12, 9))
    ax = Axes3D(fig)

    Y = [value.item() for value in totdf['y'].to_numpy()]
    X = [value.item() for value in totdf['x'].to_numpy()]
    Z = [value.item() for value in totdf['z'].to_numpy()]
    label = totdf['label'].tolist()
    
    
    coords = list(zip(X,Y,Z))
    
    svc.fit(coords, label)
    
    ax.scatter(X, Y, Z)
    
    plt.show()
    
   
        
        
        
    
        
    
    
    
    
        
    