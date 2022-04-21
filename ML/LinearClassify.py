# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:17:40 2022

@author: Danie

Final step of classification, takes in label predictions generated from the ResNet stage 3 model predicting 
the heat maps generated in heat_mapper.

Uses a Random Forest to classify points inside an sklearn pipeline
"""

    

import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline 

def csv_to_list(csv_file: str) -> list:
    """
    Simple method that takes in a csv file and returns it as a list

    Parameters
    ----------
    csv_file : str
        Path to csv file.

    Returns
    -------
    list
        2d list representing list contents.

    """
    
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        
        return list(reader)

def getX_Y(base_dir: str, csv_name: str) -> [list, list, list]:
    """
    Get the X and Y and class list from a CSV FILE, containing the Name, PredictionA, Prediction B, 
    Prediction C, and Label asspciated with a set of images for classification

    Parameters
    ----------
    base_dir : str
        The path of the base_dir, the directory containing the csv files, and image samples
    csv_name : str
        Name of csv files of the form Name, PredictionA, PredictionB, PredictionC, Label.

    Returns
    -------
    [list, list, list]
        tuple of 3 lists, the first being the coords (PredictionA...C), the second being the labels
        1 for lession 0 for not. And the third being the name of the files.

    """
    class_list = csv_to_list(os.path.join(base_dir, predictions_train))[1:]
    coords = [[float(x), float(y), float(z)] for _, x, y, z, label in class_list]
    labels = [int(label) for _, _, _, _, label in class_list]
    
    return coords, labels, class_list
    
if __name__=="__main__":
    
    base_dir = "E:\\Coding\\Dataset"
    predictions_train = "classifications_train.csv"
    predictions_test = "classifications.csv"
    
    coords, labels, class_list = getX_Y(base_dir, predictions_train)
    
    coordsT, labelsT, class_listT = getX_Y(base_dir, predictions_test)
    
    pipe = Pipeline([('RandomForest', RandomForestClassifier())])
    
    pipe.fit(coords, labels)
    
    print("Accuracy: {}".format(pipe.score(coordsT, labelsT)))
    
    predictions = pipe.predict(coordsT, labelsT)
    
    
    with open(os.path.join(base_dir, predictions_test), "w", newline='') as f:
        writer = csv.writer(f)
        
        for index, row in enumerate(class_list):
            if index == 0:
                writer.writerow(row)
            else:
                row.append(predictions[index])
                writer.writerow(row)
    
    
    
    
    
    


    
    
    
    
    


