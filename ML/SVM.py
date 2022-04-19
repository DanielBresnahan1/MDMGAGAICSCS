# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:17:40 2022

@author: Danie

Support vector machine for spacial classification
"""

    

import os
import csv
from sklearn.ensemble import RandomForestClassifier

def csv_to_list(csv_file):
    
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        
        return list(reader)
        
def train(X, y):
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    return clf

def fit(clf, X, y):
    
    predictions = clf.predict(X)
    
    num_correct = 0
    for index, prediction in enumerate(predictions):
        if prediction == y[index]:
            num_correct += 1
    
    accuracy = num_correct/len(y)
    
    print("accuracy: {}/{} = {}".format(num_correct, len(y), accuracy))
    
    return predictions
    
    
if __name__=="__main__":
    
    base_dir = "E:\\Coding\\Dataset"
    predictions_train = "classifications_train.csv"
    predictions_test = "classifications.csv"
    
    class_list = csv_to_list(os.path.join(base_dir, predictions_train))[1:]
    coords = [[float(x), float(y), float(z)] for _, x, y, z, label in class_list]
    labels = [int(label) for _, _, _, _, label in class_list]
    
    
    clf = train(coords, labels)
    
    class_listT = csv_to_list(os.path.join(base_dir, predictions_test))[1:]
    coordsT = [[float(x), float(y), float(z)] for _, x, y, z, label in class_listT]
    labelsT = [int(label) for _, _, _, _, label in class_listT]
    
    predictions = fit(clf, coordsT, labelsT)
    
    with open(os.path.join(base_dir, predictions_test), "w", newline='') as f:
        writer = csv.writer(f)
        
        for index, row in enumerate(class_list):
            if index == 0:
                writer.writerow(row)
            else:
                row.append(predictions[index])
                writer.writerow(row)
    
    
    
    
    
    


    
    
    
    
    


