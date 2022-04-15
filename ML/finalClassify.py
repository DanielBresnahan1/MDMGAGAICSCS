# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:12:01 2022

@author: Danie
"""
import os
import tensorflow as tf
from test_iterator import TestIterator
from pathlib import Path
import csv


if __name__=="__main__":
    
    
    base_dir = "E:\\Coding\\Dataset"
    image_dir = os.path.join(base_dir, "Test_Map_Heat")
    batch_size = 32
    csvf = "test_labels.csv"
    save_csvf = "classifications.csv"
    
    csv_columns = ['Image','PredictionA','PredictionB','PredictionC',"True_Label"]
    
    csv_dir = os.path.join(base_dir, csvf)
    save_csv_dir = os.path.join(base_dir, save_csvf)
    
    batches = TestIterator(batch_size, csv_dir, image_dir, preserve_image=True)
    
    file_name = 'network_T_1'
    
    model_path = os.path.join(os.getcwd(),file_name+'_best_weights.h5')

    model = tf.keras.models.load_model(file_name+'_best_weights.h5')
    
    model.summary()
    
    predictions = model.predict(x=batches, batch_size=batch_size, steps=len(batches), verbose=1)
    predictions = predictions.flatten()
    
    print(predictions)
    
    results = {}
    
    for index, file in enumerate(batches.files):
        A_B_C = Path(file[0]).parts[-1]
        image_name = Path(file[0]).parts[-2]
        
        if not image_name in results:
            results.update({image_name: [0, 0, 0, file[1]]})
        
        if A_B_C=="predictionA.png":
            results[image_name][0] = predictions[index]
        elif A_B_C=="predictionB.png":
            results[image_name][1] = predictions[index]
        elif A_B_C=="predictionC.png":
            results[image_name][2] = predictions[index]
    
    
    with open(os.path.join(base_dir, save_csvf), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        for key in results.keys():
            writer.writerow([key]+results[key])
        
        
    