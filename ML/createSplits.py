# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:05:37 2022

@author: Danie
"""

import os
import csv
import random

def find_unique(annotations):
    
    return_dict = {}
    
    with open(annotations, "r") as f:
        reader = csv.reader(f)
        
        next(reader)
        
        for line in reader:
            if line[0] in return_dict.keys():
                return_dict[line[0]].append(line)
            else:
                return_dict.update({line[0]:[line]})
    
    return return_dict


def dict_split(train, test, val):
    
    removed_set = []
    for i in range(int(len(train.keys())*0.33)):
        
        while True:    
            key = random.randint(0, len(train.keys())-1)
            key_s = list(train.keys())[key]
            
            if key not in removed_set:
                removed_set.append(key)
                
                if i%2:
                    test.update({key_s:train[key_s]})
                else:
                    val.update({key_s:train[key_s]})
                
                break
    
    return removed_set

def clean_dict(removed_set, dictionary):
    removed_set.sort()
    removed_set.reverse()
    
    for key in removed_set:
        key_s = list(dictionary.keys())[key]
        
        dictionary.pop(key_s)
        

def write_csv(dict_annotations, file_name, orig_dir="", dest_dir=""):
    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f)
        
        for key in dict_annotations.keys():
            if len(orig_dir):
                file = os.path.join(orig_dir, key)
                os.rename(file, os.path.join(dest_dir, key))
            for line in dict_annotations[key]:
                writer.writerow(line)
    
        

if __name__=="__main__":
    
    
    base_dir = "E:\Coding\Dataset"

    annotations = os.path.join(base_dir, "annotations_handheld.csv")

    pic_locations = os.path.join(base_dir, "images_handheld")

    test_dir = os.path.join(base_dir, "images_test")
    val_dir = os.path.join(base_dir, "images_validation")

    train = {}
    test = {}
    val = {}
    
    train = find_unique(annotations)
    removed_set = dict_split(train, test, val)
    clean_dict(removed_set, train)
    
    train_annotations = os.path.join(base_dir, "annotations_train.csv")
    test_annotations = os.path.join(base_dir, "annotations_test.csv")
    val_annotations = os.path.join(base_dir, "annotations_val.csv")
    
    write_csv(train, train_annotations)
    write_csv(test, test_annotations, orig_dir=pic_locations, dest_dir=test_dir)
    write_csv(val, val_annotations, orig_dir=pic_locations, dest_dir=val_dir)



    
    
    
