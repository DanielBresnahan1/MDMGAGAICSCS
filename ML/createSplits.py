# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:05:37 2022

@author: Danie

CreateSplits is responsible for looping through the images_dir and images_csv, and generating the
validation split, and the test_split. The validation split is a sample of 100 random positive and 100
random negative images; the test split is generated likewise. 
"""

import os
import csv
import random

def find_unique(annotations: str) -> dict:
    """
    find_unique will loop through the annotations csv, and generate a dictionary which contains
    the unique images, and all associated rows. This is neccesary as the csv sometimes contains
    multiple entries for an image, specifically when an image has more than 1 lesion present. 

    Parameters
    ----------
    annotations : str
        path to csv file containing annotations.

    Returns
    -------
    dict
        Dictionary containing unique images. Of the form
        {Unique_image: [[row1],[row2],...,[rowN]]}

    """

    
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


def dict_split(train: dict, test: dict, val: dict) -> list:
    """
    dict_split takes in a train dictionary, that should contain all images, and two empty dictionaries
    (test, and val) which should be empty. It will then iterator over train and remove 100 random negative
    and 1000 random positive samples and place them in test (and likewise for val)
    
    Additionally, it will return a list of the indexes of the keys of all the randomly selected samples 
    in the list removed_set. this is so that these samples may be removed from train. 

    Parameters
    ----------
    train : dict
        Dictionary containing all images.
    test : dict
        Empty dictionary that will contain test images.
    val : dict
        Empty dictionary that will contain val iamges.

    Returns
    -------
    list
        A list containing the index of the keys populated into the val and test dicionaries.

    """
    
    removed_set = []
    val_negative = 0
    test_negative = 0
    val_positive = 0
    test_positive = 0
    keys = list(train.keys())
    
    random.shuffle(keys)
    
    for idx, key in enumerate(keys):

        coord_list = [int(c) for c in train[key][0][1:5]]
        if all(coord_list):
            
            if test_positive < 100:
                removed_set.append(keys.index(key))
                test_positive += 1
                test.update({key:train[key]})
            elif val_positive < 100:
                removed_set.append(keys.index(key))
                val_positive += 1
                val.update({key:train[key]})
        
        elif not all(coord_list):
            
            if test_negative < 100:
                removed_set.append(keys.index(key))
                test_negative += 1
                test.update({key:train[key]})
            elif val_negative < 100:
                removed_set.append(keys.index(key))
                val_negative += 1
                val.update({key:train[key]})
        
        if val_positive >= 100 and test_positive >= 100 and val_negative >= 100 and test_negative >= 100:
            break
        
                
    
    return removed_set

def clean_dict(removed_set: list, dictionary: dict):
    """
    clean_dict takes in a list of indexes of samples occuring in a dictionary as keys, and a dictionary
    the keys appearing in removed_set will be removed from the dictionary.
    

    Parameters
    ----------
    removed_set : list
        list of indexes of samples.
    dictionary : dict
        The train dict to remove from.

    Returns
    -------
    None.

    """
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



    
    
    
