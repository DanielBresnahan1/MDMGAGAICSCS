"""
Original File Author: Chad DeChant
Github: https://github.com/chaddech

Updated for python 3, specifically the implimentation of the __next__ method.
Original File Structure was different from this projects implementation, so the _init__ 
method needed to be changed. Specifically, there are no subfolders in the 'Positive' 
and 'Negative' directories
"""


from tensorflow.keras.utils import Sequence
import numpy as np
import os
import PIL
import math

class BatchesIterator(Sequence):
    def __init__(self, batch_size, no_lesion_folder_path, 
              lesion_folder_path, lesion=True, no_lesion=True, return_dir=False):

        self.batch_size = batch_size
        self.files = []
        self.return_dir = return_dir
        if lesion == True:
            for file in os.listdir(lesion_folder_path):
                self.files.append((os.path.join(lesion_folder_path, file), 1))
        if no_lesion == True:
            for file in os.listdir(no_lesion_folder_path):
                self.files.append((os.path.join(no_lesion_folder_path, file), 0))
        
        np.random.shuffle(self.files)
        
    def __len__(self):
        return math.ceil(len(self.files)/self.batch_size)
    
    def on_epoch_end(self):
        np.random.shuffle(self.files)
    

    def __getitem__(self, index):

        if len(self.files)-(index*self.batch_size) <= self.batch_size:
            this_batch_size = len(self.files) - (index*self.batch_size)
        else:
            this_batch_size = self.batch_size

        batch_x = np.zeros((this_batch_size,3,224,224), dtype='uint8')
		
        batch_y = np.zeros((this_batch_size), dtype='uint8')

        batch_file_names = []
        for x in range(this_batch_size):
            this_file_name = self.files[(index*self.batch_size)+x][0]
            pic = PIL.Image.open(this_file_name)
            pic = np.array(pic)
            batch_x[x]=pic.reshape(3,224,224)
            batch_y[x]=self.files[(index*self.batch_size)+x][1]
            batch_file_names.append(this_file_name) 
            
        if self.return_dir:
            return batch_x, batch_y, batch_file_names
        else:
            return batch_x, batch_y
    


if __name__=="__main__":
    
    base_dir = "Z:\Coding\Dataset\Train"
    
    batchIter = BatchesIterator(2, os.path.join(base_dir, "Negative"), os.path.join(base_dir, "Positive"))
    
    for index, tup in enumerate(batchIter):
        print(index)
