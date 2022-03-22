"""
Original File Author: Chad DeChant
Github: https://github.com/chaddech

Updated for python 3, specifically the implimentation of the __next__ method.
Original File Structure was different from this projects implementation, so the _init__ 
method needed to be changed. Specifically, there are no subfolders in the 'Positive' 
and 'Negative' directories
"""


try:
    from collections.abc import Iterator
except ImportError:
    from collections import Iterator  
import numpy as np
import os
import PIL

class BatchesIterator(Iterator):
    def __init__(self, batch_size, no_lesion_folder_path, 
              lesion_folder_path, lesion=True, no_lesion=True):

        self.batch_size = batch_size
        self.batch_start_index = 0
        self.files = []
        self.need_to_shuffle = True
        if lesion == True:
            for file in os.listdir(lesion_folder_path):
                self.files.append((os.path.join(lesion_folder_path, file), 1))
        if no_lesion == True:
            for file in os.listdir(no_lesion_folder_path):
                self.files.append((os.path.join(no_lesion_folder_path, file), 0))


    def __next__(self):
        if self.need_to_shuffle:
            np.random.shuffle(self.files)
            self.need_to_shuffle = False
            self.batch_start_index = 0


        if len(self.files)-self.batch_start_index <= self.batch_size:
            self.need_to_shuffle = True
            this_batch_size = len(self.files) - self.batch_start_index
        else:
            this_batch_size = self.batch_size

        batch_x = np.zeros((this_batch_size,3,224,224), dtype='uint8')
		
        batch_y = np.zeros((this_batch_size), dtype='uint8')

        batch_file_names = []
        for x in range(this_batch_size):
            this_file_name = self.files[self.batch_start_index+x][0]
            pic = PIL.Image.open(this_file_name)
            pic = np.array(pic)
            batch_x[x]=pic.reshape(3,224,224)
            batch_y[x]=self.files[self.batch_start_index+x][1]
            batch_file_names.append(this_file_name) 
        self.batch_start_index+=self.batch_size
        return batch_x, batch_y, batch_file_names
    


if __name__=="__main__":
    
    base_dir = "Z:\Coding\Dataset\Train"
    
    batchIter = BatchesIterator(2, os.path.join(base_dir, "Negative"), os.path.join(base_dir, "Positive"))
    
    for index, tup in enumerate(batchIter):
        print(index)
