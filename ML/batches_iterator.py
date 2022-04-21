"""
Original File Author: Chad DeChant
Github: https://github.com/chaddech

Updated for python 3, specifically the implimentation of the __next__ method.
Original File Structure was different from this projects implementation, so the _init__ 
method needed to be changed. Specifically, there are no subfolders in the 'Positive' 
and 'Negative' directories

__next__ -> __getitem__
    * Moved shuffeling to occur at the end of every epoch
    
__init__ 
    * removed sub_dirs in structure, so just loop through files
    
"""


from tensorflow.keras.utils import Sequence
import numpy as np
import os
import PIL
import math


class BatchesIterator(Sequence):
    """
    This class for generating an object to perform batch iteration on the saved image patches,
    for training Models A, B, and C
    
    Attributes:
        files: Contains paths and labels to both positive and negative image patches
    """
    
    def __init__(self, batch_size: int, no_lesion_folder_path: str, 
              lesion_folder_path: str, lesion=True, no_lesion=True, return_dir=False):
        """
        Constructor for BatchesIterator

        Parameters
        ----------
        batch_size : int
            Number of Images for a batch.
        no_lesion_folder_path : str
            Full Path to Negative image patches.
        lesion_folder_path : str
            Full Path to Positive image patches.
        lesion : Boolean, optional
            If the batches iterator should include positive samples. The Default is true
        no_lesion : Boolean, optional
            If the batches iterator should include negative samples. The Default is False
        return_dir : Boolean, optional
            If the batches iterator should return the directory associated with each sample,
            in the batch. The default is False.

        Returns
        -------
        None.

        """
    
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
        
    def __len__(self) -> int:
        """
        Represents the Number of batches of size self.batch_size in the dataset

        Returns
        -------
        int
            The length of the batches Iterator.

        """
        return math.ceil(len(self.files)/self.batch_size)
    
    def on_epoch_end(self):
        """
        Keras Sequence Specific function. Shuffles the dataset after each epoch

        Returns
        -------
        None.

        """
        np.random.shuffle(self.files)
    

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Keras Sequence Specific dunder, will return a batch of images.
        Will return a full batch if possible, if not will return len(files) - index*batch_size

        Parameters
        ----------
        index : int
            The index of the batch to access, should be 0<=index<len.

        Returns
        -------
        ndarray
            batch_x: The sample of batch(index), is of size (batch_size, 3, 224, 224).
        ndarray
            batch_y: The labels of batch(index), is of size (batch_size, 1)
        list
            batch_file_names: CONDITIONAL, if return_dir is set to tru, return a list of len(batch_size)
            containg the names of the file of each sample in the batch

        """

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
