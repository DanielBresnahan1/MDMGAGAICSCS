# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import PIL
import random
import os

class ImagePatcher:
    
    def __init__(self, saveDir: str, patchSize: tuple, majorAxisDif=4,
                 stride=10, imageSize=(6000, 4000)):
        """
        

        Parameters
        ----------
        saveDir : str
            Location to save images
        patchSize : tuple
            The size of the dimensions of the patch.
            (width, height)
        majorAxisDif : int, optional
            The random distance from the major axis of the lesion to generate 
            the patch from.
            ie. size 15 means that the center of the patch will be a random 
            amount within 15 pixels of the lesion
        rotBoosting : Boolean, optional
            IF a random rotation should be applied to boost the total number 
            of patches from a single image. The default is True.
        stride : int, optional
            how far to stride along the major axis of the legion. The default is 10.

        Returns
        -------
        None.

        """
        self.majorAxisDif = majorAxisDif
        self.saveDir = saveDir
        self.patchSize = patchSize
        self.rotBoosting = True
        self.stride = stride
        self.imageSize = imageSize
    
    def set_save_dir(self, savedir):
        """
        Setter to change save Directory, primarly used for test image patching, as all images need to be split cubicly

        Parameters
        ----------
        savedir : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.saveDir = savedir
    
    def patch_path(self, x1, y1, x2, y2):
        """
        This generates a list of coordinates, that represent the location on each
        image, in which a path should be generated. should only be called on 
        images in which a lesion exists.

        Parameters
        ----------
        x1 : int
            Lesion x1
        y1 : int
            Lesion y1
        x2 : int
            Lesion x2
        y2 : int
            Lesion y2

        Returns
        -------
        None.

        """
        
        slope = 1
        
        if x2-x1:
            slope = (y2-y1)/(x2-x1)
    
        coords = []
        
        yshift=0
        for x in range(x1, x2, self.stride):
            yshift += self.stride*slope
            coords.append((x, y1+yshift))
        
        return coords
    
    
    def patch_grid(self):
        """
        Generate a list of coordinates, that represents the center of each patch.
        Used for Negative pictures, such that a grid of the entire image is generated. 

        Returns
        -------
        None.

        """
        difx = self.imageSize[0] % self.patchSize[0]
        dify = self.imageSize[1] % self.patchSize[1]
        
        coords = []
        
        for x in range(0, self.imageSize[0]-difx-self.patchSize[0], self.patchSize[0]):
            for y in range(0, self.imageSize[1]-dify-self.patchSize[1], self.patchSize[1]):
                coords.append((x+self.patchSize[0]/2, y+self.patchSize[1]/2))
                
        return coords
    
    
    def patch(self, imagePath: str, lessionCoords: tuple):
        """
        Will generate a number of patches from an image, if the image contains
        a lession, it will stride along the major axis of the NLCB and create patches.
        If the image is negative, it will instead break up the images into a grid.
        

        Parameters
        ----------
        imagePath : String
            The path to save an image to
        lessionCoords : Tuple
            Tuples that contains the coordinates for the lession, of the form
            (x1, y1, x2, y2)

        Returns
        -------
        None.

        """
        
        #determine weather or not a lession exists, if 0, there is no lession
        lessionSum = sum(lessionCoords)
        
        Folder = ""
        
        if lessionSum:
            Folder = "Positive"
            patch_centers = self.patch_path(*lessionCoords)
        else:
            Folder = "Negative"
            patch_centers = self.patch_grid()
        
        im = PIL.Image.open(imagePath)
        
        print(os.path.basename(imagePath))
        
        image_name = os.path.basename(imagePath).split(".")[0]
        
        for x, y in patch_centers:
            
            x = x + random.randint(-1 * self.majorAxisDif/2, self.majorAxisDif/2)
            y = y + random.randint(-1 * self.majorAxisDif/2, self.majorAxisDif/2)
            
            crop = (x - self.patchSize[0]/2, y - self.patchSize[1]/2,
                                    x + self.patchSize[0]/2, y +  self.patchSize[1]/2)
        
            if crop[0] > 0 and crop[0] < self.imageSize[0] and crop[1] > 0 and crop[1] < self.imageSize[1]:
                im_crop = im.crop(crop)
                
                for a in range(0, 360, 90):
                    
                    rot = im_crop.rotate(a, PIL.Image.BICUBIC, expand=1)
                    
                    if not rot.size == self.patchSize:
                        rot = rot.resize(self.patchSize)
                    
                    rot.save(os.path.join(self.saveDir, Folder, 
                                               "{}_{}_{}_{}.jpg".format(image_name, int(crop[0]), int(crop[1]), a)))
        


if __name__=="__main__":
    curDir = os.getcwd()
    image = "DSC00025T.jpg"
    image_dir = os.path.join(curDir, image)
    
    im = PIL.Image.open(image)
    
    print(im.size)
    
    patcher = ImagePatcher(os.path.join(curDir, "save"), (224, 224), imageSize=(6000, 4000))
    patcher.patch(image_dir, (1864,2064,2864,1648))
    
    image = "DSC00027.jpg"
    image_dir = os.path.join(curDir, image)
    
    patcher.patch(image_dir, (0,0,0,0))
        
        
    
    
        
    
        
    
        