# Ag-AI
Capstone Project

The goal of this project is to build trust in farmers on the idea of artificial intelligence. We plan to do this by allowing the user to interact with an active learning system, providing information to a machine learning algorithm and seeing how their input changes the accuracy of the algorithm.

The CapstoneMain is used to run this project. Images folder is used to hold all images. Images must be separated by classification by adding a folder that holds all images of the classification. There must be a path to each file in master to run this program.

The user will interact with a web-based user interface, where they’ll be asked to label pictures of corn as either healthy or unhealthy. 


--Random Forest Retraining--
To retrain the random forest algorithm, first create a random forest model by classifying images presented by webapp. 
Once enough confidence is reached, results screen for the model will show. 
Scroll down until one is able to see the test images the model classified itself, not the training set provided by the user. 
Use checkboxes to disagree with pictures labeled by model, then scroll down to bottom of page and select "retrain" button. 

--Image Patching--
To patch the images to create usefull data for the Neural Network, you need to use the ImagePatching.py File. This file contains a class that will handle all the logic and implementation for iamge striding, and tiling. To utilize, simply instantiate the class with a few important parameters 1. The directory you with to save the new files. 2. The size of the patches to generate (reccomend 224x224) and 3. The size of the original image. 

With the class instantiated, call the class method patch (patcher.patch) and supply the location of the image, and the coordinates of the lession in the form (x1, y2, x2, y2). If the image is negative (ie. does not contain a lesion) supply (0,0,0,0) instead. 

It is possible to use colate data if you wish to rerun the expiriment entirely, however, do note it is a hard coded script, so you will have to change each path name to represent the associated path on your system. 
