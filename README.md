#Ag-AI
#Capstone Project
The goal of this project is to build trust in farmers on the idea of artificial intelligence. We plan to do this by allowing the user to interact with an active learning system, providing information to a machine learning algorithm and seeing how their input changes the accuracy of the algorithm.
The CapstoneMain is used to run this project. Images folder is used to hold all images. Images must be separated by classification by adding a folder that holds all images of the classification. There must be a path to each file in master to run this program.
The user will interact with a web-based user interface, where they’ll be asked to label pictures of corn as either healthy or unhealthy.
--Random Forest Retraining-- To retrain the random forest algorithm, first create a random forest model by classifying images presented by webapp. Once enough confidence is reached, the results screen for the model will show. Scroll down until one is able to see the test images the model classified itself, not the training set provided by the user. Use checkboxes to disagree with pictures labeled by model, then scroll down to the bottom of the page and select the "retrain" button.
--Image Patching-- To patch the images to create useful data for the Neural Network, you need to use the ImagePatching.py File. This file contains a class that will handle all the logic and implementation for image striding, and tiling. To utilize, simply instantiate the class with a few important parameters 1. The directory you wish to save the new files. 2. The size of the patches to generate (recommend 224x224) and 3. The size of the original image.
With the class instantiated, call the class method patch (patcher.patch) and supply the location of the image, and the coordinates of the lesion in the form (x1, y2, x2, y2). If the image is negative (ie. does not contain a lesion) supply (0,0,0,0) instead.
It is possible to use collate data if you wish to rerun the experiment entirely, however, do note it is a hard coded script, so you will have to change each path name to represent the associated path on your system.

#Release Notes Milestone 2:
–Random Forest Saving–
Ahead of schedule, we implemented the saving of user training of the random forest algorithm, and added additional necessary functionality to the app to back the saving, such as the reset button.
--Man versus Machine--
Play against a dummy AI by clicking the second button on the home screen.
User is then tasked with classifying a series of corn images.
Finally, the user is taken to a results page to view the statistics and the machine's selections. 

–Professional Neural Network 
Implementation of Model A is complete, which you can view in the file ModelA.py. To train simply run the script, as the implementation is located within “if __name__==”__main__”:”

–Split generation
The file createSplits is responsible for organizing the total data into 70% train 15% validate and 15% test. To call, make sure your directory is structure as 
→images_handheld
→Train
→Test
→Validate
handheld_annotations.csv
And then run the file. 

–Batches Iterator
batches_iterator.py (and batches_iterator_half_no_lesion.py, but that is broken as of yet) is responsible for handling all batch processing for training the neural networks. It will select n/2 number of positive and n/2 number of negative samples to train the neural network. Additionally, the file will correctly parse and format the image data into numpy arrays of size (3,224,224). When the iterator runs out of images to push for training, it will reshuffle the image list and continue. To use, simply importBatchesIterator from batches_iterator, and instantiate the class with parameters for batch size and directory path. Then, call next() on the saved object.  

–3D Visualization This class will be added to the webpage at a later date but currently if you manually run the python script you can see a demo of the Visualization.
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

