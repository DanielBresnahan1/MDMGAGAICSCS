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
