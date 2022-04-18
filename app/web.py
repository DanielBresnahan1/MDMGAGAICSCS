# -*- coding:utf-8 -*-
"""@package web
This method is responsible for the inner workings of the different web pages in this application.
"""
import io
from flask import Flask
from flask import render_template, flash, redirect, url_for, session, request, jsonify
from app import app
from app.DataPreprocessing import DataPreprocessing
from app.ML_Class import Active_ML_Model, AL_Encoder, ML_Model
from app.SamplingMethods import lowestPercentage
from app.forms import LabelForm
from flask_bootstrap import Bootstrap
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
import boto3
import pickle
from io import BytesIO, StringIO
from PIL import Image
import PIL
import requests
import csv
import random  # TODO: delete me later because i only exist for the architecture walkthrough

bootstrap = Bootstrap(app)


def getData():
    """
    Gets and returns the csvOut.csv as a DataFrame.

    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """
    s3 = boto3.client('s3')
    path = 's3://cornimagesbucket/csvOut.csv'

    data = pd.read_csv(path, index_col=0, header=None)
    data.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']

    data_mod = data.astype({'8': 'int32', '9': 'int32', '10': 'int32', '12': 'int32', '14': 'int32'})
    return data_mod.iloc[:, :-1]

def getImage(image):
    """
    Gets and returns the csvOut.csv as a DataFrame.

    Returns
    -------
    data : Pandas DataFrame
        The data that contains the features for each image.
    """
    
    url = "https://mdmgcapstone.s3.amazonaws.com/"
    image = "predictionA.png"
    a = 0
    b = 0
    c = 0
    mapA = 0
    mapB = 0
    mapC = 0
    label = 0
    with open('eggs.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if(row[0] == image):
                a = row[1]
                b = row[2]
                c = row[3]
                mapA = row[4]
                mapB = row[5]
                mapC = row[6]
                label = row[7]
    
                responseA = requests.get(url + mapA)
                responseB = requests.get(url + mapB)
                responseC = requests.get(url + mapC)
                imgA = Image.open(BytesIO(responseA.content))
                imgB = Image.open(BytesIO(responseB.content))
                imgC = Image.open(BytesIO(responseC.content))
                break
    
    return [a,b,c,imgA,imgB,imgC,label]




def createMLModel(data):
    """
    Prepares the training set and creates a machine learning model using the training set.

    Parameters
    ----------
    data : Pandas DataFrame
        The data that contains the features for each image

    Returns
    -------
    ml_model : ML_Model class object
        ml_model created from the training set.
    train_img_names : String
        The names of the images.
    """
    train_img_names, train_img_label = list(zip(*session['train']))
    train_set = data.loc[train_img_names, :]
    train_set['y_value'] = train_img_label
    ml_model = ML_Model(train_set, RandomForestClassifier(), DataPreprocessing(True))
    return ml_model, train_img_names


def renderLabel(form):
    """
    prepairs a render_template to show the label.html web page.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    queue = session['queue']
    img = queue.pop()
    session['queue'] = queue
    return render_template(url_for('label'), form=form, picture=img, confidence=session['confidence'])


def initializeAL(form, confidence_break=.7):
    """
    Initializes the active learning model and sets up the webpage with everything needed to run the application.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    confidence_break : number
        How confident the model is.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    preprocess = DataPreprocessing(True)
    ml_classifier = RandomForestClassifier()
    data = getData()
    al_model = Active_ML_Model(data, ml_classifier, preprocess)

    session['confidence'] = 0
    session['confidence_break'] = confidence_break
    session['labels'] = []
    session['sample_idx'] = list(al_model.sample.index.values)
    session['test'] = list(al_model.test.index.values)
    session['train'] = al_model.train
    session['model'] = True
    session['queue'] = list(al_model.sample.index.values)

    return renderLabel(form)


def getNextSetOfImages(form, sampling_method):
    """
    Uses a sampling method to get the next set of images needed to be labeled.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html
    sampling_method : SamplingMethods Function
        function that returns the queue and the new test set that does not contain the queue.

    Returns
    -------
    render_template : flask function
        renders the label.html webpage.
    """
    data = getData()
    ml_model, train_img_names = createMLModel(data)
    test_set = data[data.index.isin(train_img_names) == False]

    session['sample_idx'], session['test'] = sampling_method(ml_model, test_set, 5)
    session['queue'] = session['sample_idx'].copy()

    return renderLabel(form)


def prepairResults(form):
    """
    Creates the new machine learning model and gets the confidence of the machine learning model.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying label.html

    Returns
    -------
    render_template : flask function
        renders the appropriate webpage based on new confidence score.
    """
    session['labels'].append(form.choice.data)
    session['sample'] = tuple(zip(session['sample_idx'], session['labels']))

    if session['train'] != None:
        session['train'] = session['train'] + session['sample']
    else:
        session['train'] = session['sample']

    data = getData()
    ml_model, train_img_names = createMLModel(data)

    session['confidence'] = np.mean(ml_model.K_fold())
    session['labels'] = []

    if session['confidence'] < session['confidence_break']:
        health_pic, blight_pic = ml_model.infoForProgress(train_img_names)
        return render_template('intermediate.html', form=form,
                               confidence="{:.2%}".format(round(session['confidence'], 4)), health_user=health_pic,
                               blight_user=blight_pic, healthNum_user=len(health_pic), blightNum_user=len(blight_pic))
    else:
        test_set = data.loc[session['test'], :]
        health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(
            train_img_names, test_set)
        return render_template('final.html', form=form, confidence="{:.2%}".format(round(session['confidence'], 4)),
                               health_user=health_pic_user, blight_user=blight_pic_user,
                               healthNum_user=len(health_pic_user), blightNum_user=len(blight_pic_user),
                               health_test=health_pic, unhealth_test=blight_pic, healthyNum=len(health_pic),
                               unhealthyNum=len(blight_pic), healthyPct="{:.2%}".format(
                len(health_pic) / (200 - (len(health_pic_user) + len(blight_pic_user)))), unhealthyPct="{:.2%}".format(
                len(blight_pic) / (200 - (len(health_pic_user) + len(blight_pic_user)))), h_prob=health_pic_prob,
                               b_prob=blight_pic_prob)


def renderMVMLabel(form):
    """
    prepairs a render_template to show the man_vs_machine.html web page.

    Parameters
    ----------
    form : LabelForm class object
        form to be used when displaying man_vs_machine.html

    Returns
    -------
    render_template : flask function
        renders the man_vs_machine.html webpage.
    """
    if session['mvm_pics']:
        # img = session['mvm_pics'].pop()  # why is this not popping the strings off the list?
        new_mvm_pics = session['mvm_pics']
        img = new_mvm_pics.pop()
        session['mvm_pics'] = new_mvm_pics
        return render_template('label.html', form=form, picture=img)
    else:
        session.pop('mvm_pics')
        return mvm_results()


@app.route("/", methods=['GET'])
@app.route("/index.html", methods=['GET'])
def home():
    """
    Operates the root (/) and index(index.html) web pages.
    """
    session.permanent = True
    #session.pop('model', None)
    getImage("test")
    return render_template('index.html')


@app.route("/label.html", methods=['GET', 'POST'])
def label():
    """
    Operates the label(label.html) web page.
    """
    form = LabelForm()
    if 'model' not in session:  # Start
        return initializeAL(form, .7)
      
    if 'queue' not in session:
        session['queue'] = []

    if 'labels' not in session:
        session['labels'] = []

    if session['queue'] == [] and session['labels'] == []: # Need more pictures
        return getNextSetOfImages(form, lowestPercentage)

    elif form.is_submitted() and session['queue'] == []:  # Finished Labeling
        return prepairResults(form)

    elif form.is_submitted() and session['queue'] != []:  # Still gathering labels
        session['labels'].append(form.choice.data)
        return renderLabel(form)

    return initializeAL(form, .7)


@app.route("/intermediate.html", methods=['GET'])
def intermediate():
    """
    Operates the intermediate(intermediate.html) web page.
    """
    return render_template('intermediate.html')


@app.route("/final.html", methods=['GET'])
def final():
    """
    Operates the final(final.html) web page.
    """
    return render_template('final.html')


@app.route("/feedback/<h_list>/<u_list>/<h_conf_list>/<u_conf_list>", methods=['GET'])
def feedback(h_list, u_list, h_conf_list, u_conf_list):
    """
    Operates the feedback(feedback.html) web page.
    """
    h_feedback_result = list(h_list.split(","))
    u_feedback_result = list(u_list.split(","))
    h_conf_result = list(h_conf_list.split(","))
    u_conf_result = list(u_conf_list.split(","))
    h_length = len(h_feedback_result)
    u_length = len(u_feedback_result)

    return render_template('feedback.html', healthy_list=h_feedback_result, unhealthy_list=u_feedback_result,
                           healthy_conf_list=h_conf_result, unhealthy_conf_list=u_conf_result, h_list_length=h_length,
                           u_list_length=u_length)



@app.route("/retrain.html", methods=['GET'])
@app.route("/retrain/<h_disagree_list>/<u_disagree_list>", methods=['GET'])
def retrain(h_disagree_list, u_disagree_list):
    """
    Retrain the random forest algorithm with the images the user already classified
    and with the images the user disagrees with from the current model.

    Parameters
    ----------
    h_disagree_list : list of image names
        the images that the model classified as healthy,
        but the user believes are actually unhealthy

    u_disagree_list : list of image names
        the images that the model classified as unhealthy,
        but the user believes are actually healthy
    """
    new_healthy_images = list(u_disagree_list.split(","))
    new_unhealthy_images = list(h_disagree_list.split(","))
    if new_healthy_images[0] != 'null':
        for image_name in new_healthy_images:
            session['train'] = session['train'] + ((image_name, 'H'),)
            session['test'].remove(image_name)
    if new_unhealthy_images[0] != 'null':
        for image_name in new_unhealthy_images:
            session['train'] = session['train'] + ((image_name, 'B'),)
            session['test'].remove(image_name)

    # make a model in here
    data = getData()
    ml_model, train_img_names = createMLModel(data)

    session['confidence'] = np.mean(ml_model.K_fold())
    # https://cornimagesbucket.s3.us-east-2.amazonaws.com/images_compressed/imageName is this reliable for images?

    # which test_set to use? same?
    test_set = data[data.index.isin(train_img_names) == False]
    test_set = data.loc[session['test'], :]
    health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(
        train_img_names, test_set)
    return render_template('retrain.html', confidence="{:.2%}".format(round(session['confidence'], 4)),
                           health_user=health_pic_user, blight_user=blight_pic_user,
                           healthNum_user=len(health_pic_user), blightNum_user=len(blight_pic_user),
                           health_test=health_pic, unhealth_test=blight_pic, healthyNum=len(health_pic),
                           unhealthyNum=len(blight_pic), healthyPct="{:.2%}".format(
            len(health_pic) / (200 - (len(health_pic_user) + len(blight_pic_user)))), unhealthyPct="{:.2%}".format(
            len(blight_pic) / (200 - (len(health_pic_user) + len(blight_pic_user)))), h_prob=health_pic_prob,
                           b_prob=blight_pic_prob)


# create new form? because i don't understand current form?
@app.route("/man_vs_machine.html", methods=['GET', 'POST'])
def man_vs_machine():
    form = LabelForm()
    if request.method == 'POST':  # label selected for a corn picture.
        session['mvm_choices'] = session['mvm_choices'] + [(form.corn_picture.data, form.choice.data)]
        return renderMVMLabel(form)
    else:  # a GET request, clicking the button to play MVM from the home page, starting a new game.
        session['mvm_choices'] = []
        session['mvm_pics'] = ['DSC00027.JPG', 'DSC00076.JPG', 'DSC00029.JPG', 'DSC00030.JPG', 'DSC00031.JPG',
                               'DSC00033.JPG', 'DSC00025.JPG', 'DSC00037.JPG', 'DSC00038.JPG', 'DSC00036.JPG']
        return renderMVMLabel(form)


# jank results page for the man vs machine game.
@app.route("/mvm_results.html", methods=['GET'])
def mvm_results():
    # jank literal copy of session['mvm_pictures']
    session['temp_pics'] = ['DSC00027.JPG', 'DSC00076.JPG', 'DSC00029.JPG', 'DSC00030.JPG', 'DSC00031.JPG',
                            'DSC00033.JPG', 'DSC00025.JPG', 'DSC00037.JPG', 'DSC00038.JPG', 'DSC00036.JPG']
    session['temp_pics'].reverse()  # reversed order because session['mvm_pictures'] pictures obtained with pop().

    # jank distribution of corn pictures between healthy & unhealthy, represents how machine classification. random
    # jank determination of true labels for corn pictures. also random
    machine_choices = []
    true_labels = []
    for picture_label in session['mvm_choices']:
        if random.randint(0, 1) == 1:  # machine selects picture as healthy
            machine_choices.append((picture_label[0], 'H'))
        else:  # machine selects picture as unhealthy
            machine_choices.append((picture_label[0], 'B'))
        if random.randint(0, 1) == 1:  # picture is given healthy true label
            true_labels.append((picture_label[0], 'H'))
        else:  # picture is given unhealthy true label
            true_labels.append((picture_label[0], 'B'))

    print(f"mvm choices {session['mvm_choices']}")
    print(f"mac choices {machine_choices}")
    print(f"true labels {true_labels}")  # actually random

    user_correct = 0
    machine_correct = 0
    for i in range(0, len(true_labels)):
        if session['mvm_choices'][i][1] == true_labels[i][1]:
            user_correct += 1
        if machine_choices[i][1] == true_labels[i][1]:
            machine_correct += 1
    user_accuracy = user_correct / len(true_labels)
    machine_accuracy = machine_correct / len(true_labels)

    return render_template('mvm_results.html',
                           user_healthy_pics=[picture_label[0] for picture_label in session['mvm_choices']
                                              if picture_label[1] == 'H'],
                           user_unhealthy_pics=[picture_label[0] for picture_label in session['mvm_choices']
                                                if picture_label[1] == 'B'],
                           machine_healthy_pics=[picture_label[0] for picture_label in machine_choices
                                                 if picture_label[1] == 'H'],
                           machine_unhealthy_pics=[picture_label[0] for picture_label in machine_choices
                                                   if picture_label[1] == 'B'],
                           user_accuracy=user_accuracy,
                           machine_accuracy=machine_accuracy, )
    health_pic_user, blight_pic_user, health_pic, blight_pic, health_pic_prob, blight_pic_prob = ml_model.infoForResults(train_img_names, test_set)
    return render_template('retrain.html', confidence = "{:.2%}".format(round(session['confidence'],4)), health_user = health_pic_user, blight_user = blight_pic_user, healthNum_user = len(health_pic_user), blightNum_user = len(blight_pic_user), health_test = health_pic, unhealth_test = blight_pic, healthyNum = len(health_pic), unhealthyNum = len(blight_pic), healthyPct = "{:.2%}".format(len(health_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), unhealthyPct = "{:.2%}".format(len(blight_pic)/(200-(len(health_pic_user)+len(blight_pic_user)))), h_prob = health_pic_prob, b_prob = blight_pic_prob)


@app.route("/restart.html", methods=['GET'])
def restart():
    session.pop('model', None)
    return redirect(url_for('home'))

#app.run( host='127.0.0.1', port=5000, debug='True', use_reloader = False)

