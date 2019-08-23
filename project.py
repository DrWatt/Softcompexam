#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:43:13 2019

@author: marco
"""
#numpy
import numpy as np
# pandas
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt
# keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from joblib import dump, load
import requests


def baseline_model():
    model = Sequential()
    model.add(Dense(5, input_dim=3, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def model_upload(modpath):
    if("http" in modpath):
        print("Downloading Model")
        mod = requests.get(modpath)
        with open("model.joblib","wb") as o:
            o.write(mod.content)
        modpath = "model.joblib"
    print("Loading Model from Disk")
    estimator = load(modpath)
    return estimator

def data_upload(datapath):
    if("http" in datapath):
        print("Downloading Dataset")
        dataset = requests.get(datapath)
        with open("dataset.csv","wb") as o:
            o.write(dataset.content)
        datapath = "dataset.csv"
    print("Loading Dataset from Disk")
    dataset = pd.read_csv(datapath,header=0)
    ##data = dataset.values
    ##X = data[:200,1:8]
    return dataset

a=2
b=4
def nn_perf(preds, true_labels):
    distance = np.sqrt((preds-true_labels)*(preds-true_labels))
    distplot = plt.hist(distance)
    return distance

def prediction(datapath,modelpath):

    data = data_upload(datapath).values
    X = data[:200,1:8]
    estimator = model_upload(modelpath)
    pred=estimator.predict(np.array(X))
    
    
    return pred

preds=prediction("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree2.csv","https://github.com/DrWatt/softcomp/blob/master/asd.joblib?raw=true")
