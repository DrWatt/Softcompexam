#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:43:13 2019

@author: marco
"""
#%%
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
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from joblib import load, dump
import requests
from hypothesis import given
import hypothesis.strategies as st
#from scipy.stats import poisson
#from scipy.optimize import curve_fit
#from scipy.misc import factorial
#%%
seed = 5
np.random.seed(seed)
def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=7, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

encoder = LabelEncoder()
encoder.fit([-3,-2,-1,0,1,2,3])

#%%

def model_upload(modpath):
    if("http" in modpath):
        print("Downloading Model")
        try:    
            mod = requests.get(modpath)
            mod.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            #while(requests.exceptions.RequestException):
                #print("Try again")
                #modpath =input()
                #try:
                  #  mod = requests.get(modpath)
                 #   mod.raise_for_status()
                #except requests.exceptions.RequestException:
                 #   continue
                #else:
                  #  break
            return None
        with open("model.joblib","wb") as o:
            o.write(mod.content)
        modpath = "model.joblib"
    print("Loading Model from Disk")
    try:
        estimator = load(modpath)
    except Exception:
        print("Error: File not found or empty")
        return None
    return estimator

def data_upload(datapath):
    if("http" in datapath):
        print("Downloading Dataset")
        try:
            dataset = requests.get(datapath)
            dataset.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            return pd.DataFrame()
            
        with open("dataset.csv","wb") as o:
            o.write(dataset.content)
        datapath = "dataset.csv"
    print("Loading Dataset from Disk")
    try:
        dataset = pd.read_csv(datapath,header=0)
    except Exception:
        print("Error: File not found or empty")
        return pd.DataFrame()
    ##data = dataset.values
    ##X = data[:200,1:8]
    return dataset
#%%

def nn_performance(model, datatest):
    test = prediction(datatest,model,performance=1)
    distance = (test[0]-test[1])
    #binrange=[-0.5,0.5,1.5,2.5,3.5]
    distplot = plt.hist(distance)#,bins=binrange)
    print(100*np.unique(distance,return_counts=True)[1]/distance.size)
    plt.hist
    return distance

def prediction(datapath,modelpath,performance=0,NSamples=0):

    dataset = data_upload(datapath)
    if dataset.empty: 
        return 1 
    else: 
        data = dataset.values
    estimator = model_upload(modelpath)
    if estimator == None:
        return 2
    if NSamples== 0:
        X = data[:,1:8]
    elif NSamples > dataset.size:
        print("Sample requested is greater than dataset provided: using whole dataset")
        X = data[:,1:8]
    else:
         X = data[:NSamples,1:8]
    print("Thinking about BXs..")
    pred=encoder.inverse_transform(estimator.predict(np.array(X)))
    
    if performance:
        labels = data[15000:,0]
        return [pred,labels]

    return pred
#%%
def training_data_loader(datapath,NSample=0):
    dataset = data_upload(datapath)
    if dataset.empty:
        return 1
    else:
        data = dataset.values
    if NSample== 0:
        X = data[:,1:8]
        BX = data[:,0]
    elif NSample > dataset.size:
        print("Sample requested is greater than dataset provided: using whole dataset")
        X = data[:,1:8]
        BX = data[:,0]
    else:
        X = data[:NSample,1:8]
        BX = data[:1000,0]
    encoder = LabelEncoder()
    encoder.fit(BX)
    encoded_BX = encoder.transform(BX)
    transformed_BX = np_utils.to_categorical(encoded_BX)
    return [X,transformed_BX]

def training_model(datapath,NSample=0,Nepochs=100,batch=10):
    try:
        dataset,encoded_labels = training_data_loader(datapath,NSample)
    except Exception:
        return 3
    estimator = KerasClassifier(build_fn=baseline_model, Nepochs=200, batch_size=batch, verbose=2)
    history = estimator.fit(dataset, encoded_labels, Nepochs=300, batch_size=batch,verbose=2)
    out=dump(estimator,"model2.joblib")
    return out[0]

def cross_validation(modelpath,datapath):
    X,Y=training_data_loader(datapath)
    estimator = model_upload(modelpath)
    if estimator == None:
        return 2
    kf=KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator,X,Y,cv=kf,verbose=1,n_jobs=4)
    return results.mean()

training_model('',0,0,0)
############################################
    #%%
@given(path = st.text())
def test_data_upload_link(path):
    path.join("http")
    print(path)
    assert (data_upload(path) == pd.DataFrame()).all(None)
@given(path = st.text())
def test_data_upload(path):
    print(path)
    data_upload(path)
def test_data_upload_working():
    path="https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv"
    assert (data_upload(path) == data_upload("datatree.csv")).all(None)
@given(path = st.text())
def test_model_upload_link(path):
    path.join("http")
    print(path)
    assert model_upload(path) == None
@given(path = st.text())    
def test_model_upload(path):
    print(path)
    try:
        model_upload(path)
    except Exception:
        return 0

def test_model_upload_working():
    path="https://github.com/DrWatt/softcomp/blob/master/asd.joblib?raw=true"
    assert (model_upload(path) == model_upload("asd.joblib"))
    
@given(mod=st.text(),dat=st.text(),n=st.integers(),perf=st.integers(0,1))
def test_prediction(mod,dat,perf,n):
    assert prediction(dat,mod,perf,n)
@given(dat=st.text(),n=st.integers(),ne=st.integers(),b=st.integers())
def test_training(dat,n,ne,b):
    training_model(dat,n,ne,b)
    

    
    
#%%
    
    
    