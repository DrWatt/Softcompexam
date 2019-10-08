#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:43:13 2019

@author: marco
"""
#%%
import time
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from joblib import load, dump
import requests
from hypothesis import given
import hypothesis.strategies as st
#from scipy.stats import poisson
#from scipy.optimize import curve_fit
#from scipy.misc import factorial

#xgboost
import xgboost as xgb

#%%
seed = 1563
np.random.seed(seed)
def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=7, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
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
        print("Entries ", len(dataset))
    except Exception:
        print("Error: File not found or empty")
        return pd.DataFrame()
    ##data = dataset.values
    ##X = data[:200,1:8]
    return dataset
#%%

def nn_performance(modelpath, datatest):
    test = prediction(datatest,modelpath,performance=1)
    distance = (test[0]-test[1])
    accuracy = np.count_nonzero(distance)/len(test[0])
    #binrange=[-0.5,0.5,1.5,2.5,3.5]
    #distplot = plt.hist(distance)#,bins=binrange)
    #print(100*np.unique(distance,return_counts=True)[1]/distance.size)
    return accuracy

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
        labels = data[:len(X[:,0]),0]
        return [pred,labels]

    return pred
#%%
def training_data_loader(datapath,NSample=None):
    dataset = data_upload(datapath)
    if dataset.empty:
        return 1
    else:
        data = dataset.values
    if NSample== None or NSample == 0:
        X = data[:,1:8]
        BX = data[:,0]
    elif NSample > dataset.size:
        print("Sample requested is greater than dataset provided: using whole dataset")
        X = data[:,1:8]
        BX = data[:,0]
    else:
        X = data[:NSample,1:8]
        BX = data[:NSample,0]
    encoder = LabelEncoder()
    encoder.fit(BX)
    encoded_BX = encoder.transform(BX)
    #print(encoded_BX)
    transformed_BX = np_utils.to_categorical(encoded_BX)
    #print(transformed_BX)
    return [X,transformed_BX]

def training_model(datapath,NSample=0,Nepochs=30,batch=10):
    try:
        dataset,encoded_labels = training_data_loader(datapath,NSample)
    except Exception:
        return 3
    estimator = KerasClassifier(build_fn=baseline_model, epochs=Nepochs, batch_size=batch, verbose=2)
    history = estimator.fit(dataset, encoded_labels, epochs=Nepochs, batch_size=batch,verbose=2)
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


############################################
#%%
#xgboost
args = {'max_depth':20,
            'eta':0.1,
            'subsample':0.82,
            'colsample_bytree': 0.68,
            'eval_metric': 'merror',
            'silent':0,
            'objective':'multi:softmax',
            'num_class':len(encoder.classes_),
            'seed':seed
            #'num_parallel_tree':1,
            #'tree_method': 'gpu_hist'
            }
def xgtrain(datapath,datate,args={},iterations=10):
   
    dataset = data_upload(datapath)
    datatest = data_upload(datate)
    if dataset.empty or datatest.empty :
        return 1
    else:
        data = dataset.copy()
        datats = datatest.copy()
        encoder = LabelEncoder()
        encoder.fit(data['bxout'])
        #print(encoder.classes_)
    dtest = xgb.DMatrix(datats[['bx','phi','phiB','wheel','sector','station','quality']])
    print("Dataset length: ",len(data))
    

    Xtrain,Xvalid,Ytrain,Yvalid=train_test_split(data[['bx','phi','phiB','wheel','sector','station','quality']],data["bxout"],random_state=seed,test_size=0.3)
    dtrain = xgb.DMatrix(Xtrain,label=encoder.transform(Ytrain))
    dvalid = xgb.DMatrix(Xvalid,label=encoder.transform(Yvalid))
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    
    print(args)

       
    #scibst = XGBClassifier(max_depth=20,learning_rate=0.3,subsample=0.82,colsample_bytree=0.68,objective='multi:softmax',seed=seed,verbosity=2,n_jobs=-1)
    #scibst.fit(Xtrain,Ytrain,eval_metric="merror",verbose=True)
    evals_result={}    
    bst = xgb.train(args,dtrain,iterations,evallist,early_stopping_rounds=10,evals_result=evals_result)
    bst.dump_model('dump.raw.txt')
    metric = list(evals_result['eval'].keys())
    results=evals_result['eval'][metric[0]]
    #xgb.plot_importance(bst,importance_type='gain')
    #pl=plt.plot(range(len(results)),results)
    #plt.savefig("foo2.pdf",bbox_inches='tight')
    pred=bst.predict(dtest)
    #evals=bst.evals_result()
    #print(evals)
    #plt = plot
    #res = {'label' :encoder.transform(datats["bxout"]),'pred': pred}
    #sub = pd.DataFrame(res, columns=['label','pred'])
    #sub.to_csv("errrr.csv",index=False)
    #results = scibst.evals_result()
    return accuracy_score(encoder.transform(datats["bxout"]),pred)
   

#%%    
def run():
    
    time0 = time.time()
    bdtres = xgtrain("datatree4g.csv",
            "https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree2.csv",
            args,
            20)
    print("XGboost's accuracy", bdtres)    
    #model = training_model("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree2.csv",
                   #NSample=0)
    #results = 1- nn_performance(model,"https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv")
    #print("Neural Network's accuracy: ", results)
    print("XGboost's accuracy", bdtres)
    
    print("Executed in %s s" % (time.time() - time0))
#%%
    
#%%
#class method:
#    def __init__(self,type,datatrain):
        



#%%    

def seed_selector():
    acc = pd.read_csv("accu4.csv",header=0)
    seeds = pd.DataFrame(columns=['seed','accuracy'])
    for x in range(5):
        seed = acc.max(axis=0)
        acc=acc.drop(index=(seed[0].astype(int)-acc.head()['seed']))
        seeds = seeds.append(seed,ignore_index=True)
    seeds.to_csv("seeds.csv",index=False,mode='a')
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
    
    
    