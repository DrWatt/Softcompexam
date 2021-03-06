#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import project
from project import seed,fold
#from build_NN import baseline_model
import pandas as pd
import numpy as np
import os
import hashlib
from math import ceil
import requests
from subprocess import check_output
import json
np.random.seed(seed)
pretrperf = 0.10859411489957964

xgparams = {'max_depth':5,
                            'eta':0.3,
                            'subsample':0.82,
                            'colsample_bytree': 0.68,
                            'eval_metric': ['merror','mlogloss'],
                            'silent':0,
                            'objective':'multi:softmax',
                            'num_class':len(project.encoder.classes_),
                            'seed':project.seed,
                            'num_parallel_tree': 5
                            #'tree_method': 'gpu_hist'
                            }
    
def test_set_param_NN():
    a = project.training_model("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", par = [5,5,0.2])
    print(len(a))
    assert len(a) == 5


def test_prediction_xgb_zeros():

    a = project.xgtrain("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1",xgparams)
    c = project.prediction("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", fold+"/XGBoost_Model.joblib")
    b = np.zeros_like(c)
    os.remove(fold+"/XGBoost_Model.joblib")
    assert np.equal(c,b).all()

def test_prediction_nn_zeros():
    a = project.training_model("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1")
    c = project.prediction("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", fold+"/KerasNN_Model.h5")
    b = np.zeros_like(c)
    os.remove(fold+"/KerasNN_Model.h5")
    assert np.equal(c,b).all()

def test_consistency_inference_xgb():
    b = project.prediction("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv", "Pretrained_models/XGBoost_Model.joblib")
    c = project.prediction("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv", "Pretrained_models/XGBoost_Model.joblib")
    
    assert np.equal(b,c).all()
        
def test_model_upload():
    c = project.model_upload("https://www.dropbox.com/s/yhfwutwu6nyj345/XGBoost_Model.joblib?dl=1")
    assert c.best_score == 0.024466

def test_data_upload():
    a = project.data_upload("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv")
    assert not a.empty
    print("Data succesfully downloaded and loaded into memory")
    
def test_preprocessing():
    a = project.preprocessing("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv")
    assert a.iloc[:,2:].le(1).all().all()
    
    
    
def test_nn_performance():
    a = project.nn_performance("Pretrained_models/XGBoost_Model.joblib","https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv")
    assert a == pretrperf

def test_train_data_load():
    a = project.training_data_loader("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv",0)
    assert np.size(a[0],0) == 4282
    a = project.training_data_loader("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv",5000)
    assert np.size(a[0],0) == 4282
    a = project.training_data_loader("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv",10)    
    assert np.size(a[0],0) == 10
    
    assert np.equal(np.count_nonzero(a[1],1),1).all()
    assert np.size(a[1],1) == 9
    
    
    
# def test_X_val():
    
def test_xg_data():
    a,b= project.xg_data_loader("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1")
    assert a.feature_names == ['bx', 'phi', 'phiB', 'wheel', 'sector', 'station', 'quality']
    assert b.feature_names == ['bx', 'phi', 'phiB', 'wheel', 'sector', 'station', 'quality']
    
    assert ceil((a.num_row()+b.num_row())*0.3) == b.num_row()
    
def test_xg_train():
    
    a = project.xgtrain("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1",xgparams,20)
    b = project.model_upload(fold+"/XGBoost_Model.joblib")
    c = project.model_upload("https://www.dropbox.com/s/yhfwutwu6nyj345/XGBoost_Model.joblib?dl=1")
    os.remove(fold+"/XGBoost_Model.joblib")
    os.remove(fold+"/model.joblib")
    assert b.best_score == c.best_score
    

def test_xg_train_set_param():
    a = project.xgtrain("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1",xgparams,20)
  
    hash_md5 = hashlib.md5()
    with open(fold+"/XGBoost_Model.joblib","rb") as f:
        hash_md5.update(f.read())
    h = hash_md5.hexdigest()
    
    xgparams['eta'] = 0.5
    a = project.xgtrain("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1",xgparams,20)

    hash_md5 = hashlib.md5()
    with open(fold+"/XGBoost_Model.joblib","rb") as f:
        hash_md5.update(f.read())
    hh = hash_md5.hexdigest()
    
    assert h != hh
    
    
def test_xg_save():
    
    a = project.model_upload("https://www.dropbox.com/s/yhfwutwu6nyj345/XGBoost_Model.joblib?dl=1")
    b = requests.get("https://www.dropbox.com/s/gib9if1rsd4yprp/evres.json?dl=1").json()
    c = project.xg_save_model(a,b)
    
    assert c.head(10).equals(pd.read_csv("https://www.dropbox.com/s/o7003cftkgyeoef/ress.csv?dl=1",index_col=0).head(10))
    
    
    