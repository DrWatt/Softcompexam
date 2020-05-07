#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import project
from project import seed
#from build_NN import baseline_model
import pandas as pd
import numpy as np
import os
from hypothesis import given
import hypothesis.strategies as st
import argparse
import hashlib
np.random.seed(seed)
pretrperf = 0.10859411489957964

    
def test_set_param_NN():
    a = project.training_model("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", par = [5,5,0.2])
    b = project.model_upload("KerasNN_Model.joblib")
    os.remove("KerasNN_Model.joblib")
    assert [b.get_params()['epochs'],b.get_params()['batch_size']] == [5,5]



def test_prediction_xgb_zeros():
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
    a = project.xgtrain("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1",xgparams)
    c = project.prediction("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", "XGBoost_Model.joblib")
    b = np.zeros_like(c)
    os.remove("XGBoost_Model.joblib")
    assert np.equal(c,b).all()

def test_prediction_nn_zeros():
    a = project.training_model("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1")
    c = project.prediction("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", "KerasNN_Model.joblib")
    b = np.zeros_like(c)
    print(b)
    print(c)
    print(project.preprocessing("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1"))
    os.remove("KerasNN_Model.joblib")
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
    
    
def test_xg_train():
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
    a = project.xgtrain("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1",xgparams,20)
    hash_md5 = hashlib.md5()
    with open("XGBoost_Model.joblib","rb") as f:
        hash_md5.update(f.read())
    
    assert hash_md5.hexdigest() == '35eba9d25828097909063314fac16688'