#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import project
from project import data_upload, model_upload, prediction, training_model,xgtrain,encoder,seed,preprocessing
import pandas as pd
import numpy as np
from hypothesis import given
import hypothesis.strategies as st
import argparse

@given(path = st.text())
def test_data_upload_link_fail(path):
    path.join("http")
    print(path)
    assert (data_upload(path) == pd.DataFrame()).all(None)
@given(path = st.text())
def test_data_upload_fail(path):
    print(path)
    data_upload(path)

@given(path = st.text())
def test_model_upload_link_fail(path):
    path.join("http")
    print(path)
    assert model_upload(path) == 404
@given(path = st.text())    
def test_model_upload_fail(path):
    print(path)
    try:
        model_upload(path)
    except Exception:
        return 0

def test_data_upload_link():
    a = data_upload("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv")
    print("Data succesfully downloaded and loaded into memory")
    print(a.head())
    return 0

@given(mod=st.text(),dat=st.text(),n=st.integers(),perf=st.integers(0,1))
def test_prediction_failure(mod,dat,perf,n):
    assert prediction(dat,mod,perf,n) == 404
@given(dat=st.text(),n=st.integers(),ne=st.integers(),b=st.integers(), a = st.floats() )
def test_training_failure(dat,n,ne,b,a):
    assert training_model(dat,n,[ne,b,a]) == 4
    
def test_set_param_NN():
    a = training_model("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", par = [5,5,0.2])
    b = model_upload("KerasNN_Model.joblib")
    assert [b.get_params()['epochs'],b.get_params()['batch_size']] == [5,5]

  
def test_prediction_xgb_zeros():
    xgparams = {'max_depth':5,
                            'eta':0.3,
                            'subsample':0.82,
                            'colsample_bytree': 0.68,
                            'eval_metric': ['merror','mlogloss'],
                            'silent':0,
                            'objective':'multi:softmax',
                            'num_class':len(encoder.classes_),
                            'seed':seed,
                            'num_parallel_tree': 5
                            #'tree_method': 'gpu_hist'
                            }
    a = xgtrain("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1",xgparams)
    c = prediction("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", "XGBoost_Model.joblib")
    b = np.zeros_like(c)
    assert np.equal(c,b).all()

def test_prediction_nn_zeros():
    a = training_model("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1")
    c = prediction("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1", "KerasNN_Model.joblib")
    b = np.zeros_like(c)
    print(b)
    print(c)
    print(preprocessing("https://www.dropbox.com/s/v4sys56bqhmdfbd/fake.csv?dl=1"))
    assert np.equal(c,b).all()

