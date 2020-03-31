#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from project import data_upload, model_upload, prediction, training_model
import pandas as pd
from hypothesis import given
import hypothesis.strategies as st

@given(path = st.text())
def test_data_upload_link(path):
    path.join("http")
    print(path)
    assert (data_upload(path) == pd.DataFrame()).all(None)
@given(path = st.text())
def test_data_upload(path):
    print(path)
    data_upload(path)

@given(path = st.text())
def test_model_upload_link(path):
    path.join("http")
    print(path)
    assert model_upload(path) == 404
@given(path = st.text())    
def test_model_upload(path):
    print(path)
    try:
        model_upload(path)
    except Exception:
        return 0

    
@given(mod=st.text(),dat=st.text(),n=st.integers(),perf=st.integers(0,1))
def test_prediction(mod,dat,perf,n):
    assert prediction(dat,mod,perf,n) == 404
@given(dat=st.text(),n=st.integers(),ne=st.integers(),b=st.integers(), a = st.floats() )
def test_training(dat,n,ne,b,a):
    assert training_model(dat,n,[ne,b,a]) == 4
    
