#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:43:13 2019

@author: marco
"""
#%%
import time
import argparse
import json
import os
#numpy
import numpy as np
# pandas
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt
# Downloading and save/load from disk libraries
from joblib import load, dump
import requests
# keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, plot_model
# sklearn
from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import neighbors
# Testing libraries
from hypothesis import given
import hypothesis.strategies as st
#xgboost
import xgboost as xgb

#%%
seed = 1563
np.random.seed(seed)
encoder = LabelEncoder()
encoder.fit([-4,-3,-2,-1,0,1,2,3,4])

#Model constructor definition as needed to use scikit-learn wrapper with keras.

def baseline_model(indim=7,hidden_nodes=[8,8],outdim=9):
    model = Sequential()
    
    #Nodes of the NN.
    model.add(Dense(hidden_nodes[0], input_dim=indim, activation='relu'))
    for a in hidden_nodes[1:]:
        model.add(Dense(a,activation='relu'))
    

    # # model.add(Dense(10, activation='relu'))
    # # model.add(Dense(12, activation='relu'))
    # # model.add(Dense(10, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(outdim, activation='softmax'))
    
    # Compile model.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



#%%

# Function to load pretrained NN.
    # modpath: string argument specifing path (local or URL) of model in joblib format.

def model_upload(modpath):
    if("http" in modpath):
        print("Downloading Model")
        try:    
            # Download
            mod = requests.get(modpath)
            mod.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            return None
        # Writing model on disk.
        with open("model.joblib","wb") as o:
            o.write(mod.content)
        modpath = "model.joblib"
    print("Loading Model from Disk")
    try:
        # Uploading model from disk.
        estimator = load(modpath)
    except Exception:
        print("Error: File not found or empty")
        return None
    return estimator

# Function to load data from disk or using an URL.
    # datapath: string argument specifing path (local or URL) of data in csv format.

def data_upload(datapath):
    if("http" in datapath):
        print("Downloading Dataset")
        try:
            # Download
            dataset = requests.get(datapath)
            dataset.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            return pd.DataFrame()
        
        # Writing dataset on disk.    
        with open("dataset.csv","wb") as o:
            o.write(dataset.content)
        datapath = "dataset.csv"
    print("Loading Dataset from Disk")
    try:
        # Reading dataset and creating pandas.DataFrame.
        dataset = pd.read_csv(datapath,header=0)
        print("Entries ", len(dataset))        
        
    except Exception:
        print("Error: File not found or empty")
        return pd.DataFrame()
    return dataset

# Feature scaling: rescaling via min-max normalization.
    # datapath: string argument specifing path (local or URL) of data in csv format.

def preprocessing(datapath,cor=False):
    data = data_upload(datapath).copy()
    if cor == True:
            f = plt.figure(figsize=(19, 15))
            plt.matshow(data.drop(columns=['bxout']).corr(), fignum=f.number)
            plt.xticks(range(data.shape[1]), data.drop(columns=['bxout']).columns, fontsize=14, rotation=45)
            plt.yticks(range(data.shape[1]), data.drop(columns=['bxout']).columns, fontsize=14)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.title('Correlation Matrix', fontsize=16);
            plt.savefig("Correlation_training_set.png")
            plt.clf()
    
    # Normalization factor (min - max for every feature).
    norm = data.max() - data.min()
    
    # Subtracting for every feature its minimum.
    data2 = data - [0,0,data.min()[2],data.min()[3],data.min()[4],data.min()[5],data.min()[6],data.min()[7],0]
    
    # Normalization.
    data = data2.div([1,1,norm[2],norm[3],norm[4],norm[5],norm[6],norm[7],1])
    
    # Casting to make bx labels integers again.
    data['bxout'] = data['bxout'].astype(int)
    data['bx'] = data['bx'].astype(int)

    return data
#%%
# Function to compute classification of a dataset using a pretrained NN.
    # datapath: string argument specifing path (local or URL) of data in csv format.
    # modpath: string argument specifing path (local or URL) of model in joblib format.
    # performance: boolean arguement changing return mode: False -> return only predictions;
    #                                                      True -> return predictions and true labels if provided (for evaluating performance).
    # NSamples: int argument specifing number of entries used of the dataset. If NSamples == 0 or NSamples > data size the all dataset will be used.
    
def prediction(datapath,modelpath,performance=False,NSamples=0):
    
    # Loading dataset and preprocessing it.
    dataset = preprocessing(datapath)
    
    # Failed loading handling (empty dataset exception).
    if dataset.empty: 
        return 1 

    
    # Loading NN.
    estimator = model_upload(modelpath)
    
    # Failed loading handling.
    if estimator == None:
        return 2
    
    # Handling of number of entries argument (NSample).
    if NSamples == 0:
        X = dataset.drop(columns=['bxout','is2nd'])
    elif NSamples > dataset.size:
        print("Sample requested is greater than dataset provided: using whole dataset")
        X = dataset.drop(columns=['bxout','is2nd'])
    else:
         X = dataset.head(NSamples).drop(columns='bxout')
    print("Thinking about BXs..")
    # Actual prediction method + inverse encoding to get actual BX values.
    try:
    # if argss.xgb:
        pred=encoder.inverse_transform(estimator.predict(X))
        
    # else:
    except Exception: 
        dtest = xgb.DMatrix(X[['bx','phi','phiB','wheel','sector','station','quality']])
        pred=encoder.inverse_transform(estimator.predict(dtest).astype(int))   

        

    # condition to return also labels.
    if performance:
        labels = dataset.head(len(X.index)).get('bxout')
        return [pred,labels]

    return pred


# Function to perform a simple comparison between prediction and known labels of a test sample.
    # modelpath: string argument specifing path (local or URL) of model in joblib format.
    # datatest: string argument specifing path (local or URL) of data in csv format.

def nn_performance(modelpath, datatest):
    
    # Performing prediction
    test = prediction(datatest,modelpath,performance=True)
    
    # Definition of a 'distance' between prediction and true label.
    distance = (test[0]-test[1])
    
    # Evaluating the percentage of bad predictions.
    # It is done counting the times when the 'distance' is different from 0 and dividing by the entries in our test sample.
    accuracy = np.count_nonzero(distance)/len(test[0])
    
    # Plotting accuracy.
    #binrange=[-0.5,0.5,1.5,2.5,3.5]
    #distplot = plt.hist(distance)#,bins=binrange)
    #print(100*np.unique(distance,return_counts=True)[1]/distance.size)
    
    return accuracy


#%%
    
# Function performing one-hot encoding.
    # datapath: string argument specifing path (local or URL) of data in csv format.
    # NSamples: int argument specifing number of entries used of the dataset. If NSamples == None or NSamples > data size the all dataset will be used.

def training_data_loader(datapath,NSample=None):
    
    # Uploading preprocessed dataset.
    dataset = preprocessing(datapath,cor=True)
    if dataset.empty:
        return 1
    else:
        data = dataset.values
        
    # Handling of number of entries argument (NSample).
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
    
    # Encoding BXs to have labels from 0 to 9.
    #encoder = LabelEncoder()
    #encoder.fit(BX)
    encoded_BX = encoder.transform(BX)
    #print(encoded_BX.max())
    transformed_BX = np_utils.to_categorical(encoded_BX,9)
    #print(transformed_BX)
    return [X,transformed_BX]

# NN training function.
    # datapath: string argument specifing path (local or URL) of data in csv format.
    # NSamples: int argument specifing number of entries used of the dataset. If NSamples == 0 or NSamples > data size the all dataset will be used.
    # Nepochs: int argument specifing number of epochs the NN will be trained for passed to the NN costructor.
    # batch: int argument specifing the size of the batches used to update the weights passed to the NN costructor.
    
    #small data: batch = 8
    #medium data: batch = 30
def training_model(datapath,NSample=0,Nepochs=48,batch=30):
    
    # Loading and preparing data for training.
    try:
        dataset,encoded_labels = training_data_loader(datapath,NSample)
    except Exception:
        return 3
    
    # Keras constructor with scikit-learn API.
    estimator = KerasClassifier(build_fn=baseline_model, epochs=Nepochs, batch_size=batch, verbose=2)
    
    # Training method for our model. 
    history = estimator.fit(dataset, encoded_labels, epochs=Nepochs, batch_size=batch,verbose=2,validation_split=0.3)
    
    # Saving trained model on disk. (Only default namefile ATM)
    out=dump(estimator,"model2.joblib")
    
    plot_model(estimator.model, to_file='model.png',show_shapes=True)
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')      
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("Keras_NN_Accuracy.png")
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("Keras_NN_Loss.png")
    plt.clf()
    # Returning namefile of model in order to use the trained model in other functions e.g. only for predictions.
    return out[0]

# KFold cross validation function using scikit-learn API.
    # modelpath: string argument specifing path (local or URL) of model in joblib format.
    # datapath: string argument specifing path (local or URL) of data in csv format.

def cross_validation(modelpath,datapath):

    # Loading and preparing data for validation.
    X,Y=training_data_loader(datapath)
    
    # Loading model from disk.
    estimator = model_upload(modelpath)
    if estimator == None:
        return 2
    
    # Defining Folds specifing number of splits.
    kf=KFold(n_splits=10, shuffle=True, random_state=seed)
    
    # Actual validation (verbose and using max number of jobs)
    results = cross_val_score(estimator,X,Y,cv=kf,verbose=1,n_jobs=-1)
    
    # Returning the mean value among the validation results for each fold.
    return results.mean()


############################################
#%%
# Dict of arguments passed to XGboost constructor.
args = {'max_depth':5,
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

# Function to construct and train a BDT using the XGboost library.
    # datapath: string argument specifing path (local or URL) of training data in csv format.
    # datate: string argument specifing path (local or URL) of test data in csv format.
    # args: dict argument with list of parameters.
    # iterations: int argument specifing number of iterations performed in training.

def xgtrain(datapath,datate,args={},iterations=10):
   
    # Loading and preparing training data and test data.
    dataset = preprocessing(datapath,cor=True)
    datatest = preprocessing(datate)
    if dataset.empty or datatest.empty :
        return 1
    else:
        data = dataset.copy()
        datats = datatest.copy()
        
        # Encoding 
        #encoder = LabelEncoder()
        #encoder.fit(data['bxout'])
        #print(encoder.classes_)
        
    # Constructing from test data DMatrix object to pass to XGboost methods.
    dtest = xgb.DMatrix(datats[['bx','phi','phiB','wheel','sector','station','quality']])
    print("Dataset length: ",len(data))
    
    # Train validation splitting.
    Xtrain,Xvalid,Ytrain,Yvalid=train_test_split(data[['bx','phi','phiB','wheel','sector','station','quality']],data["bxout"],random_state=seed,test_size=0.3)
    
    # Constructing from training and validation data DMatrix objects to pass to XGboost methods.
    dtrain = xgb.DMatrix(Xtrain,label=encoder.transform(Ytrain))
    dvalid = xgb.DMatrix(Xvalid,label=encoder.transform(Yvalid))
    
    # Creating list used to tell XGboost training method to validate while training.
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    
    print(args)

       
    #scibst = XGBClassifier(max_depth=20,learning_rate=0.3,subsample=0.82,colsample_bytree=0.68,objective='multi:softmax',seed=seed,verbosity=2,n_jobs=-1)
    #scibst.fit(Xtrain,Ytrain,eval_metric="merror",verbose=True)
    evals_result={}    
    # Training method.
    bst = xgb.train(args,dtrain,iterations,evallist,early_stopping_rounds=10, evals_result=evals_result)
    
    # Saving tree snapshot.
    bst.dump_model('bstdump.raw.txt')
    
    metric = list(evals_result['eval'].keys())
    results = evals_result['eval'][metric[0]]
    print(evals_result)

    plt.plot(list(1-a for a in evals_result['train']['merror']))
    plt.plot(list(1-a for a in evals_result['eval']['merror']))
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')      
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Eval'], loc='upper left')
    plt.savefig("XGBoost_model_accuracy.png")
    plt.clf()
    
    plt.plot(list(evals_result['train']['mlogloss']))
    plt.plot(list(evals_result['eval']['mlogloss']))
    plt.title('Model Loss')
    plt.ylabel('Loss')      
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Eval'], loc='upper left')
    plt.savefig("XGboost_Model_Loss.png")
    plt.clf()
    
    out=dump(bst,"bst.joblib")
    # xgb.plot_importance(bst,importance_type='gain')
    # pl=plt.plot(range(len(results)),results)
    #plt.savefig("foo2.pdf",bbox_inches='tight')
    
    # Evaluating predictions for test data.
    pred=bst.predict(dtest)
    
    #evals=bst.evals_result()
    #print(evals)
    #plt = plot
    #res = {'label' :encoder.transform(datats["bxout"]),'pred': pred}
    #sub = pd.DataFrame(res, columns=['label','pred'])
    #sub.to_csv("errrr.csv",index=False)
    #results = scibst.evals_result()
    
    # Returning accuracy evaluated using the test data provided.
    return accuracy_score(encoder.transform(datats["bxout"]),pred)
   

#%%    
def neighbor(datapath):

    time0 = time.time()

    dataset = preprocessing(datapath)
    if dataset.empty:
        return 1
    else:
        data = dataset.values
    
    datatest = preprocessing("dataset.csv")
    if datatest.empty:
        return 1
    else:
        datat = datatest.values
    # Handling of number of entries argument (NSample).
    
    X = data[:,1:8]
    BX = data[:,0]
    test = datat[:,1:8]
    testlabel = datat[:,0]
    # Encoding BXs to have labels from 0 to 9.
    #encoder = LabelEncoder()
    #encoder.fit(BX)
    y = encoder.transform(BX)
    ytest = encoder.transform(testlabel)
    clf = neighbors.KNeighborsClassifier(n_neighbors=15,algorithm='kd_tree', weights='uniform',n_jobs=-1)
    clf.fit(X,y)
    
    #Z = clf.predict(test)
    
    # Definition of a 'distance' between prediction and true label.
    
    print(clf.score(test,ytest))
    
    #print(Z)

    print("Executed in %s s" % (time.time() - time0))

    
    
    
    
    return 0

def hyperparam_search(data,param_grid={}):
    
    dataset,encoded_labels = training_data_loader(data)
    
    estimator = KerasClassifier(build_fn=baseline_model, verbose=2, epochs=48)

    search = GridSearchCV(estimator, param_grid,n_jobs=-1)
    search.fit(dataset,encoded_labels)
    print(search.best_params_)
#%%
def run(argss):
    
    
    if argss.xgb:
        if argss.p:
            pred = prediction(argss.data,argss.modelupload)
            pred.astype(int).tofile("xgbres.csv",sep='\n',format='%1i')
                
            
        else:
            try:
                xgparams = json.load(open(pars.xgparams)) if pars.xgparams[0][0] == '/' else json.load(open(os.path.dirname(os.path.realpath(__file__))+'/'+pars.xgparams))
                xgparams['num_class'] = eval(xgparams['num_class'])
                xgparams['seed'] = eval(xgparams['seed'])
            except Exception:
                print("XGboost parameters not found! Using default values")
                xgparams = {}
            bdtres = xgtrain(argss.data,
                    "datatree.csv",
                    xgparams,
                    20)
            print("XGboost's accuracy", bdtres)    
    
    if argss.nn:
        if argss.p:
            pred = prediction(argss.data,argss.modelupload)
            pred.astype(int).tofile("kerasres.csv",sep='\n',format='%1i')
        else:
            model = training_model(argss.data,
                            NSample=0
                            )
            
            results = 1- nn_performance(model,"datatree.csv")
            print("Neural Network's accuracy: ", results)
    #print("XGboost's accuracy", bdtres)
    if argss.nn == 0 and argss.xgb == 0:
        print("Choose a model using the --xgb and/or --nn flags")
        return 1
    
    return 0


#%%    
# Function used to inspect various seeds in order to have a better accuracy.
def seed_selector():
    acc = pd.read_csv("accu4.csv",header=0)
    seeds = pd.DataFrame(columns=['seed','accuracy'])
    for x in range(5):
        seed = acc.max(axis=0)
        acc=acc.drop(index=(seed[0].astype(int)-acc.head()['seed']))
        seeds = seeds.append(seed,ignore_index=True)
    seeds.to_csv("seeds.csv",index=False,mode='a')
#%%
    
# TESTING
    
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
if __name__ == '__main__':
    time0 = time.time()
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=str,help="Url or path of dataset in csv format.")
    parser.add_argument('--xgb', action='store_true', help='If flagged activate xgboost model')
    parser.add_argument('--nn', action='store_true', help='If flagged activate keras nn model')
    parser.add_argument('--nnlayout', type=dict, help="Layout for the Keras NN")
    # parser.add_argument('--modeltraining', help="Choice of ML model between NN, xgboost BDT or KNN")
    parser.add_argument('--xgparams', help="Hyperparameters for xgboost")
    parser.add_argument('--nnparams', help="Hyperparameters for Keras NN")
    parser.add_argument('-p', action='store_true', help='If flagged set predecting mode using a previously trained model')
    parser.add_argument('--modelupload',type=str,help="Url or path of model in joblib format")
    
    #parser.set_defaults
    #print(parser.parse_args())
    pars = parser.parse_args()
    #xgparams = json.load(open(pars.xgparams)) if pars.xgparams[0][0] == '/' else json.load(open(os.path.dirname(os.path.realpath(__file__))+'/'+pars.xgparams))
    

    
    run(pars)
    
    print("Executed in %s s" % (time.time() - time0))
    
    