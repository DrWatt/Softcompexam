#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#xgboost
import xgboost as xgb

#%%
seed = 1563
np.random.seed(seed)
encoder = LabelEncoder()
encoder.fit([-4,-3,-2,-1,0,1,2,3,4])

def baseline_model(indim=7,hidden_nodes=[8,8],outdim=9):
    '''
    Model constructor definition, as needed to use scikit-learn wrapper with keras.    
    
    Parameters
    ----------
    indim : int, optional
        Number of features of dataset and dimension of input layer. The default is 7.
    hidden_nodes : list, optional
        List of number of nodes per layer. The default is [8,8].
    outdim : int, optional
        Number of classes and dimension of output layer. The default is 9.

    Returns
    -------
    model : keras.engine.sequential.Sequential
        Sequntial NN object to be used inside KerasClassifer method.

    '''
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

def model_upload(modpath):
    '''
    Function to load pretrained NN.

    Parameters
    ----------
    modpath : String
        path (local or URL) of model in joblib format..

    Returns
    -------
    keras.wrappers.scikit_learn.KerasClassifier 
        Wrapper from the Scikit.learn library of a the Keras Classifier.
    or
    xgboost.core.Booster
        Booster is the model of xgboost, that contains low level routines for training, prediction and evaluation.

    '''
    if("http" in modpath):
        print("Downloading Model")
        try:    
            # Download
            mod = requests.get(modpath)
            mod.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            raise 
        # Writing model on disk.
        with open("model.joblib","wb") as o:
            o.write(mod.content)
        modpath = "model.joblib"
    print("Loading Model from Disk")
    try:
        # Uploading model from disk. 
        estimator = load(modpath)
    except Exception:    
        raise 
    return estimator

def data_upload(datapath):
    '''
    Function to load data from disk or using an URL.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format.

    Returns
    -------
    pandas.dataframe
        Dataframe containing data used for training and/or inference.

    '''
    if("http" in datapath):
        print("Downloading Dataset")
        try:
            # Download
            dataset = requests.get(datapath)
            dataset.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            raise        
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
        raise
    return dataset


def preprocessing(datapath,cor=False):
    '''
    Feature scaling: rescaling via min-max normalization.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format..
    cor : Boolean, optional
        Set True if correlation matrix plot is needed. The default is False.

    Returns
    -------
    pandas.dataframe
        Dataframe rescaled using min-max normalization.

    '''
    

    data = data_upload(datapath)

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
    
    # Max Min Normalization: for each column x_normalized = (x - x_min)/(x_max-x_min)
    cs = MinMaxScaler()
    
    data.iloc[:,2:] = cs.fit_transform(data.iloc[:,2:])
    
    # # # Old normalization "by hand" 
    # # Normalization factor (max - min for every feature).
    # norm = data.max() - data.min()
    # # Subtracting for every feature its minimum.
    # data2 = data - [0,0,data.min()[2],data.min()[3],data.min()[4],data.min()[5],data.min()[6],data.min()[7],0]
    # # Normalization.
    # data = data2.div([1,1,norm[2],norm[3],norm[4],norm[5],norm[6],norm[7],1])
    # # Casting to make bx labels integers again.
    # data['bxout'] = data['bxout'].astype(int)
    # data['bx'] = data['bx'].astype(int)

    return data
#%%

def prediction(datapath,modelpath,performance=False,NSamples=0):
    '''
    Function to compute classification of a dataset using a pretrained NN.
    

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format..
    modelpath : String
        path (local or URL) of model in joblib format..
    performance : Boolean, optional
        Set between two return mode: False -> return only predictions; True -> return predictions and true labels if provided (for evaluating performance). The default is False.
    NSamples : int, optional
        number of entries used of the dataset. If NSamples == 0 or NSamples > data size the all dataset will be used. The default is 0.

    Returns
    -------
    pandas.dataframe
        Dataframe containing inferences made by the model for each entry of the data in input.
        
    list of pandas.dataframe
        List made up of two dataframes, the first contains the inferences, the second contains the true labels for validation.

    '''
    # Loading dataset and preprocessing it.

    dataset = preprocessing(datapath)

    # Loading NN.
    estimator = model_upload(modelpath)

    # Failed loading handling.
    # if estimator == 404:
    #     return 404
    if type(estimator) != KerasClassifier and type(estimator) != xgb.core.Booster:
        print("Check loaded model compatibility.")
        raise TypeError(estimator)
    
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
        pred=encoder.inverse_transform(estimator.predict(X))
    except Exception: 
        dtest = xgb.DMatrix(X[['bx','phi','phiB','wheel','sector','station','quality']])
        pred=encoder.inverse_transform(estimator.predict(dtest).astype(int))   

    # condition to return also labels.
    if performance:
        labels = dataset.head(len(X.index)).get('bxout')
        return [pred,labels]

    return pred


def nn_performance(modelpath, datatest):
    '''
    Function to perform a simple comparison between prediction and known labels of a test sample.

    Parameters
    ----------
    modelpath : String
        path (local or URL) of model in joblib format..
    datatest : String
        path (local or URL) of data in csv format..

    Returns
    -------
    Float
        Fraction of good inferences made by a model.

    '''
    # Performing prediction
    test = prediction(datatest,modelpath,performance=True)
    # Catching error in prediction function
    # if type(test) != list or type(test[0]) != np.ndarray or type(test[1]) != pd.Series:
    #     print("Prediction failed")
    #     return 3
    
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

def training_data_loader(datapath,NSample=None):
    '''
    Function performing one-hot encoding.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format.
    NSample : int, optional
        number of entries used of the dataset. If NSamples == None or NSamples > data size the all dataset will be used.. The default is None.

    Returns
    -------
    List of pandas.dataframe
        List made up of two Dataframes: the first contains the preprocessed data and the second one contains the one hot encoded labels.

    '''
    
    # Uploading preprocessed dataset.
    dataset = preprocessing(datapath,cor=True)
    data = dataset.values
        
    # Handling of number of entries argument (NSample).
    if NSample == None or NSample == 0:
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


    #small data: batch = 8
    #medium data: batch = 30
def training_model(datapath,NSample=0, par = [48,30,0.3],plotting=False):
    '''
    NN training function.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format..
    NSample : int, optional
        number of entries used of the dataset. If NSamples == 0 or NSamples > data size the all dataset will be used.. The default is 0.
    par : List of int,int,float, optional
        list of paramaters passed to the NN costructor [number of epochs the NN will be trained for, size of the batches used to update the weights, fraction of the input dataset used for validation]. The default is [48,30,0.3].

    Returns
    -------
    pandas.DataFrame
        Values assumed by evaluation metrics through the epochs.

    '''
    
    # Loading and preparing data for training.

    dataset,encoded_labels = training_data_loader(datapath,NSample)

    # Setting default values in case of some missing parameter.
    if par[0] == 0 : par[0] = 48
    if par[1] == 0 : par[1] = 30
    if par[2] == 0 : par[2] = 0.3
    print(par)
    # Keras constructor with scikit-learn API.
    estimator = KerasClassifier(build_fn=baseline_model, epochs=par[0], batch_size=par[1], verbose=2)
    
    # Training method for our model. 
    history = estimator.fit(dataset, encoded_labels, epochs=par[0], batch_size=par[1],verbose=2,validation_split=par[2])

    # Saving trained model on disk. (Only default namefile ATM)
    out=dump(estimator,"KerasNN_Model.joblib")
    if plotting:
        plotting_NN(estimator, history)
    # Returning namefile of model in order to use the trained model in other functions e.g. only for predictions.
    return pd.DataFrame.from_dict(history.history)
    
def plotting_NN(estimator,history):
    '''
    Plotting function that saves three different .png images: 
    1) Representation of the neural network;
    2) Plot of the model accuracy thorugh epochs for training and validation sets;
    3) Plot of the model loss function thorugh epochs for training and validation sets.

    Parameters
    ----------
    estimator : keras.wrappers.scikit_learn.KerasClassifier
        Object containing NN model.
    history : keras.callbacks.History
        Return of fit function of the NN model.

    Returns
    -------
    None.

    '''
    #plot_model(estimator.model, to_file='model.png',show_shapes=True)
    
    # Accuracy and Loss function plots saved in png format.
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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
    


def cross_validation(modelpath,datapath):
    '''
    KFold cross validation function using scikit-learn API.

    Parameters
    ----------
    modelpath : String
        path (local or URL) of model in joblib format.
    datapath : String
        path (local or URL) of data in csv format.

    Returns
    -------
    Float
        Mean between the inference accuracy of each class.

    '''
    
    # Loading model from disk.
    estimator = model_upload(modelpath)

    # Loading and preparing data for validation.
    X,Y=training_data_loader(datapath)
    
    

    if type(estimator) != KerasClassifier and type(estimator) != xgb.core.Booster:
        print("Check loaded model compatibility.")
        raise TypeError(estimator)
    
    # Defining Folds specifing number of splits.
    kf=KFold(n_splits=10, shuffle=True, random_state=seed)
    
    # Actual validation (verbose and using max number of jobs)
    results = cross_val_score(estimator,X,Y,cv=kf,verbose=1,n_jobs=-1)
    
    # Returning the mean value among the validation results for each fold.
    return results.mean()


############################################
#%%


def xgtrain(datapath,args={'eval_metric': ['merror','mlogloss']},iterations=10):
    '''
    
    Function to construct and train a BDT using the XGboost library.
    
    Parameters
    ----------
    datapath : String
        path (local or URL) of training data in csv format.
    args : dictionary, optional
        list of parameters. The default is {'eval_metric': ['merror','mlogloss']}.
    iterations : int, optional
        number of iterations performed in training. The default is 10.

    Returns
    -------
    pandas.DataFrame
        Values assumed by evaluation metrics through the epochs.

    '''   
    # Loading and preparing training data.
    dataset = preprocessing(datapath,cor=True)
    data = dataset.copy()
        
    print("Dataset length: ",len(data))
    
    # Train validation splitting.
    Xtrain,Xvalid,Ytrain,Yvalid=train_test_split(data[['bx','phi','phiB','wheel','sector','station','quality']],data["bxout"],random_state=seed,test_size=0.3)
    
    # Constructing from training and validation data DMatrix objects to pass to XGboost methods.
    dtrain = xgb.DMatrix(Xtrain,label=encoder.transform(Ytrain))
    dvalid = xgb.DMatrix(Xvalid,label=encoder.transform(Yvalid))
    
    # Creating list used to tell XGboost training method to validate while training.
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    
    print(args)

    evals_result={}    
    # Training method.
    bst = xgb.train(args,dtrain,iterations,evallist,early_stopping_rounds=10, evals_result=evals_result)
    
    # Saving tree snapshot.
    bst.dump_model('bstdump.raw.json',dump_format='json')
    
    # Converting evals_result dict in a tidier dataframe
    errmetric = ['train '+ i for i in evals_result['train'].keys()]
    valmetric = ['eval '+ i for i in evals_result['eval'].keys()]
    metrics = errmetric+valmetric
    res = pd.DataFrame.from_dict({k:evals_result[k.split()[0]][k.split()[1]] for k in metrics})
    
    # Objective and evaluation functions plots.
    
    plotting_xgb(evals_result)


    # Saving XGBoost model in joblib format.
    out = dump(bst,"XGBoost_Model.joblib")
   
    
    
    
    if not ('merror' in evals_result['train'] and 'mlogloss' in evals_result['train']):
        print("\n\n\nUSING EVALUATION METRICS NOT SUITED FOR MULTICLASSIFICATION. USE AT YOUR RISK\n\n\n")
    # Returning evaluation metrics values through the epochs.
    return res


def plotting_xgb(evals_result):
    '''
    Plotting function for the trained XGBoost model.

    Parameters
    ----------
    evals_result : dictionary
        Dictionary with the values of the error metrics in each iteration, divided in train and validation. For example: {'train':[{'merror':##,'mlogloss':##}],'eval':[{'merror':##,'mlogloss':##}]}.

    Returns
    -------
    None.

    '''
    
    for met in evals_result['train']:
        if met == 'merror':
            plt.plot(list(1-a for a in evals_result['train']['merror']))
            plt.plot(list(1-a for a in evals_result['eval']['merror']))
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')      
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Eval'], loc='upper left')
            plt.savefig("XGBoost_model_accuracy.png")
            plt.clf()
            continue
            
        plt.plot(list(a for a in evals_result['train'][met]))
        plt.plot(list(a for a in evals_result['eval'][met]))
        t = "Model_" + met
        plt.title(t)
        plt.ylabel(met)      
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Eval'], loc='upper left')
        plt.savefig("XGBoost_" + t +".png")
        plt.clf()
    
    
    

#%%
    
def run(argss):
    '''
    Main function invoked by execution in shell.

    Parameters
    ----------
    argss : argparse.ArgumentParser
        Arguments parsed to the invoked function. Contains flags which control the execution of the script: e.g. the model of choice and if you want to train or infer.

    Returns
    -------
    Dictionary
        Values assumed by evaluation metrics through the epochs for both models.

    '''
    resul = {'XGBoost':0,'KerasNN':0}
    if argss.testnn:
         b = prediction("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv", "Pretrained_models/XGBoost_Model.joblib")
         c = prediction("https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv", "Pretrained_models/XGBoost_Model.joblib")
         assert np.equal(b,c).all()
         a = baseline_model()
         b = model_upload("https://www.dropbox.com/s/gr1apt6na9szclg/KerasNN_Model.joblib?dl=1").model
         os.remove("model.joblib")
         assert a.count_params() == b.count_params()
         return resul
        
    if argss.nn == 0 and argss.xgb == 0:
        raise Exception("Choose a model using the --xgb and/or --nn flags")
        #print("Choose a model using the --xgb and/or --nn flags")
    
    if argss.data==None: argss.data = "https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv"
    # Routine followed when --xgb is True
    if argss.xgb:
        # Selection between prediction, using a pretrained model, and training a new one.
        if argss.modelupload:
            pred = prediction(argss.data,argss.modelupload)
            pred.astype(int).tofile("xgbres.csv",sep='\n',format='%1i')
            print("Predictions saved in .csv format")                
            
        else:
            # Reading parameters in json format if found. If no parameters are specified, default values will be used.
            try:
                xgparams = json.load(open(pars.xgparams)) if pars.xgparams[0][0] == '/' else json.load(open(os.path.dirname(os.path.realpath(__file__))+'/'+pars.xgparams))
                xgparams['num_class'] = eval(xgparams['num_class'])
                xgparams['seed'] = eval(xgparams['seed'])
            except Exception:
                print("XGboost parameters not found! Using default values")
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
            # Construction and training of XGBoost BDT.
            bdtres = xgtrain(argss.data,
                    xgparams,
                    20)  
            print("Plots of evaluation metrics vs epochs saved. \nModel in .joblib format saved for prediction and testing")
            resul['XGBoost'] = bdtres
    # Routine followed when --nn is True
    if argss.nn:
        
        # Selection between prediction, using a pretrained model, and training a new one.
        if argss.modelupload:
            pred = prediction(argss.data,argss.modelupload)
            pred.astype(int).tofile("kerasres.csv",sep='\n',format='%1i')
            print("Predictions saved in .csv format")
        else:
            # try:
            #     nnparams = json.load(open(pars.nnparams)) if pars.nnparams[0][0] == '/' else json.load(open(os.path.dirname(os.path.realpath(__file__))+'/'+pars.nnparams))
            # except Exception:
            #     print("NN parameters not found! Using default values")
            #     nnparams = [48,30,0.3]
            
            # Reading list of parameters. If no parameters are specified, default values will be used.
            if argss.nnparams==None:argss.nnparams = [0,0,0]
            print (argss.nnparams)
            pr = [int(argss.nnparams[0]),int(argss.nnparams[1]),float(argss.nnparams[2])]
            print(pr)
            
            # Construction and training of Keras NN.
            model = training_model(argss.data,
                            par=pr,
                            plotting=True)
            
            # results = 1- nn_performance(model,"datatree.csv")
            # print("Neural Network's accuracy: ", results)
            print("Plots of evaluation metrics vs epochs saved. \nModel in .joblib format saved for prediction and testing")
            resul['KerasNN'] = model
    #print("XGboost's accuracy", bdtres)
            

    
    return resul

    
#%%
if __name__ == '__main__':
    time0 = time.time()
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=str,help="Url or path of dataset in csv format.")
    parser.add_argument('--xgb', action='store_true', help='If flagged activate xgboost model')
    parser.add_argument('--nn', action='store_true', help='If flagged activate keras nn model')
    #parser.add_argument('--nnlayout', type=dict, help="Layout for the Keras NN")
    # parser.add_argument('--modeltraining', help="Choice of ML model between NN, xgboost BDT or KNN")
    parser.add_argument('--xgparams', help="Hyperparameters for xgboost in .json format")
    parser.add_argument('--nnparams',nargs='+', help="Hyperparameters for Keras NN")
    #parser.add_argument('-p', action='store_true', help='If flagged set predecting mode using a previously trained model')
    parser.add_argument('--modelupload',type=str,help="Url or path of model in joblib format")
    parser.add_argument('--testnn',action='store_true')
    
    #parser.set_defaults
    #print(parser.parse_args())
    pars = parser.parse_args()
    #xgparams = json.load(open(pars.xgparams)) if pars.xgparams[0][0] == '/' else json.load(open(os.path.dirname(os.path.realpath(__file__))+'/'+pars.xgparams))

    
    run(pars)
    
    print("Executed in %s s" % (time.time() - time0))
    

