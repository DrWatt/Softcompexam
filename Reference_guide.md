# Reference Guide

## baseline_model(indim=7,hidden_nodes=[8,8],outdim=9)

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

## model_upload(modpath)
    
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

## data_upload(datapath):
    
Function to load data from disk or using an URL.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format.

    Returns
    -------
    pandas.dataframe
        Dataframe containing data used for training and/or inference.

    

## preprocessing(datapath,cor=False):
    
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

    

## prediction(datapath,modelpath,performance=False,NSamples=0):
    
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

    

## nn_performance(modelpath, datatest):
    
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

    

## training_data_loader(datapath,NSample=None):
    
Function performing one-hot encoding.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format..
    NSample : int, optional
        number of entries used of the dataset. If NSamples == None or NSamples > data size the all dataset will be used.. The default is None.

    Returns
    -------
    List of pandas.dataframe
        List made up of two Dataframes: the first contains the preprocessed data and the second one contains the one hot encoded labels.

    

## training_model(datapath,NSample=0, par = [48,30,0.3],plotting=False):
    
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
    String
        Namefile of the model saved to disk.

    

## plotting_NN(estimator,history):
    
Plotting function that saves three different .png images: 
- Representation of the neural network;
- Plot of the model accuracy thorugh epochs for training and validation sets;
- Plot of the model loss function thorugh epochs for training and validation sets.
```
    Parameters
    ----------
    estimator : keras.wrappers.scikit_learn.KerasClassifier
        Object containing NN model.
    history : keras.callbacks.History
        Return of fit function of the NN model.

    Returns
    -------
    None.
```
    

## cross_validation(modelpath,datapath):
    
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

    

## xgtrain(datapath,args={'eval_metric': ['merror','mlogloss']},iterations=10):
    
    
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
    int
        No Error code return.

       
## plotting_xgb(evals_result):
    
    Plotting function for the trained XGBoost model.

    Parameters
    ----------
    evals_result : dictionary
        Dictionary with the values of the error metrics in each iteration, divided in train and validation. For example: {'train':[{'merror':##,'mlogloss':##}],'eval':[{'merror':##,'mlogloss':##}]}.

    Returns
    -------
    None.

    
    

## neighbor(datapath):
    
K-Nearest neighbor implementation function. Work in progress and unused for now.

    Parameters
    ----------
    datapath : String
        path (local or URL) of training data in csv format.

    Returns
    -------
    int
        No Error code return.

    

## hyperparam_search(data,param_grid={}):
    
Function used to asses the optimal parameters for the Keras NN using a brute force approach.

    Parameters
    ----------
    data : String
        path (local or URL) of training data in csv format.
    param_grid : dictionary, optional
        Dict with paramaters we want to search for as KEYS and list of values for each parameter as VALUE. The default is {}.

    Returns
    -------
    int
        No Error code return.

    

## run(argss):
    
Main function invoked by execution in shell.

    Parameters
    ----------
    argss : argparse.ArgumentParser
        Arguments parsed to the invoked function. Contains flags which control the execution of the script: e.g. the model of choice and if you want to train or infer.

    Returns
    -------
    int
        Error code return.

    

## seed_selector():
    
Function used to inspect various seeds in order to have a better accuracy.

    Returns
    -------
    None.

    
