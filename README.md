# Softcompexam  [![Build Status](https://travis-ci.com/DrWatt/Softcompexam.svg?branch=master)](https://travis-ci.com/DrWatt/Softcompexam)

Exam code for Software and Computing for Nuclear and Subnuclear Physics course. 

## Description

The project consists on a small application based on two different approaches to simulate the bunch-crossing assignment done by the TwinMux breadboard in the muon tracking system of CMS:

- A Deep Neural Network multi-classificator developed using the Keras library: 7 input nodes -> 2 hidden layers of densely connected nodes (8 -> 8) -> 9 output nodes corresponding to the 9 possible BX identifications.

- A boosted decision tree provided by the Xgboost library.

In both cases the layouts and hyper-parameters were chosen after trying various configurations in order to get the highest test accuracy possible on my machine.  

The data used for this analysis was collected by the CMS detector during the proton-proton running at s^2 = 13 TeV^2 of the Large Hadron Collider at the beginning of September 2018, and it corresponds to around 400 pb^(-1) of integrated luminosity. After being stored, data was further processed with the CMS offline software to perform reconstruction of physics objects. In particular track segments were built within single DT chambers. A further event selection, requiring pairs of oppositely charged muons with an invariant mass above 60 GeV, was performed. The resulting dataset has then been saved in a flat ROOT tree format for easier analysis. This ROOT Tree has then been parsed into a csv file, keeping the relevant variables, in order to use the DataFrame functionalities from within the Pandas module.

The script, written in python3, can be executed from the command line, typing `./project.py` with the following flags and options:

- `--data` to specify the Url or path of the dataset in csv format (if unspecified a default dataset will be used); 
- `--xgb` to activate the XGBoost model;
- `--nn` to activate the NN built with Keras
- `--xgparams` to specify the path of a json file specifying the following hyper parameters:
  - `max_depth` of the tree;
  - `eta` i.e. learning rate;
  - `subsample` of the total dataset to use for each iteration;
  - `eval_metric` the metrics used to quantify the validity of the model;
  - `silent` verbosity setting;
  - `objective`
  - `num_class` number of classes the algorithm has to deal with and predict;
  - `seed` seed of the random generator;
  - `num_parallel_tree` number of parallel trees to activate the Random Forest techinque.
- `--nnparams` number of epochs, batch size and validation split performed by the NN (input values simply separated by space and set equal to 0 a parameter to use the relative default value);
- `--modelupload` to specify the Url or path of a pretrained model in joblib format.


## Examples

For a thorough explanation on how to use this application consult the [How to guide](HowTo.md).

Here there are some examples of using the script from command line:
```bash
$ ./project.py --xgb --data "https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv"
```
Starting training using XGBoost with data donwloaded from the URL provided, using default parameters.
```bash
$ ./project.py --nn --data "https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv"
```
Starting training using Keras Neural Network with data donwloaded from the URL provided, using default parameters.
```bash
$ ./project.py --xgb --data "https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv" --xgparams "params.json"
```
Starting training using XGBoost with data donwloaded from the URL provided and parameters taken from JSON configuration file.
```bash
$ ./project.py --nn --data "https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv" --nnparams 5 5 0.5
```
Starting training using Keras Neural Network with data donwloaded from the URL provided, with 5 Epochs, batch size of 5 input and a validation split of the data of 0.5. 


The ROOT tree used to populate the test database can be found [here](https://www.dropbox.com/s/5cywqqkcv04649l/DTTree_zMuSkim_70k.root?dl=0).
For more information about XGBoost and Keras, you can find their documentation, respectively, [here](https://xgboost.readthedocs.io/en/latest/index.html) and [here](https://keras.io/).
If you want to know more about the Muon Trigger system at CMS, you can have a look at my [bachelor's thesis](http://amslaurea.unibo.it/16943/1/Tesi.pdf).


[comment]: # (This is done by supplying 15000  tracks to the NN for training. I have trained the NN on the Colaboratory platform developed by Google, due to the long time needed in order to complete the process.)
