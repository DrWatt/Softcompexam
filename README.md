# Softcompexam

Exam code for Software and Computing for Nuclear and Subnuclear Physics course. 

## Description

The project consists on a small application based on two (maybe three) different approaches to simulate the bunch-crossing assignement done by the TwinMux breadboard in the muon tracking system of CMS:

- A Deep Neural Network multi-classificator developed using the Keras library: 8 input nodes -> 4 hidden layers of densely connected nodes (10 -> 12 -> 10 -> 8) -> 9 output nodes corresponding to the 9 possible BX identifications.

- A boosted decision tree provided by the Xgboost library: 1 tree with a maximum depth of 10 layers and learning rate of 0.25.

In both cases the layouts and hyper-parameters were chosen after trying various configurations in order to get the highest test accuracy possible on my machine.  

The data used for this analysis was collected by the CMS detector during the proton-proton running at s^2 = 13 TeV^2 of the Large Hadron Collider at the beginning of September 2018, and it corresponds to around 400 pb^(-1) of integrated luminosity. After being stored, data was further processed with the CMS offline software to perform reconstruction of physics objects. In particular track segments were built within single DT chambers. A further event selection, requiring pairs of oppositely charged muons with an invariant mass above 60 GeV, was performed. The resulting dataset has then been saved in a flat ROOT tree format for easier analysis.



The ROOT tree used to popolate the test database can be found here: https://www.dropbox.com/s/5cywqqkcv04649l/DTTree_zMuSkim_70k.root?dl=0

(WORK IN PROGRESS)

[comment]: # (This is done by supplying 15000  tracks to the NN for training. I have trained the NN on the Colaboratory platform developedby Google, due to the long time needed in order to complete the process.)