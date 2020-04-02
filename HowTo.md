# How to guide

## Retrieving the data from a ROOT Tree (Optional)

This project is aimed to deal with data coming from the muon trigger chain from the CMS Collaboration at CERN. That means the way the data is stored inside a Tree can be considered widely standardized. This gives us the chance to use a single ROOT macro to retrieve this data and put it in a more easy to handle format, i.e. csv format.

An example of this can be seen inside the Root_macros folder in the Repository.
Having already installed ROOT on your PC (if not installed refer to [this link](https://root.cern.ch/downloading-root) to download and this [guide](https://root.cern.ch/building-root) to build ROOT in your system), open a terminal and type
```bash
$ root
```
To start a new ROOT session. Then load the .root file of tree:
```bash
root[0] TFile* f = new TFile("path")
root[1] f->ls()
```
After the second command, you will see something like:
```
TFile**		DTTree_ZMuSkim2017F_94X_1kev.root	
TFile*		DTTree_ZMuSkim2017F_94X_1kev.root	
KEY: TTree	DTTree;1	CMSSW DT tree
```
In the last line you can see the name of the TTree inside the file (DTTree).
Then we have to call the method MakeClass(), which will create an header and an implementation in c++ that make the access of the data inside the Tree easy, by making each leaf available as a variable or pointer to variables.
```bash
root[2] DTTree->MakeClass()
```
Once you can see the two file now generated, open the .C one and paste this inside the Loop() function which is already there (you can see the macro I have used inside the Root_macros directory for comparison):

```c++
	ofstream o;
	o.open("datatree2.csv");
	o << "bxout,bx,phi,phiB,wheel,sector,station,quality,is2nd\n";

   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      int ov = 0;
      float t = jentry/1000.;
      if (t == (int)t) clog <<'\r' << "Entry " << jentry;
      for (unsigned int i = 0; i < ltTwinMuxIn_phi->size(); ++i)
      {
     		for (int j = 0; j < ltTwinMuxOut_bx->size(); ++j)
     		{
     			if(ov==0&&ltTwinMuxIn_wheel->at(i) == ltTwinMuxOut_wheel->at(j) && ltTwinMuxIn_sector->at(i) == ltTwinMuxOut_sector->at(j)&&ltTwinMuxIn_station->at(i)==ltTwinMuxOut_station->at(j))
     			{

     				o << ltTwinMuxOut_bx->at(j) << ',';
     				o << ltTwinMuxIn_bx->at(i) << ',';
            		o << ltTwinMuxIn_phi->at(i) << ',';
            		o << ltTwinMuxIn_phiB->at(i)<< ',';
            		o << ltTwinMuxIn_wheel->at(i)<< ',';
            		o << ltTwinMuxIn_sector->at(i)<< ',';
            		o << ltTwinMuxIn_station->at(i)<< ',';
					o << ltTwinMuxIn_quality->at(i)<< ',';
					o << ltTwinMuxIn_is2nd->at(i);
					o << '\n';
					++ov;

     			}
     		}
        ov=0;

      }



   }
   o.close();
```
Then you can start from a new session of ROOT (you can close the one used before by typing `.q`) and load the macro and create an object of the class DTTree (using the name seen before, however the name of the class depends on the name of the TTree inside the Root file):
```bash
root[0] .L DTTree.C
root[1] DTTree m
```
Finally let's call the method Loop() which will create the .csv file with our data inside:
```bash
root[2] m.Loop()
```

## Using the project.py script

If an error occurs due to missing permission of execution, you can correct that with this command on linux:
```bash
$ chmod -v u+x project.py
```

Now the file you are interested in is ready on your disk if you have followed the steps above. However the script allows you to use directly a URL of a data in csv format with the same features you find in the default one (also downloaded automatically if no data is specified)

There are two Machine Learning models available: a XGBoost Boosted Decision Tree and a Neural Network built with the Keras library.

### Boosted Decision Tree (BDT)

To run a training of a BDT you have to type on the Command Line:
```bash
./project.py --xgb
```
This will trigger a training of the default dataset (that you can find [here](https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv) however the script will automatically download it) with default parameters:
```json
{
    "max_depth":5,
    "eta":0.3,
    "subsample":0.82,
    "colsample_bytree": 0.68,
    "eval_metric": ["merror","mlogloss"],
    "silent":0,
    "objective":"multi:softmax",
    "num_class": "len(encoder.classes_)",
    "seed" : "seed",
    "num_parallel_tree" : 5
}
```
If you want to train the BDT on different data, you can do it by passing the flag ` --data ` followed by the path on your disk or an URL to the dataset in csv format:
```bash
$ ./project.py --xgb --data https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv
```
To specify different parameters than the default ones, you have to create a JSON file with inside a dictionary like the one seen above (you can find an example [here](params.json)) and specify it with the `--xgparams` flag:
```bash
$ ./project.py --xgb --xgparams params.json
```
You can also perform only inference with a pretrained model specifing the path or URL of an already trained BDT in .joblib format:
```bash
$ ./project.py --xgb --modelupload "XGBoost_Model.joblib"
```
After the training you will find your model saved in joblib format and plots for accuracy and loss function. While, after the predictions, you will find a csv file with all the labels for each entry of the dataset in input.

### Neural Network from Keras (NN)

To run a training of a BDT you have to type on the Command Line:
```bash
./project.py --nn
```
his will trigger a training of the default dataset (that you can find [here](https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv) however the script will automatically download it) with default parameters:
- Number of epochs : 48
- Batch size: 30
- Validation split: 0.3

To specify different values for these parameters you can add them after the `--nnparams` flag, writing 0 for the ones you still want the default value:
```bash
./project.py --nn --nnparams 50 2 0.1
```
If you want to train the NN on different data, you can do it by passing the flag ` --data ` followed by the path on your disk or an URL to the dataset in csv format:
```bash
$ ./project.py --nn --data https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv
```
You can also perform only inference with a pretrained model specifing the path or URL of an already trained BDT in .joblib format:
```bash
$ ./project.py --nn --modelupload "KerasNN_Model.joblib"
```
After the training you will find your model saved in joblib format and plots for accuracy and loss function. While, after the predictions, you will find a csv file with all the labels for each entry of the dataset in input.
