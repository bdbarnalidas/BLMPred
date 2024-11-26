# BLMPred

There are 8 folders present in this project's Github repo. There are separate READMEs present inside each folder for the ease of user's understanding.

A brief summary of the contents is as follows:
1) Benchmarking_Datasets -> The datasets used to compare performances of various methods are present.
2) BLMPred_Datasets -> The datasets used for model training and testing are present.
3) BLMPred_models -> Contain trained BLMPred models.
4) Conda_environments -> Contain the environments needed to execute the provided scripts.
5) Cross-validation -> Scripts for cross-validating various machine learning models are present.
6) Embeddings -> Contain scripts for generating ProtTrans embeddings and processing them.
7) Model_testing -> Scripts for testing BLMPred models on independent datasets and user-provided datasets are present.
8) Model_training -> Contain scripts for training BLMPred models.

# Demo of the entire workflow

## BLMPred
BLMPred is a peptide-based linear B-cell epitope predictor. It is a Support vector machine model trained on the ProtTrans protein Language Embeddings (pLMs). BLMPred is a binary classifier which can predict whether an input peptide sequence is a potential B-cell epitope or not.

## Set up
The project is implemented with python (3.9.0). 
We provided two Conda environments needed for executing the scripts. The environment yml files are present in /Conda_environments.

- rapids.yml
- blmpred.yml

Please use the following commands to download the Conda environments from their respective .yml files:
- conda env create --name rapids -f rapids.yml
- conda env create --name blmpred -f blmpred.yml

You can use the following command to check out the list of packages installed in the Conda environments:

- conda list

### Requirements
- python=3.9
- transformers
- torch
- h5py

If the above environment export from the yml files fails, then please
execute the following steps:

- conda create -n blmpred
- conda install python=3.9
- conda install conda-forge::transformers
- conda install pytorch::pytorch
- conda install anaconda::h5py  

To install Rapids-AI, please follow the steps mentioned in https://rapids.ai

## Embeddings
BLMPred was trained based on the pre-trained protein language models (PLMs) embeddings [ProtTrans](https://github.com/agemagician/ProtTrans) of the input dataset.

For the demo illustration, we suggest to use two of the input dataset files used for benchmarking purposes. Here, it is /Benchmarking_Datasets/Benchmark_20_neg.csv and
/Benchmarking_Datasets/Benchmark_20_pos.csv.

### Convert csv files to fasta format
Firstly, we convert the csv files to fasta formatted files since
the pre-trained embeddings are generated from the input files
in fasta format.

- cd Embeddings/
- python 1_csv_to_fasta.py

### Generate ProtTrans embeddings in a binary file (.h5 file)
We generate the ProtTrans embeddings for the input file.

- python 2_create_embeddings.py

*** Please note that this step for generating embeddings is time-
consuming if executed on a CPU. Proper hardware support like GPU
is needed for faster execution.

### Merge embeddings for positive and negative samples
In the previous step, we generate embeddings separately for the
positive (experimentally verified linear B-cell epitopes) and negative (experimentally verified non-B-cell epitopes) samples.
Here, we merge both the embeddings into a single vector. 
Merge epitope embeddings and non-epitope embeddings into a single numpy array.

- python 3_merge_embeddings.py

### Processing embeddings vector
Converts .h5 file to .npy file (binary data to numpy array required for machine learning).

- python 4_process_embeddings.py

## Training models
We train models on the input dataset embeddings. Due to space limitation, we could not upload 
the embeddings for our dataset to Github. 

For executing the model training scripts, please download the following:
    a) blmpred5to60.npy.gz -> Pre-generated ProtTrans embeddings for BLMPred_5to60 dataset from the link 
    https://drive.google.com/file/d/18QQJyNyxisQWPItd559iTRo9q-PexJn0/view?usp=share_link
    b) blmpred8to25.npy.gz -> Pre-generated ProtTrans embeddings for BLMPred_8to25 dataset from the link
    https://drive.google.com/file/d/1cGBG1ikeStQiz_79NysnCpxg7D7fgTR9/view?usp=share_link

After downloading these embeddings, please unzip them by using the following commands:
    - gunzip blmpred5to60.npy.gz
    - gunzip blmpred8to25.npy.gz

** Please note that GPU along with recent CUDA version and NVIDIA driver pairs are needed to execute the model training scripts.
Please use 'rapidsAI' for parallelizing SVM training. 

To train BLMPred_5to60 model
- python blmpred_5to60.py

To train BLMPred_8to25 model
- python blmpred_8to25.py

## Testing models
Testing trained models on independent blind datasets.

For executing the model testing scripts, please download the following:
    a) blmpred5to60.npy.gz -> Pre-generated ProtTrans embeddings for BLMPred_5to60 dataset from the link 
    https://drive.google.com/file/d/18QQJyNyxisQWPItd559iTRo9q-PexJn0/view?usp=share_link
    b) blmpred8to25.npy.gz -> Pre-generated ProtTrans embeddings for BLMPred_8to25 dataset from the link
    https://drive.google.com/file/d/1cGBG1ikeStQiz_79NysnCpxg7D7fgTR9/view?usp=share_link
    c) blmpred_5to60.sav.gz -> Trained BLMPred_5to60 model from the link
    https://drive.google.com/file/d/14_rVBkX3GBjEm31NZVrhkocw1PBnN8s8/view?usp=sharing
    d) blmpred_8to25.sav.gz -> Trained BLMPred_8to25 model from the link
    https://drive.google.com/file/d/1dMES36vgPx4IGVbubdTIc2dxMD0R7NoE/view?usp=sharing

After downloading the models and the embeddings, please unzip them by using the following commands:
    a) gunzip blmpred5to60.npy.gz
    b) gunzip blmpred8to25.npy.gz
    c) gunzip blmpred_5to60.sav.gz
    d) gunzip blmpred_8to25.sav.gzs

** Please note that GPU along with recent CUDA version and NVIDIA driver pairs are needed to execute the model testing scripts.
Please use RAPIDS package for executing the scripts. 

To test BLMPred_5to60 model n the independent BLMPred_5to60_test dataset. This script will generate the performance metrics and the confusion matrix as output.
- python blmpred_5to60.py 

To test BLMPred_8to25 model n the independent BLMPred_8to25_test dataset. This script will generate the performance metrics and the confusion matrix as output.
- python blmpred_8to25.py 

*********************** Testing model on user-provided datasets ****************************************

If the users want to make predictions on their own dataset, then please follow the steps below:
1) Place your data in a .csv file following the format of [peptide length, peptide sequence].
2) Go to 'Embeddings' folder and execute 1_csv_to_fasta.py, 2_create_embeddings.py, and 4_process_embeddings.py.
3) Execute make_predictions.py for generating predictions on your dataset which will be stored in a file 'predictions.csv' [peptide id, Epitope/Non-epitope].
    3.1) 1st user-input -> Filename storing embeddings on line 5.
    3.2) 2nd user-input -> Load either BLMPred_5to60 or BLMPred_8to25 model on line 8.

For example,
Sample input file - /Benchmarking_Datasets/Benchmark_20_neg.csv
- cd Embeddings/
- python 1_csv_to_fasta.py
- python 2_create_embeddings.py
- python 4_process_embeddings.py
- cd ../Model_testing
- Download trained model blmpred_5to60.sav.gz from the link
    https://drive.google.com/file/d/14_rVBkX3GBjEm31NZVrhkocw1PBnN8s8/view?usp=sharing
- python make_predictions.py



