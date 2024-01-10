from cuml.svm import SVC
import numpy as np
import pickle

with open('../Embeddings/Benchmark_20.npy', 'rb') as f: # Load the features/embeddings
    features = np.load(f)

loaded_model = pickle.load(open('blmpred_5to60.sav', 'rb')) # Load the model
peptide_filename = open('../Embeddings/peptide_list.txt','r') 
peptide_list = []

predictions_file = open('predictions.csv','w')

for ln in peptide_filename:
    ln = ln.replace('\n','')
    peptide_list.append(ln)

new_predictions = loaded_model.predict(features)

int_list = [int(item) for item in new_predictions]

# print(len(int_list))
# print(len(peptide_list))

for i in range(0,len(peptide_list)):
    if int_list[i] == 0:
        predictions_file.write(peptide_list[i] + ',' + 'Non-epitope' + '\n')
    elif int_list[i] == 1:
        predictions_file.write(peptide_list[i] + ',' + 'Epitope' + '\n')