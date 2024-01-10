import h5py
import numpy as np
import pickle

file_with_features = "Benchmark_20_pos.h5" # input 1 (embeddings)
output_file = 'test_20.npy' # input 2 (output file name)
peptide_file = open('peptide_list.txt','w') # input 3 (file storing peptide ids)

features = np.zeros(shape=(74,1024)) # input 4 (how many peptides? put that number in place of 74)

with h5py.File(file_with_features, "r") as f: # Reading embeddings of dataset and storing them as numpy array
    for i in range(0, len(f.keys())):
        a_group_key = list(f.keys())[i]
        peptide_file.write(a_group_key + '\n')
        data = np.array(f[a_group_key])
        features[i] = data
with open(output_file, 'wb') as f: # Store the features in a numpy file 
    np.save(f, features)