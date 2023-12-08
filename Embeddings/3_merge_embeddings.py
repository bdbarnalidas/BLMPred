import h5py
import numpy as np
import pickle

file_with_positive_features = "Benchmark_20_pos.h5" # input 1 (epitope embeddings)
file_with_negative_features = "Benchmark_20_neg.h5" # input 2 (non-epitope embeddings)
output_file = 'Benchmark_20.npy' # input 3 (output file name)

pos_features = np.zeros(shape=(74,1024)) # Positive samples = 74 # input 4 (#epitopes, embedder shape is #epitopes X 1024)
neg_features = np.zeros(shape=(21,1024)) # Negative samples = 21 # input 5 (#non-epitopes, embedder shape is #non-epitopes X 1024)

# print('Reading embeddings of positive dataset and storing them as numpy array')
with h5py.File(file_with_positive_features, "r") as f: # Reading embeddings of positive dataset and storing them as numpy array
    for i in range(0, len(f.keys())):
        a_group_key = list(f.keys())[i]
        data = np.array(f[a_group_key])
        pos_features[i] = data
# print('Reading embeddings of negative dataset and storing them as numpy array')
with h5py.File(file_with_negative_features, "r") as f: # Reading embeddings of negative dataset and storing them as numpy array
    for i in range(0, len(f.keys())):
        a_group_key = list(f.keys())[i]
        data = np.array(f[a_group_key])
        neg_features[i] = data
features = np.concatenate((pos_features, neg_features), axis=0)
with open(output_file, 'wb') as f: # Store the features in a numpy file containing positive features + negative features
    np.save(f, features)