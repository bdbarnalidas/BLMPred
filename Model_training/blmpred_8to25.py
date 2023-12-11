from cuml.svm import SVC
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from math import sqrt
import matplotlib.pyplot as plt
from collections import Counter
from numpy import where
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
import random
from sklearn.model_selection import KFold, cross_val_score

def mcc(tp, fp, tn, fn):
    result = []
    for i in range(0,len(tp)):
        x = (tp[i] + fp[i]) * (tp[i] + fn[i]) * (tn[i] + fp[i]) * (tn[i] + fn[i])
        y = ((tp[i] * tn[i]) - (fp[i] * fn[i])) / sqrt(x)
        result.append(y)
    return result

def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],'fn': cm[1, 0], 'tp': cm[1, 1]}


with open('blmpred8to25.npy', 'rb') as f: # Load the features
    features = np.load(f)

labels = np.zeros(489178)

print('Features Shape:', features.shape)
print('Labels Shape:', labels.shape)

for i in range(0,102023):
	labels[i] = 1

X = features # feature vector
y = labels	#label

# oversample = SMOTE()
# X, y = oversample.fit_resample(X, y)

# print('Features Shape:', features.shape)
# print('Labels Shape:', labels.shape)

positive = list(range(102023))
negative = list(range(102023,489178)) # full dataset
pos = positive
# random.shuffle(pos)
neg = negative
# random.shuffle(neg) # shuffling full dataset

# blind data
test_pos = pos[:10202] # test data
test_neg = neg[:10202]
test_indices = test_pos + test_neg
random.shuffle(test_indices)
X_test = np.array([X[i] for i in test_indices])
y_test = np.array([y[i] for i in test_indices])
test_features = X_test
test_labels = y_test

# CV data
cv_indices = pos[10202:]+neg[10202:102023] # remaining
random.shuffle(cv_indices)

model = SVC(kernel='rbf', C=10, gamma=1, cache_size=2000)

trainingfeat = X[cv_indices]
traininglab = y[cv_indices]

finmodel = model.fit(trainingfeat, traininglab) 
filename = 'blmpred_8to25.sav'
pickle.dump(finmodel, open(filename, 'wb'))

new_predictions = finmodel.predict(test_features)
# with open('testinglabels','wb') as f: pickle.dump(new_predictions, f)
# with open('truetestinglabels','wb') as f: pickle.dump(test_labels, f)
        
print('\n--------------------------Testing-------------------------------\n')
acc = accuracy_score(test_labels, new_predictions)
print("Accuracy Score = " + str(acc)) 
prec = precision_score(test_labels, new_predictions)
print("Precision Score = " + str(prec)) 
rec = recall_score(test_labels, new_predictions)
print("Recall Score = " + str(rec)) 
f1 = f1_score(test_labels, new_predictions)
print("F1 Score = " + str(f1) ) 

tn, fp, fn, tp = confusion_matrix(test_labels, new_predictions).ravel() # Confusion matrix
specificity = tn / (tn+fp) # Specificity
print('Specificity = ' + str(specificity))
print('TP = ' + str(tp)) # TP
print('FP = ' + str(fp)) # FP
print('TN = ' + str(tn)) # TN
print('FN = ' + str(fn)) # FN

mcc_val = matthews_corrcoef(test_labels, new_predictions) # MCC
print('MCC = ' + str(mcc_val))

auroc = roc_auc_score(test_labels, new_predictions) # AUROC
print('AUROC = ' + str(auroc))
auprc = average_precision_score(test_labels, new_predictions) # AUPRC
print('AUPRC = ' + str(auprc)) 