from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
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


with open('blmpred5to60.npy', 'rb') as f: # Load the features
    features = np.load(f)

labels = np.zeros(501604)

print('Features Shape:', features.shape)
print('Labels Shape:', labels.shape)

for i in range(0,111015):
	labels[i] = 1

X = features # feature vector
y = labels	#label

positive = list(range(111015))
negative = list(range(111015,501604)) # full dataset
pos = positive
# random.shuffle(pos)
neg = negative
# random.shuffle(neg) # shuffling full dataset

# blind data
test_pos = pos[:11102] # test data
test_neg = neg[:11102]
test_indices = test_pos + test_neg
random.shuffle(test_indices)
X_test = np.array([X[i] for i in test_indices])
y_test = np.array([y[i] for i in test_indices])
test_features = X_test
test_labels = y_test

# CV data
cv_indices = pos[11102:]+neg[11102:111015] # remaining
random.shuffle(cv_indices)

model = BaggingClassifier(SVC(),max_samples=0.5, max_features=0.5)
k_fold = KFold(n_splits=10)
j = 1
axis = 0

val_acc = []
val_prec = []
val_rec = []
val_f1 = []	
val_spec = []
val_mcc = []
val_auroc = []
val_auprc = []

test_acc = []
test_prec = []
test_rec = []
test_f1 = []	
test_spec = []
test_mcc = []
test_auroc = []
test_auprc = []


for train_indices, val_indices in k_fold.split(cv_indices):
	print('\n------------------------- Fold ' + str(j) + ' ------------------------------------\n')
	trainingfeat = np.take(X[cv_indices],train_indices,axis)
	traininglab = np.take(y[cv_indices],train_indices,axis)
	validationfeat = np.take(X[cv_indices],val_indices,axis)
	validationlab = np.take(y[cv_indices],val_indices,axis)
	
	finmodel = model.fit(trainingfeat, traininglab) 
	# filename = 'finalized_model_' + str(j) + '.sav'
	# pickle.dump(model, open(filename, 'wb'))

	new_predictions = finmodel.predict(validationfeat)

	print('\n--------------------------Validation-------------------------------\n')
	acc = accuracy_score(validationlab, new_predictions)
	print("Accuracy Score = " + str(acc)) 
	prec = precision_score(validationlab, new_predictions)
	print("Precision Score = " + str(prec)) 
	rec = recall_score(validationlab, new_predictions)
	print("Recall Score = " + str(rec)) 
	f1 = f1_score(validationlab, new_predictions)
	print("F1 Score = " + str(f1)) 

	tn, fp, fn, tp = confusion_matrix(validationlab, new_predictions).ravel() # Confusion matrix
	specificity = tn / (tn+fp) # Specificity
	print('Specificity = ' + str(specificity))
	print('TP = ' + str(tp)) # TP
	print('FP = ' + str(fp)) # FP
	print('TN = ' + str(tn)) # TN
	print('FN = ' + str(fn)) # FN

	mcc_val = matthews_corrcoef(validationlab, new_predictions) # MCC
	print('MCC = ' + str(mcc_val))

	auroc = roc_auc_score(validationlab, new_predictions) # AUROC
	print('AUROC = ' + str(auroc))
	auprc = average_precision_score(validationlab, new_predictions) # AUPRC
	print('AUPRC = ' + str(auprc)) 

	new_predictions = finmodel.predict(test_features)
        
	val_acc.append(acc)
	val_prec.append(prec)
	val_rec.append(rec)
	val_f1.append(f1)
	val_spec.append(specificity)
	val_mcc.append(mcc_val)
	val_auroc.append(auroc)
	val_auprc.append(auprc)

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
        
	test_acc.append(acc)
	test_prec.append(prec)
	test_rec.append(rec)
	test_f1.append(f1)
	test_spec.append(specificity)
	test_mcc.append(mcc_val)
	test_auroc.append(auroc)
	test_auprc.append(auprc)

	j = j+1

print("--------------Validation----------------\n")
print("Mean accuracy = " + str(np.mean(val_acc)))
print("Std accuracy = " + str(np.std(val_acc)))
print("Mean precision = " + str(np.mean(val_prec)))
print("Std precision = " + str(np.std(val_prec)))
print("Mean recall = " + str(np.mean(val_rec)))
print("Std recall = " + str(np.std(val_rec)))
print("Mean f1 = " + str(np.mean(val_f1)))
print("Std f1 = " + str(np.std(val_f1)))
print("Mean specificity = " + str(np.mean(val_spec)))
print("Std specificity = " + str(np.std(val_spec)))
print("Mean mcc = " + str(np.mean(val_mcc)))
print("Std mcc = " + str(np.std(val_mcc)))
print("Mean auroc = " + str(np.mean(val_auroc)))
print("Std auroc = " + str(np.std(val_auroc)))
print("Mean auprc = " + str(np.mean(val_auprc)))
print("Std auprc = " + str(np.std(val_auprc)))

print("--------------Testing----------------\n")
print("Mean accuracy = " + str(np.mean(test_acc)))
print("Std accuracy = " + str(np.std(test_acc)))
print("Mean precision = " + str(np.mean(test_prec)))
print("Std precision = " + str(np.std(test_prec)))
print("Mean recall = " + str(np.mean(test_rec)))
print("Std recall = " + str(np.std(test_rec)))
print("Mean f1 = " + str(np.mean(test_f1)))
print("Std f1 = " + str(np.std(test_f1)))
print("Mean specificity = " + str(np.mean(test_spec)))
print("Std specificity = " + str(np.std(test_spec)))
print("Mean mcc = " + str(np.mean(test_mcc)))
print("Std mcc = " + str(np.std(test_mcc)))
print("Mean auroc = " + str(np.mean(test_auroc)))
print("Std auroc = " + str(np.std(test_auroc)))
print("Mean auprc = " + str(np.mean(test_auprc)))
print("Std auprc = " + str(np.std(test_auprc)))