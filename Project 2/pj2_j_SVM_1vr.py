import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.datasets import fetch_20newsgroups
from sklearn.multiclass import OneVsRestClassifier

#Prepare the training and testing data
tr_dataset = np.loadtxt('multitrain_out.csv', delimiter = ',')
ts_dataset = np.loadtxt('multitest_out.csv', delimiter = ',')
tr_data = tr_dataset[:, :-1]
ts_data= ts_dataset[:, :-1]
tr_label = tr_dataset[:,-1]
ts_label= ts_dataset[:,-1]

# Use one versus the rest classification method
clf = OneVsRestClassifier(LinearSVC(random_state = 0))
OnevsRest = clf.fit(tr_data, tr_label)
#Calculate the result & accuracy
result=clf.predict(ts_data)
accuracy = metrics.accuracy_score(result, ts_label)
#Calculate confusion matrix, precision & recall for each class
conf_mat=metrics.confusion_matrix(ts_label,result)
precision=metrics.precision_score(ts_label,result, average= None)
recall=metrics.recall_score(ts_label,result, average= None)

print('The accuracy is:')
print(accuracy)
print('The precision is:')
print(precision)
print('The recall is:')
print(recall)
print('The confusion matrix is:')
print(conf_mat)
