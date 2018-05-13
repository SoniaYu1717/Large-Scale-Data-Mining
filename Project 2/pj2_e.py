import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#Prepare the training and testing data
tr_dataset = np.genfromtxt('train_out.csv', delimiter = ',')
ts_dataset = np.genfromtxt('test_out.csv', delimiter = ',')
tr_data = tr_dataset[:, :-1]
ts_data= ts_dataset[:, :-1]
tr_label = tr_dataset[:,-1]
ts_label= ts_dataset[:,-1]

#Use the SVC method, penalty parameter C =1.0, kernel type linear and enable probability estimates
clf = SVC(C = 1.0, kernel = 'linear', probability=True)
#Fit the model
svm = clf.fit(tr_data, tr_label)
#Calculate the result & accuracy
result=clf.predict(ts_data)
accuracy=clf.score(ts_data,ts_label)
#Calculate the probability estimates of the positive class
prob_data=clf.predict_proba(ts_data)
prob_data=prob_data[:,1]
#Calculate fpr & ftr 
fpr,tpr,thresholds=metrics.roc_curve(ts_label,prob_data)
#Calculate confusion matrix, precision & recall
conf_mat=metrics.confusion_matrix(ts_label,result)
precision=metrics.precision_score(ts_label,result)
recall=metrics.recall_score(ts_label,result)
#Calculate auc
roc_auc=metrics.auc(fpr,tpr)
print(roc_auc)

print('The accuracy is:')
print(accuracy)
print('The precision is:')
print(precision)
print('The recall is:')
print(recall)
print('The confusion matrix is:')
print(conf_mat)
print('The range of threshold is:')
print(thresholds[0])
print(thresholds[len(thresholds)-1])

plt.figure()
plt.plot(fpr, tpr, color='deeppink', lw=3, label='ROC curve (area = %0.4f )' %roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


