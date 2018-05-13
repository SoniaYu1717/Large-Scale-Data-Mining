# http://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

news_train=np.loadtxt('train_out.csv', delimiter=',')
news_test=np.loadtxt('test_out.csv', delimiter=',')

x_train=news_train[:, :-1]
x_test=news_test[:, :-1]
y_train=news_train[:, -1]
y_test=news_test[:, -1]

parameters = {'gamma':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

svr=SVC(probability=True, random_state=40)
clf=GridSearchCV(svr, parameters, cv=5)
clf.fit(x_train, y_train)
news_true, news_predicted=y_test, clf.predict(x_test)

print("best parameter:")
print(clf.best_params_)
print("best score:")
print(clf.best_score_)
print("accuracy:")
print(metrics.accuracy_score(news_true, news_predicted))
print("precision:")
print(metrics.precision_score(news_true, news_predicted))
print("recall:")
print(metrics.recall_score(news_true, news_predicted))
print("confusion matrix:")
print(metrics.confusion_matrix(news_true, news_predicted))

fpr, tpr, thresholds=metrics.roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
