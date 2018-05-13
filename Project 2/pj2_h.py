import numpy as np
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt

news_train=np.loadtxt('train_out.csv', delimiter=',')
news_test=np.loadtxt('test_out.csv', delimiter=',')

x_train=news_train[:, :-1]
x_test=news_test[:, :-1]
y_train=news_train[:, -1]
y_test=news_test[:, -1]

model=linear_model.LogisticRegression() #using default parameters
model.fit(x_train, y_train)

news_true=y_test;
news_predict=model.predict(x_test)

fpr, tpr, thresholds=metrics.roc_curve(y_test, model.predict_proba(x_test)[:, 1])

print("accuracy:")
print(metrics.accuracy_score(news_true, news_predict))
print("precision:")
print(metrics.precision_score(news_true, news_predict))
print("recall:")
print(metrics.recall_score(news_true, news_predict))
print("confusion matrix:")
print(metrics.confusion_matrix(news_true, news_predict))

plt.figure()
plt.plot(fpr, tpr, label="ROC CURVE")
plt.plot([0,1], [0,1], '--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE-logistic regression')
plt.legend(loc="lower right")
plt.show()
