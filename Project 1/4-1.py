#4-1 Housing part I Linear Regression + Cross Validation

import numpy as np
import csv
import math
import random
import matplotlib.pyplot as plt
import pandas
from sklearn import metrics
from sklearn import model_selection
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#To run this program in another computer, you should change the path here.
with open('D:\\UCLA\\17Winter\\EE239\\Data\\housing_data.csv', newline = '') as st :
    temp = list(csv.reader(st))

data_hou = temp;
random.shuffle(data_hou)

feat_CRIM = list()
feat_ZN = list()
feat_INDUS=list()
feat_CHAS=list()
feat_NOX=list()
feat_RM=list()
feat_AGE=list()
feat_DIS=list()
feat_RAD=list()
feat_TAX=list()
feat_PTRATIO=list()
feat_B=list()
feat_LSTAT=list()
tar_MEDV=list()


feat_CRIM_a = list()
feat_ZN_a = list()
feat_INDUS_a =list()
feat_CHAS_a =list()
feat_NOX_a =list()
feat_RM_a =list()
feat_AGE_a =list()
feat_DIS_a =list()
feat_RAD_a =list()
feat_TAX_a =list()
feat_PTRATIO_a =list()
feat_B_a =list()
feat_LSTAT_a =list()
tar_MEDV_a =list()


i = 0
while i<len(data_hou):
    feat_CRIM.append(float(data_hou[i][0]))
    feat_ZN.append(float(data_hou[i][1]))
    feat_INDUS.append(float(data_hou[i][2]))
    feat_CHAS.append(float(data_hou[i][3]))
    feat_NOX.append(float(data_hou[i][4]))
    feat_RM.append(float(data_hou[i][5]))
    feat_AGE.append(float(data_hou[i][6]))
    feat_DIS.append(float(data_hou[i][7]))
    feat_RAD.append(float(data_hou[i][8]))
    feat_TAX.append(float(data_hou[i][9]))
    feat_PTRATIO.append(float(data_hou[i][10]))
    feat_B.append(float(data_hou[i][11]))
    feat_LSTAT.append(float(data_hou[i][12]))
    tar_MEDV.append(float(data_hou[i][13]))

    feat_CRIM_a.append(float(data_hou[i][0]))
    feat_ZN_a.append(float(data_hou[i][1]))
    feat_INDUS_a.append(float(data_hou[i][2]))
    feat_CHAS_a.append(float(data_hou[i][3]))
    feat_NOX_a.append(float(data_hou[i][4]))
    feat_RM_a.append(float(data_hou[i][5]))
    feat_AGE_a.append(float(data_hou[i][6]))
    feat_DIS_a.append(float(data_hou[i][7]))
    feat_RAD_a.append(float(data_hou[i][8]))
    feat_TAX_a.append(float(data_hou[i][9]))
    feat_PTRATIO_a.append(float(data_hou[i][10]))
    feat_B_a.append(float(data_hou[i][11]))
    feat_LSTAT_a.append(float(data_hou[i][12]))
    tar_MEDV_a.append(float(data_hou[i][13]))
    
    i = i + 1

LinReg = LinearRegression()
plot_y_predict = list()
plot_y_residue = list()

for m in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    n = m*50
    mean_se = 0
    y_tar = tar_MEDV[:]
    x_CRIM = feat_CRIM[:]
    x_ZN = feat_ZN[:]
    x_INDUS = feat_INDUS[:]
    x_CHAS = feat_CHAS[:]
    x_NOX = feat_NOX[:]
    x_RM = feat_RM[:]
    x_AGE = feat_AGE[:]
    x_DIS = feat_DIS[:]
    x_RAD = feat_RAD[:]
    x_TAX = feat_TAX[:]
    x_PTRATIO = feat_PTRATIO[:]
    x_B = feat_B[:]
    x_LSTAT = feat_LSTAT[:]

    del y_tar[m*50:(m+1)*50]
    del x_CRIM[m*50:(m+1)*50]
    del x_ZN[m*50:(m+1)*50]
    del x_INDUS[m*50:(m+1)*50]
    del x_CHAS[m*50:(m+1)*50]
    del x_NOX[m*50:(m+1)*50]
    del x_RM[m*50:(m+1)*50]
    del x_AGE[m*50:(m+1)*50]
    del x_DIS[m*50:(m+1)*50]
    del x_RAD[m*50:(m+1)*50]
    del x_TAX[m*50:(m+1)*50]
    del x_PTRATIO[m*50:(m+1)*50]
    del x_B[m*50:(m+1)*50]
    del x_LSTAT[m*50:(m+1)*50]

    y_curr = y_tar
    x_curr = [x_CRIM, x_ZN, x_INDUS, x_CHAS, x_NOX, x_RM, x_AGE, x_DIS, x_RAD, x_TAX, x_PTRATIO, x_B, x_LSTAT]

    y_curr_temp = np.asarray(y_curr)
    x_curr_temp = np.asarray(x_curr).T

    LinReg.fit(x_curr_temp, y_curr_temp)
    print('Coefficient is:')
    print(LinReg.coef_)

    mse = 0
    while n < 50*(m+1):
        x_check = np.asarray([feat_CRIM[n], feat_ZN[n], feat_INDUS[n], feat_CHAS[n], feat_NOX[n], feat_RM[n], feat_AGE[n], feat_DIS[n], feat_RAD[n], feat_TAX[n], feat_PTRATIO[n], feat_B[n], feat_LSTAT[n]]).reshape(1, -1)
        y_predict = LinReg.predict(x_check)
        list_y_pre = y_predict.tolist()[0]
        if m == 9:
            plot_y_predict.append(y_predict)
            plot_y_residue.append(y_predict - tar_MEDV[n])

        mse = mse + (list_y_pre - tar_MEDV[n])**2
        n = n + 1

    print ('RMSE is:')
    print (math.sqrt(mse/50)) 
    
    
    
y = tar_MEDV
x = np.matrix.transpose(np.asarray([feat_CRIM, feat_ZN, feat_INDUS, feat_CHAS, feat_NOX, feat_RM, feat_AGE, feat_DIS, feat_RAD, feat_TAX, feat_PTRATIO, feat_B, feat_LSTAT]))

result = model_selection.cross_val_predict(LinReg, x, y, cv=10)
RMSE = math.sqrt(np.mean(result))

print ('Average RMSE is:')
print (RMSE)

#Plot predicted value with original value in scattered status
plt.figure()
plt.scatter(range(50), tar_MEDV[450:500], marker = 'x', color = 'y')
plt.hold(True)
plt.scatter(range(50), plot_y_predict, marker = '+', color = 'r')
plt.title('Predicted value, original value plotted scattered')
plt.show()

#Plot predicted value with original value
plt.figure()
plt.scatter(plot_y_predict, tar_MEDV[450:500])
plt.hold(True)
plt.plot([0, 49], [0, 49], 'r-', linewidth = 2.5)
plt.axis([0, 53, 0, 53])
plt.title('Predicted value VS Original value')
plt.show()

#Plot residual value with predicted value
plt.figure()
plt.scatter(plot_y_predict, plot_y_residue)
plt.title('Residuals vs Predicted Values')
plt.show()






