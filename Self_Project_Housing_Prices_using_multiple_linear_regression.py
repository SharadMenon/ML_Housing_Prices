# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 01:54:55 2024

@author: Shyam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Housing.csv")
X = data.iloc[:, 1:].values
y = data.iloc[:, 0:1] .values

#encoding categorical data like mainroad/guestroom/basementtwaterheating/conditioning/prefarea/furnishihng status
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
columns_to_encode = [4, 5, 6, 7, 8, 10, 11]
for col in columns_to_encode:
    X[:, col] = LabelEncoder().fit_transform(X[:, col])
column_transformer = ColumnTransformer(
    [("one_hot_encoder", OneHotEncoder(), columns_to_encode)],
    remainder='passthrough'  # Keep the rest of the columns unchanged
)

X = column_transformer.fit_transform(X)
#To avoid dummy variable trap
X = X[:, 1:]
#We will build our optimal model using backwards elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((545,1)).astype(int),values=X, axis=1)
#Now we have added 1 column of 1's to X at the beg so that statsmodel recognizes the multiple regression pattern with constant b0x0 and x0=1 this is required by the statsmodel module
#Step 2
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
 
SL = 0.05
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
X_opt = X_opt.astype(float)
X_Modeled = backwardElimination(X_opt, SL)

#Now we will split the modelled data set in to training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_Modeled,y,test_size=1/5,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the training set result
y_pred = regressor.predict(X_test)
#Visualizing on graph for test set
plt.scatter(y_test,y_pred,color='red')
plt.title("Actual vs predicted price(test set)")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.show()
#Visualizing on graph for Training set
y_pred_train = regressor.predict(X_train)
plt.scatter(y_train,y_pred_train,color='red')
plt.title("Actual vs predicted price(Training set)")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.show()