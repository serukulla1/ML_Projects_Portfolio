# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 21:19:23 2025

@author: swapna
"""

##Multi Linear Regression


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Swapna\Learning\PYTHON\FS DataScience - Theory\ML\10th- mlr\MLR\Investment.csv')

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

X = pd.get_dummies(X, dtype=int)  ##To convert categorical values in state column to numbers

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

bias = regressor.score(X_train, y_train)
bias

variance = regressor.score(X_test, y_test)
variance

slope = regressor.coef_  ## value of m, which means how much y increases when x=1

intercept = regressor.intercept_  ##value of c, when x=0

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)

import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())


