# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:17:38 2025

@author: swapna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### The dataset contains information about users with their age and salary, the dependent feature is the purchased column shows categorical values of purchased 'yes' or 'no'
###yes - 1; no - 0
###We are building a model that can predict wether a user could purchase SUV or not based on their sal and age.   
### We need to find a relation between age, estimated sal of the user and thier decision to purchase the car.
dataset = pd.read_csv(r'C:\Swapna\Learning\PYTHON\FS DataScience\ML Projects\LogisticRegression - 2\Data\Social_Network_Ads.csv')

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values


##Feature scaling

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Following steps will build logistic model from the traing set data

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


#Predicting the Test set results
y_pred=classifier.predict(X_test)   ## Compare the y_pred with y_test data 


# with confusion matrx we calculate the model accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix", cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("accuracy score", ac)

## Classification report, bas and variance
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print("classification report", cr)

bias = classifier.score(X_train, y_train)
print("Bias", bias)

variance = classifier.score(X_test, y_test)
print("variance", variance)

dataset1 = pd.read_csv(r'C:\Swapna\Learning\PYTHON\FS DataScience\ML Projects\LogisticRegression - 2\Data\Future prediction1.csv')

X = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, -1].values
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, [2,3]].values

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
M = sc.fit_transform(dataset1)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred1 = pd.DataFrame()
d2['y_pred1'] = classifier.predict(M)
d2.to_csv('predicted output.csv')

## Visualising the Test set results

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() -1, stop = X_set[:, 0].max() + 1, step=.01),
                     np.arange(start = X_set[:,1].min() -1, stop =X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
   
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




