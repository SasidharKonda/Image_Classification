# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import sklearn.cross_validation
import sklearn.grid_search
import sklearn.metrics
import sklearn.neighbors
import sklearn.decomposition
import sklearn
from sklearn.metrics import accuracy_score

#Set PATH for train and test files
os.chdir('E:/MS BAPM/Python/Project')

#Read Train and Test files
train_csv = pd.read_csv('fashion-mnist_train.csv')
test_csv = pd.read_csv('fashion-mnist_test.csv')

#Separate independent and dependant variables in train set
X = np.array(train_csv.ix[:,1:785])
y = np.array(train_csv['label'])

#Partition train set as training and validation
x_train, x_val, y_train, y_val = cross_validation.train_test_split(X,y,train_size=.8, test_size = 0.2,stratify = y)

#Initialize k to 10 values to use in grid search
k = np.arange(10)+1
parameters = {'n_neighbors': k}


#KNN Model
knn = KNeighborsClassifier()
#Grid Search with CV =10
clf = sklearn.grid_search.GridSearchCV(knn,parameters,cv = 10,n_jobs = -1,scoring = 'accuracy')

#Fit the model
model = clf.fit(x_train,y_train)
model

#Make predictions on the train data set
pred_train = model.predict(x_train)

#Print the accuracy of train data set
print(accuracy_score(y_train, pred_train))

#Make predictions for validation
pred_val = model.predict(x_val)

#Print accuracy of validation
print(accuracy_score(y_val, pred_val))

#Make predictions on test set
pred_test = model.predict(test_csv.ix[:,1:785])
#Print accuracy of test set
print(accuracy_score(test_csv['label'], pred_test))