# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:49:29 2017

@author: pshrivas
"""

import numpy as np
import pandas as pd
train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")
train_y = train_set.iloc[:,0].values
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
training_set = train_set.iloc[:,1:].values
x_train = sc_x.fit_transform(training_set)
x_test = sc_x.transform(test_set)

from sklearn.decomposition.pca import PCA
pca = PCA(n_components=15)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

explained_valriance = pca.explained_variance_ratio_

##Fitting Logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,train_y)
#
#y_pred = classifier.predict(x_test)
#digit = pd.DataFrame()
#digit["ImageId"]=range(1,len(test_set)+1)
#digit["Label"] = y_pred
#digit.to_csv("sub4.csv",index=False)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy",random_state=0)
classifier.fit(x_train,train_y)


#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = train_y, cv = 10)
accuracies.mean()
accuracies.std()
#Fitting Random Forest



y_pred = classifier.predict(x_test)

digit = pd.DataFrame()
digit["ImageId"]=range(1,len(test_set)+1)
digit["Label"] = y_pred
digit.to_csv("subl280.csv",index=False)