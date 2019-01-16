# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:35:24 2018

@author: Lung
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#import matplotlib.pylab as plt
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv', header=None)

x = data.iloc[:,:-1]
#print(x.tail())
#print(np.unique(data.iloc[:,3])) #資料中此類別共有幾種

le = LabelEncoder()
y = le.fit_transform(data.iloc[:,-1].values)

x_labeled = x.copy()
for feature in range(x.shape[1]):
    x_labeled.iloc[:,feature] = le.fit_transform(x.iloc[:,feature].values)

X_train, X_test, Y_train, Y_test = train_test_split(x_labeled, y, test_size=0.2, random_state=0)

tree1 = DecisionTreeClassifier(criterion='entropy')
tree1.fit(X_train,Y_train)
y_pred = tree1.predict(X_test)
print('Missclassified: %d' %(Y_test != y_pred).sum())

ss = StandardScaler()
ss.fit(X_train)
x_std = ss.transform(x_labeled)
x_test_std = ss.transform(X_test)
x_train_std = ss.transform(X_train)

tree2 = DecisionTreeClassifier(criterion='entropy')
tree2.fit(x_train_std, Y_train)
y_pred = tree2.predict(x_test_std)
print('Missclassified: %d' %(Y_test != y_pred).sum())