# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:51:19 2018

@author: Lung
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv',header=None)
sub = pd.read_csv('sub.csv',header=None)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
#print(x.tail())
#print(np.unique(data.iloc[:,3])) #資料中此類別共有幾種
le = LabelEncoder()

x_labeled = x.copy()
test_labeled = test.copy()
available_features=[1,3,5,6,7,8,9,13]
for feature in available_features:
    le.fit(x.iloc[:,feature].values)
    x_labeled.iloc[:,feature] = le.transform(x.iloc[:,feature].values)
    test_labeled.iloc[:,feature] = le.transform(test.iloc[:,feature].values)

X_train = x_labeled
Y_train = y
X_test = test_labeled

#X_train, X_test, Y_train, Y_test = train_test_split(x_labeled, y, test_size=0.2, random_state=0)

sm = SMOTE(ratio=0.5, random_state=0)
X_train, Y_train = sm.fit_sample(X_train, Y_train)

ss = StandardScaler()
ss.fit(X_train)
x_std = ss.transform(x_labeled)
x_test_std = ss.transform(X_test)
x_train_std = ss.transform(X_train)

print('Adaboost with normalized data')
adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
adaboost.fit(x_train_std, Y_train)
y_pred = adaboost.predict(x_test_std)
'''
print('Missclassified: %d' %(Y_test != y_pred).sum())
print('Error rate: %.3f' %((Y_test != y_pred).sum()/len(y_pred)))

print('RandomForest with normalized data')
forest = RandomForestClassifier(criterion='entropy',n_estimators=100, random_state=0)
forest.fit(x_train_std, Y_train)
y_pred = forest.predict(x_test_std)
print('Missclassified: %d' %(Y_test != y_pred).sum())
print('Error rate: %.3f' %((Y_test != y_pred).sum()/len(y_pred)))
'''
df = pd.DataFrame(np.transpose(np.vstack((np.array(range(len(y_pred))),y_pred))))
df.to_csv('answer.csv',header=['ID','ans'],index=None)
