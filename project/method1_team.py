# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:19:12 2019
method1: only depend on "team"

@author: Lung
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def deleteWC(wc_data, normal_not_exist_teams):
    #delete the world cup team which not exist in normal game
    wc_not_data = wc_data.loc[wc_data['team'].isin(normal_not_exist_teams)]
    wc_use_data = wc_data.loc[~wc_data['gameid'].isin(np.unique(wc_not_data['gameid']))]
    return wc_use_data

def getScore(model, test_size, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print("Training set accuracy: ", '%.3f'%(score))
    if test_size > 0:
        score = model.score(X_val, y_val)
        print("Validation set accuracy: ", '%.3f'%(score))
    score = model.score(X_test, y_test)
    print("WC set accuracy: ", '%.3f'%(score))
    print()
    return score

filename = '2018matchdata.csv' #'2017matchdata.csv' #'2016matchdata.csv'
test_size = 0
random_state = 8

#data = pd.read_excel(filename)
data = pd.read_csv(filename)
data['gameid'] = pd.Series(data['gameid'].astype(str))
data = data[data['league'].isin(['EULCS','NALCS','LCK','LPL','WC'])]
#data = data[data['league'].isin(['LMS','EULCS','NALCS','LCK','LPL','WC'])]
normal_data = data.loc[~data['league'].isin(['WC'])]
wc_data = data.loc[data['league'].isin(['WC'])]
normal_not_exist_teams = set(wc_data['team']).difference(set(normal_data['team']))#['Albus NoX Luna','EDward Gaming','Royal Never Give Up','I MAY']
wc_use_data = deleteWC(wc_data, normal_not_exist_teams)

#team name --label encode--> value --one-hot encode--> one hot value
le = LabelEncoder()
normal_data_team = le.fit_transform(normal_data['team'])
oe = OneHotEncoder()
normal_data_team_oe = oe.fit_transform(normal_data_team.reshape(-1, 1))

###############
#training data#
###############

b = np.array(normal_data.loc[normal_data['playerid'].isin([100])]['team'])
r = np.array(normal_data.loc[normal_data['playerid'].isin([200])]['team'])
b_result = np.array(normal_data.loc[normal_data['playerid'].isin([100])]['result'])
r_result = np.array(normal_data.loc[normal_data['playerid'].isin([200])]['result'])

b_one_hot = oe.transform(le.transform(b).reshape(-1,1)).toarray()
r_one_hot = oe.transform(le.transform(r).reshape(-1,1)).toarray()
team_X = np.concatenate((b_one_hot,r_one_hot), axis=1) #[blue team,red team]
team_y = b_result

##############
#testing data#
##############
wc_b = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([100])]['team'])
wc_r = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([200])]['team'])
wc_b_result = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([100])]['result'])
wc_r_result = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([200])]['result'])

wc_b_one_hot = oe.transform(le.transform(wc_b).reshape(-1,1)).toarray()
wc_r_one_hot = oe.transform(le.transform(wc_r).reshape(-1,1)).toarray()
X_test = np.concatenate((wc_b_one_hot,wc_r_one_hot), axis=1) #[blue team,red team]
y_test = wc_b_result

times = 50
score = [0,0,0,0]
for i in range(times):
    X_train, X_val, y_train, y_val = train_test_split(team_X, team_y, test_size=test_size)
    
    print("Logistic Regression:")
    logreg = LogisticRegression()
    score[0] += getScore(logreg, test_size, X_train, y_train, X_val, y_val, X_test, y_test)/times
    
    print("DecisionTreeClassifier:")
    tree = DecisionTreeClassifier()
    score[1] += getScore(tree, test_size, X_train, y_train, X_val, y_val, X_test, y_test)/times
    
    print('Adaboost:')
    adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                                      n_estimators=40,
                                      learning_rate=3)
    score[2] += getScore(adaboost, test_size, X_train, y_train, X_val, y_val, X_test, y_test)/times
    
    print('RandomForestClassifier:')
    forest = RandomForestClassifier()
    score[3] += getScore(forest, test_size, X_train, y_train, X_val, y_val, X_test, y_test)/times