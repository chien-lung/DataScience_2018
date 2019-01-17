# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:31:17 2019
make player to one-hot encode
ex: 2016
    input:
        5 players for each team
    ['plr_b_1','plr_b_2','plr_b_3','plr_b_4','plr_b_5','plr_r_1','plr_r_2','plr_r_3','plr_r_4','plr_r_5']
----> [0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0....,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1.....]
      -->blue team, 449 dimension and 10 elements are 1<--|-->red team, 449 dimension and 10 elements are 1<--
      1575 rows for training data (normal) (1575,898)
      40 rows for testing data (world cup) (40,898)
      
    output:
        blue team win -> 1, red team win -> 0
      
    plr_name, wc_plr_name:      player 1~5 are blue team, and player 6~10 are red team
    plr_label, wc_plr_label:    plr_name, wc_plr_name after label encode, number in [0,448](449 values)
    player_X, X_test:           plr_label, wc_plr_label after one-hot encoding (1575*898),(40*898)
    
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

filename = '2016matchdata.csv'#'2017matchdata.csv'#'2016matchdata.csv'
important_leagues = ['LMS','EULCS','NALCS','LCK','LPL','WC','MSI']
test_size = 0.2
random_state = 3

#data = pd.read_excel(filename)
data = pd.read_csv(filename)
#data = data[data['league'].isin(important_leagues)]
data['gameid'] = pd.Series(data['gameid'].astype(str))
normal_data = data.loc[~data['league'].isin(['WC'])]
wc_data = data.loc[data['league'].isin(['WC'])]
#wc_data = pd.read_csv('2018-worlds-exrtra-data.csv')
normal_not_exist_teams = set(wc_data['team']).difference(set(normal_data['team']))#['Albus NoX Luna','EDward Gaming','Royal Never Give Up','I MAY']
wc_use_data = deleteWC(wc_data, normal_not_exist_teams)
#wc_use_data = wc_use_data[~wc_use_data['week'].isin(['1.1', '1.2', '1.3', '1.4', '1.5', '1.6'])] #2017
#wc_data = wc_data[~wc_data['week'].isin(['PI', 'PI-KO'])] #2018

le = LabelEncoder()
data_plr = le.fit_transform(data['player'])
oe = OneHotEncoder()
oe.fit(data_plr.reshape(-1,1))

##################
#training, player#
##################
#player_X
plr_name = pd.DataFrame()
plr_label = pd.DataFrame()
plr_b_one_hot = np.zeros((int(normal_data.shape[0]/12),len(oe.categories_[0]))) #[1575,449]
plr_r_one_hot = plr_b_one_hot.copy()
for plrid in range(1,11):
    plr = normal_data.loc[normal_data['playerid'].isin([plrid])]['player']
    plr_name['player{}'.format(plrid)] = pd.Series(np.array(plr))#player name
    plr_arr = le.transform(plr)
    plr_label['player{}'.format(plrid)] = pd.Series(plr_arr) #player label
    if plrid <= 5: #blue team player
        plr_b_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
    else: #red team player
        plr_r_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
plr_X = np.concatenate((plr_b_one_hot, plr_r_one_hot), axis=1) #player one hot
#player_y
b_result = np.array(normal_data.loc[normal_data['playerid'].isin([100])]['result'])
plr_y = b_result

#################
#testing, player#
#################
#wc_player_X
wc_plr_name = pd.DataFrame()
wc_plr_label = pd.DataFrame()
wc_plr_b_one_hot = np.zeros((int(wc_use_data.shape[0]/12),len(oe.categories_[0]))) #[40,449]
wc_plr_r_one_hot = wc_plr_b_one_hot.copy()
for plrid in range(1,11):
    plr = wc_use_data.loc[wc_use_data['playerid'].isin([plrid])]['player']
    wc_plr_name['player{}'.format(plrid)] = pd.Series(np.array(plr))#player name
    plr_arr = le.transform(plr)
    wc_plr_label['player{}'.format(plrid)] = pd.Series(plr_arr) #player label
    if plrid <= 5: #blue team player
        wc_plr_b_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
    else:#red team player
        wc_plr_r_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
X_test = np.concatenate((wc_plr_b_one_hot, wc_plr_r_one_hot), axis=1) #player one hot
#player_y
wc_b_result = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([100])]['result'])
y_test = wc_b_result
'''
X_train, X_val, y_train, y_val = train_test_split(plr_X, plr_y, test_size=test_size, random_state=random_state)

print("Logistic Regression:")
logreg = LogisticRegression()
getScore(logreg, test_size, X_train, y_train, X_val, y_val, X_test, y_test)

print("DecisionTreeClassifier:")
tree = DecisionTreeClassifier()
getScore(tree, test_size, X_train, y_train, X_val, y_val, X_test, y_test)

print('Adaboost:')
adaboost = AdaBoostClassifier()
getScore(adaboost, test_size, X_train, y_train, X_val, y_val, X_test, y_test)

print('RandomForestClassifier:')
forest = RandomForestClassifier()
getScore(forest, test_size, X_train, y_train, X_val, y_val, X_test, y_test)
'''
score = [0,0,0,0]
X_train, X_val, y_train, y_val = train_test_split(plr_X, plr_y, test_size=test_size)

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