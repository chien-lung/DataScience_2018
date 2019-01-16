# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 20:26:25 2019
difference:
    StandardScaler for competition data
    
@author: Lung
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def deleteWC(wc_data, normal_not_exist_teams):
    #delete the world cup team which not exist in normal game
    wc_not_data = wc_data.loc[wc_data['team'].isin(normal_not_exist_teams)]
    wc_use_data = wc_data.loc[~wc_data['gameid'].isin(np.unique(wc_not_data['gameid']))]
    return wc_use_data

def makeNormalDataAvgRecords(normal_data, important_records):
    teams = np.unique(normal_data['team'])
    normal_team_data = normal_data[normal_data['playerid'].isin([100,200])] #team:100 -> blue; 200 -> red
    std = StandardScaler()
    normal_team_data_std = std.fit_transform(normal_data[important_records])
    normal_team_data_std = pd.DataFrame(normal_team_data_std, columns=important_records)
    team_avg_records_df = pd.DataFrame(np.zeros((len(teams),len(important_records))), index=np.unique(normal_data['team']), columns=important_records)
    team_records_count = pd.DataFrame(np.zeros((len(teams),1)), index=np.unique(normal_data['team']), columns=['times'])
    for i in range(normal_team_data.shape[0]):
        team_name = normal_team_data.iloc[i]['team']
        team_records_count.loc[team_name] += 1
    for i in range(normal_team_data.shape[0]):
        team_name = normal_team_data.iloc[i]['team']
        team_avg_records_df.loc[team_name] += normal_team_data_std[important_records].iloc[i]/team_records_count.loc[team_name]['times']
    return team_avg_records_df

def getTeamAvgRecords(wc_team_data, team_avg_records):
    wc_team_records = pd.DataFrame(index=wc_team_data.index, columns=important_records)
    for i in range(wc_team_data.shape[0]):
        team_name = wc_team_data.iloc[i]['team']
        wc_team_records.iloc[i] = team_avg_records.loc[team_name]
    return wc_team_records

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

filename = '2017matchdata.csv'#'2017matchdata.csv'#'2016matchdata.csv'
important_leagues = ['EULCS','NALCS','LCK','LPL','WC','LMS','MSI']
important_records = ['teamtowerkills','earnedgpm','goldspent','gspd', #>0.7
                     'fbaron','k','a','kpm','teambaronkills', #>0.6
                     'teamdragkills','firsttothreetowers', #>0.5
                     'gdat15','goldat15','dmgtochampsperminute','totalgold','xpdat10','ft',#>0.3
                     'goldat10','monsterkills','csdat10','dmgtochamps','cspm',#>0.2                    
                     'herald','wcpm','xpat10','csat10',#>0.1
                     ]
test_size = 0.0
random_state = 3

data = pd.read_csv(filename)
#data = data[data['league'].isin(important_leagues)]

data['gameid'] = pd.Series(data['gameid'].astype(str))
normal_data = data.loc[~data['league'].isin(['WC'])]
wc_data = data.loc[data['league'].isin(['WC'])]
#wc_data = pd.read_csv('2018-worlds-exrtra-data.csv')
normal_not_exist_teams = set(wc_data['team']).difference(set(normal_data['team']))#['Albus NoX Luna','EDward Gaming','Royal Never Give Up','I MAY']
wc_data = deleteWC(wc_data, normal_not_exist_teams)
#wc_data = wc_data[~wc_data['week'].isin(['1.1', '1.2', '1.3', '1.4', '1.5', '1.6'])] #2017
#wc_data = wc_data[~wc_data['week'].isin(['PI', 'PI-KO'])] #2018

#encode records to train #2017
normal_mean = normal_data[important_records].mean()
normal_data = normal_data.fillna(normal_mean)
#wc_data = wc_data.fillna(normal_mean)

std = StandardScaler()
std.fit(normal_data[important_records])

team_normal_avg_records = makeNormalDataAvgRecords(normal_data, important_records)
normal_b_team_records = getTeamAvgRecords(normal_data[normal_data['playerid'].isin([100])], team_normal_avg_records)#normal_data[normal_data['playerid'].isin([100])][important_records]
normal_r_team_records= getTeamAvgRecords(normal_data[normal_data['playerid'].isin([200])], team_normal_avg_records)#normal_data[normal_data['playerid'].isin([200])][important_records]
wc_b_team_records = getTeamAvgRecords(wc_data[wc_data['playerid'].isin([100])], team_normal_avg_records)#wc_data[wc_data['playerid'].isin([100])][important_records]
wc_r_team_records = getTeamAvgRecords(wc_data[wc_data['playerid'].isin([200])], team_normal_avg_records)#wc_data[wc_data['playerid'].isin([200])][important_records]

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
plr_b_one_hot = np.zeros((int(normal_data.shape[0]/12),len(oe.categories_[0]))) 
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
plr_X = np.concatenate((plr_b_one_hot, normal_b_team_records, plr_r_one_hot, normal_r_team_records), axis=1) #player one hot
#player_y
b_result = np.array(normal_data.loc[normal_data['playerid'].isin([100])]['result'])
plr_y = b_result

        
#################
#testing, player#
#################
#wc_player_X
wc_plr_name = pd.DataFrame()
wc_plr_label = pd.DataFrame()
wc_plr_b_one_hot = np.zeros((int(wc_data.shape[0]/12),len(oe.categories_[0]))) 
wc_plr_r_one_hot = wc_plr_b_one_hot.copy()
for plrid in range(1,11):
    plr = wc_data.loc[wc_data['playerid'].isin([plrid])]['player']
    wc_plr_name['player{}'.format(plrid)] = pd.Series(np.array(plr))#player name
    plr_arr = le.transform(plr)
    wc_plr_label['player{}'.format(plrid)] = pd.Series(plr_arr) #player label
    if plrid <= 5: #blue team player
        wc_plr_b_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
    else:#red team player
        wc_plr_r_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
X_test = np.concatenate((wc_plr_b_one_hot, wc_b_team_records, wc_plr_r_one_hot, wc_r_team_records), axis=1) #player one hot
#player_y
wc_b_result = np.array(wc_data.loc[wc_data['playerid'].isin([100])]['result'])
y_test = wc_b_result

########################
#Train & Test & Scoring#
########################
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
times = 30
score = [0,0,0,0]
for i in range(times):
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