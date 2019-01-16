# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 00:03:35 2019

@author: Lung
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def deleteWC(wc_data, normal_not_exist_teams):
    wc_not_data = wc_data.loc[wc_data['team'].isin(normal_not_exist_teams)]
    wc_use_data = wc_data.loc[~wc_data['gameid'].isin(np.unique(wc_not_data['gameid']))]
    return wc_use_data

filename = '2016 complete match data OraclesElixir 2018-12-18.xlsx'
stat_list = ['player','side','team','champion','position']

data = pd.read_excel(filename)
normal_data = data.loc[~data['league'].isin(['WC'])]
wc_data = data.loc[data['league'].isin(['WC'])]
normal_not_exist_teams = set(wc_data['team']).difference(set(normal_data['team']))#['Albus NoX Luna','EDward Gaming','Royal Never Give Up','I MAY']
wc_use_data = deleteWC(wc_data, normal_not_exist_teams)

b = np.array(normal_data.loc[normal_data['playerid'].isin([100])]['team'])
b_r = np.array(normal_data.loc[normal_data['playerid'].isin([100])]['result'])
r = np.array(normal_data.loc[normal_data['playerid'].isin([200])]['team'])
r_r = np.array(normal_data.loc[normal_data['playerid'].isin([200])]['result'])
team_stat = pd.DataFrame({'B_team':b,'R_team':r})
team_X = pd.get_dummies(team_stat, columns=['B_team','R_team'])
team_y = b_r

wc_b = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([100])]['team'])
wc_b_r = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([100])]['result'])
wc_r = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([200])]['team'])
wc_r_r = np.array(wc_use_data.loc[wc_use_data['playerid'].isin([200])]['result'])
wc_stat = pd.DataFrame({'B_team':wc_b,'R_team':wc_r})
X_test = pd.get_dummies(wc_stat, columns=['B_team','R_team'])
y_test = wc_b_r


X_train, X_val, y_train, y_val = train_test_split(team_X, team_y, test_size=0.3, random_state=36)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
print("Training set accuracy: ", '%.3f'%(score))
score = logreg.score(X_val, y_val)
print("Validation set accuracy: ", '%.3f'%(score))
score = logreg.score(X_test, y_test)
print("WC set accuracy: ", '%.3f'%(score))


'''
team_stat = pd.DataFrame({'B_team':np.concatenate((b,r)), 'R_team':np.concatenate((r,b))})
team_X = pd.get_dummies(team_stat, columns=['B_team','R_team'])
team_y = np.concatenate((b_r,r_r))

X_train, X_val, y_train, y_val = train_test_split(team_X, team_y, test_size=0.3, random_state=36)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Training set accuracy: ", '%.3f'%(score))
score = logreg.score(X_val, y_val)
print("Validation set accuracy: ", '%.3f'%(score))
#score = logreg.score(X_test, y_test)
#print("WC set accuracy: ", '%.3f'%(score))
'''


'''
for plrid in range(1,6):
    plr = data.loc[data['playerid'].isin([plrid, plrid+5])]['player']
    plr_arr = np.array(plr)
    team_stat['player{}'.format(plrid)] = pd.Series(plr_arr, index=team_stat.index)

#Team
team_X = pd.get_dummies(team_stat)
X_train, X_test, y_train, y_test = train_test_split(team_X, team_result, test_size=0.25, random_state=36)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))

#Players
plr_X = pd.get_dummies(plr_stat, prefix=stat_list, columns=stat_list)
X_train, X_test, y_train, y_test = train_test_split(plr_X, plr_result, test_size=0.3, random_state=1)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))
'''