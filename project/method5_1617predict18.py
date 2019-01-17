# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:47:03 2019

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

def makeNormalDataAvgRecords(data, important_records):
    teams = np.unique(data['team'])
    team_data = data[data['playerid'].isin([100,200])] #team:100 -> blue; 200 -> red
    team_avg_records_df = pd.DataFrame(np.zeros((len(teams),len(important_records))), index=np.unique(data['team']), columns=important_records)
    team_records_count = pd.DataFrame(np.zeros((len(teams),1)), index=np.unique(data['team']), columns=['times'])
    for i in range(team_data.shape[0]):
        team_name = team_data.iloc[i]['team']
        team_records_count.loc[team_name] += 1
    for i in range(team_data.shape[0]):
        team_name = team_data.iloc[i]['team']
        team_avg_records_df.loc[team_name] += team_data[important_records].iloc[i]/team_records_count.loc[team_name]['times']
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

test_size = 0.2
random_state = 6
important_leagues = ['EULCS','NALCS','LCK','LPL','LMS']#,'WC','MSI']
important_records = ['teamtowerkills','earnedgpm','goldspent','gspd', #>0.7
                     'fbaron','k','a','kpm','teambaronkills', #>0.6
                     'teamdragkills','firsttothreetowers', #>0.5
                     'gdat15','goldat15','dmgtochampsperminute','totalgold','xpdat10','ft',#>0.3
                     'goldat10','monsterkills','csdat10','dmgtochamps','cspm',#>0.2                    
                     'herald','wcpm','xpat10','csat10',#>0.1
                     ]

#############################
#Loading data and Preprocess#
#############################
data2016 = pd.read_csv('2016matchdata.csv')
data2017 = pd.read_csv('2017matchdata.csv')
data2018 = pd.read_csv('2018matchdata.csv')
data = pd.concat([data2016, data2017, data2018] ,sort=False)
data_train =  pd.concat([data2016, data2017] ,sort=False)
diffset = set()
diffset = diffset.union(set(data.columns).difference(set(data2016)))
diffset = diffset.union(set(data.columns).difference(set(data2017)))
diffset = diffset.union(set(data.columns).difference(set(data2018)))
data = data.drop(diffset,axis=1)
data_train = data_train.drop(diffset,axis=1)
data2018 = data2018.drop(diffset.difference(set(data2016)), axis=1)
data['gameid'] = pd.Series(data['gameid'].astype(str))
data_train['gameid'] = pd.Series(data_train['gameid'].astype(str))

#remove world cup data
data_train= data_train[~data_train['league'].isin(['WC'])]
data= data[~data['league'].isin(['WC'])]
data_test= data2018[~data2018['league'].isin(['WC'])]

#preserve main league data
data_train= data_train[data_train['league'].isin(important_leagues)]
data= data[data['league'].isin(important_leagues)]
data_test= data2018[data2018['league'].isin(important_leagues)]

not_exist_teams = set(data_test['team']).difference(set(data_train['team']))#['Albus NoX Luna','EDward Gaming','Royal Never Give Up','I MAY']
data_test = deleteWC(data_test, not_exist_teams)

mean_train = data_train[important_records].mean()
data_train = data_train.fillna(mean_train)
data_test = data_test.fillna(mean_train)
team_normal_avg_records = makeNormalDataAvgRecords(data_train, important_records)

b_team_train_records = data_train[data_train['playerid'].isin([100])][important_records]
r_team_train_records = data_train[data_train['playerid'].isin([200])][important_records]
b_team_test_records = getTeamAvgRecords(data_test[data_test['playerid'].isin([100])], team_normal_avg_records)
r_team_test_records = getTeamAvgRecords(data_test[data_test['playerid'].isin([200])], team_normal_avg_records)

#################################
#Label Encoder & One-hot Encoder#
#################################
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
plr_b_one_hot = np.zeros((int(data_train.shape[0]/12),len(oe.categories_[0]))) 
plr_r_one_hot = plr_b_one_hot.copy()
for plrid in range(1,11):
    plr = data_train.loc[data_train['playerid'].isin([plrid])]['player']
    plr_name['player{}'.format(plrid)] = pd.Series(np.array(plr))#player name
    plr_arr = le.transform(plr)
    plr_label['player{}'.format(plrid)] = pd.Series(plr_arr) #player label
    if plrid <= 5: #blue team player
        plr_b_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
    else: #red team player
        plr_r_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
plr_X = np.concatenate((plr_b_one_hot, b_team_train_records, plr_r_one_hot, r_team_train_records), axis=1) #player one hot
#player_y
b_result = np.array(data_train.loc[data_train['playerid'].isin([100])]['result'])
plr_y = b_result

        
#################
#testing, player#
#################
#wc_player_X
wc_plr_name = pd.DataFrame()
wc_plr_label = pd.DataFrame()
wc_plr_b_one_hot = np.zeros((int(data_test.shape[0]/12),len(oe.categories_[0]))) 
wc_plr_r_one_hot = wc_plr_b_one_hot.copy()
for plrid in range(1,11):
    plr = data_test.loc[data_test['playerid'].isin([plrid])]['player']
    wc_plr_name['player{}'.format(plrid)] = pd.Series(np.array(plr))#player name
    plr_arr = le.transform(plr)
    wc_plr_label['player{}'.format(plrid)] = pd.Series(plr_arr) #player label
    if plrid <= 5: #blue team player
        wc_plr_b_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
    else:#red team player
        wc_plr_r_one_hot += oe.transform(plr_arr.reshape(-1,1)).toarray()
X_test = np.concatenate((wc_plr_b_one_hot, b_team_test_records, wc_plr_r_one_hot, r_team_test_records), axis=1) #player one hot
#player_y
wc_b_result = np.array(data_test.loc[data_test['playerid'].isin([100])]['result'])
y_test = wc_b_result

########################
#Train & Test & Scoring#
########################
X_train, X_val, y_train, y_val = train_test_split(plr_X, plr_y, test_size=test_size)

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


##############################################
print("--------------------------------------------------------")
##############################################
le = LabelEncoder()
data_team = le.fit_transform(data['team'])
oe = OneHotEncoder()
normal_data_team_oe = oe.fit_transform(data_team.reshape(-1, 1))

###############
#training data#
###############

b = np.array(data_train.loc[data_train['playerid'].isin([100])]['team'])
r = np.array(data_train.loc[data_train['playerid'].isin([200])]['team'])
b_result = np.array(data_train.loc[data_train['playerid'].isin([100])]['result'])
r_result = np.array(data_train.loc[data_train['playerid'].isin([200])]['result'])

b_one_hot = oe.transform(le.transform(b).reshape(-1,1)).toarray()
r_one_hot = oe.transform(le.transform(r).reshape(-1,1)).toarray()
team_X = np.concatenate((b_one_hot,r_one_hot), axis=1) #[blue team,red team]
team_y = b_result

##############
#testing data#
##############
wc_b = np.array(data_test.loc[data_test['playerid'].isin([100])]['team'])
wc_r = np.array(data_test.loc[data_test['playerid'].isin([200])]['team'])
wc_b_result = np.array(data_test.loc[data_test['playerid'].isin([100])]['result'])
wc_r_result = np.array(data_test.loc[data_test['playerid'].isin([200])]['result'])

wc_b_one_hot = oe.transform(le.transform(wc_b).reshape(-1,1)).toarray()
wc_r_one_hot = oe.transform(le.transform(wc_r).reshape(-1,1)).toarray()
X_test = np.concatenate((wc_b_one_hot,wc_r_one_hot), axis=1) #[blue team,red team]
y_test = wc_b_result

X_train, X_val, y_train, y_val = train_test_split(team_X, team_y, test_size=test_size, random_state=random_state)

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