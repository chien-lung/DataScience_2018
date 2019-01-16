# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:35:09 2019

@author: Lung
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.model_selection import train_test_split

#filename = '2018matchdata.csv' #'2017matchdata.csv' #'2016matchdata.csv'
#data = pd.read_csv(filename)

data2016 = pd.read_csv('2016matchdata.csv')
data2017 = pd.read_csv('2017matchdata.csv')
data2018 = pd.read_csv('2018matchdata.csv')
data = pd.concat([data2016, data2017, data2018] ,sort=False) 
diffset = set()
diffset = diffset.union(set(data.columns).difference(set(data2016)))
diffset = diffset.union(set(data.columns).difference(set(data2017)))
diffset = diffset.union(set(data.columns).difference(set(data2018)))
data = data.drop(diffset,axis=1)

data['ban1']=pd.Series(np.array(data['ban1'].fillna('NaN')), index=data.index)
data['ban2']=pd.Series(np.array(data['ban2'].fillna('NaN')), index=data.index)
data['ban3']=pd.Series(np.array(data['ban3'].fillna('NaN')), index=data.index)
data['herald']=pd.Series(np.array(data['herald'].fillna(0)), index=data.index)

'''
focus = ['result',
         ##'side','ban1','ban2','ban3',
         ###########for fig1#############
         'k','d','a',
         'kpm','okpm','ckpm',
         'teamdragkills','oppdragkills',
         'ft','firsttothreetowers','teamtowerkills','opptowerkills',
         'fbaron','teambaronkills','oppbaronkills',
         ###########for fig2#############
         #'dmgshare','dmgtochamps','dmgtochampsperminute',
         #'wards','wpm','wardkills','wcpm',
         #'totalgold','earnedgpm','goldspent','gspd',
         #'minionkills','monsterkills','monsterkillsownjungle','monsterkillsenemyjungle',
         ###########for fig3#############
         #'herald',
         #'visiblewardclearrate','visionwardbuys','visionwards',
         #'cspm','csat10','oppcsat10','csdat10',
         #'goldat10','oppgoldat10','gdat10','goldat15','oppgoldat15','gdat15',
         #'xpdat10','xpat10','oppxpat10'
         ]
'''
focus = ['teamtowerkills','earnedgpm','goldspent','gspd', #>0.7
                     'fbaron','k','a','kpm','teambaronkills', #>0.6
                     'teamdragkills','firsttothreetowers', #>0.5
                     'gdat15','goldat15','dmgtochampsperminute','totalgold','xpdat10','ft',#>0.3
                     #'goldat10','monsterkills','csdat10','dmgtochamps','cspm',#>0.2                    
                     #'herald','wcpm','xpat10','csat10',#>0.1
                     ]

focus_data = data.loc[data['playerid'].isin([100,200])][focus]

df_corr = focus_data._get_numeric_data()

mask = np.zeros_like(df_corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize = (15,10))
sns.heatmap(df_corr.corr(), cmap = cmap, annot = True, fmt = '.2f', mask = mask, square=True, linewidths=.5, center = 0)
plt.title('Correlations - win vs factors (all games)')
#plt.savefig('{}focus_1.jpg'.format(filename[0:4]))
plt.savefig('all_focus_1.jpg')
