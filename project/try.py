# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 01:44:18 2019

@author: Lung
"""

import matplotlib.pyplot as plt
import numpy as np

#data2016 = [0.52, 0.49, 0.49, 0.50]#[0.47, 0.55, 0.54, 0.54]#[0.43, 0.53, 0.51, 0.51]
#data2017 = [0.55, 0.56, 0.56, 0.61]#[0.56, 0.51, 0.52, 0.56]#[0.56, 0.47, 0.45, 0.58]
#data2018 = [0.64, 0.55, 0.55, 0.60]#[0.66, 0.61, 0.59, 0.62]#[0.69, 0.61, 0.57, 0.58]
#16, 17, 18
#team_name = [[0.43, 0.53, 0.51, 0.51], [0.56, 0.47, 0.45, 0.58], [0.69, 0.61, 0.57, 0.58]]
#player =    [[0.47, 0.55, 0.54, 0.54], [0.56, 0.51, 0.52, 0.56], [0.66, 0.61, 0.59, 0.62]] 
#avg_data =  [[0.52, 0.49, 0.49, 0.50], [0.55, 0.56, 0.56, 0.61], [0.64, 0.55, 0.55, 0.60]]
season_one = [0.62, 0.58, 0.59, 0.59]
season_all = [0.56, 0.56, 0.56, 0.56]
    
opacity = 0.4
bar_width = 0.4

fig=plt.gcf()

plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

plt.xticks(range(len(season_one)),('Logistic Regression', 'DecisionTree', 'Adaboost', 'RanderForest'), rotation=20)
bar1 = plt.bar(np.arange(len(season_one)), season_one, bar_width, align='center', alpha=opacity, color='b', label='Team name')
#bar2 = plt.bar(np.arange(len(season_all))+ bar_width/2, season_all, bar_width, align='center', alpha=opacity, color='r', label='Player name')
#bar3 = plt.bar(np.arange(len(avg_data))+ bar_width, avg_data, bar_width, align='center', alpha=opacity, color='g', label='Average Data')


plt.grid(axis='y', alpha=0.5)
plt.legend(loc =  'lower left')
fig=plt.gcf()
fig.suptitle('2018 data', fontsize=18)
fig.set_size_inches(15,10)
fig.savefig('./images/2018_season_one.jpg')
plt.show()