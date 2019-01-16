# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:02:43 2019

@author: Lung
"""
import matplotlib.pyplot as plt
import numpy as np

#data2016 = [0.52, 0.49, 0.49, 0.50]#[0.47, 0.55, 0.54, 0.54]#[0.43, 0.53, 0.51, 0.51]
#data2017 = [0.55, 0.56, 0.56, 0.61]#[0.56, 0.51, 0.52, 0.56]#[0.56, 0.47, 0.45, 0.58]
#data2018 = [0.64, 0.55, 0.55, 0.60]#[0.66, 0.61, 0.59, 0.62]#[0.69, 0.61, 0.57, 0.58]
#16, 17, 18
'''
team_name = [[0.43, 0.53, 0.51, 0.51], [0.56, 0.47, 0.45, 0.58], [0.69, 0.61, 0.57, 0.58]]
player =    [[0.47, 0.55, 0.54, 0.54], [0.56, 0.51, 0.52, 0.56], [0.66, 0.61, 0.59, 0.62]] 
avg_data =  [[0.52, 0.49, 0.49, 0.50], [0.55, 0.56, 0.56, 0.61], [0.64, 0.55, 0.55, 0.60]]
player_avg_data=[[0.53, 0.63, 0.47, 0.51], [0.57, 0.49, 0.53, 0.57], [0.63, 0.60, 0.68, 0.58]]
player_avg_data_std = [[0.40, 0.46, 0.42, 0.53], [0.57, 0.60, 0.55, 0.56], [0.68, 0.48, 0.51, 0.58]]
'''

for i in range(3):
    
    opacity = 0.4
    bar_width = 0.22
    
    fig=plt.gcf()
    
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    
    plt.xticks(range(len(team_name[i])),('Logistic Regression', 'DecisionTree', 'Adaboost', 'RanderForest'), rotation=20)
    bar1 = plt.bar(np.arange(len(avg_data[i]))- bar_width/2*3, avg_data[i], bar_width, align='center', alpha=opacity, color='g', label='Average Data')
    bar2 = plt.bar(np.arange(len(player[i]))- bar_width/2, player[i], bar_width, align='center', alpha=opacity, color='r', label='Player name')
    bar3 = plt.bar(np.arange(len(team_name[i]))+ bar_width/2, team_name[i], bar_width, align='center', alpha=opacity, color='b', label='Team name')
    bar4 = plt.bar(np.arange(len(player_avg_data[i]))+ bar_width/2*3, player_avg_data[i], bar_width, align='center', alpha=opacity, color='black', label='Average Data & Player name')
    
    plt.grid(axis='y', alpha=0.5)
    plt.legend(loc =  'lower left')
    fig=plt.gcf()
    fig.suptitle('{} data'.format(2016+i), fontsize=18)
    fig.set_size_inches(15,10)
    fig.savefig('./images/{}_all_comp.jpg'.format(2016+i))
    plt.show()