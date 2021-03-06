# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:22:23 2018

@author: Lung
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
#import collections

#nltk.download('punkt')

def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

if __name__=="__main__":
    trainfile="train.csv"
    testfile="test.csv"
    train = pd.read_csv(trainfile)
    train = train[['sentiment','text']]
    train, test = train_test_split(train, test_size=0.1)
    
    train_p = train[ train['sentiment']==4 ]
    train_p = train_p['text']
    train_n = train[ train['sentiment']==0 ]
    train_n = train_n['text']
    
    print("Positive words")
    wordcloud_draw(train_p,'white')
    print("Negative words")
    wordcloud_draw(train_n)
    
    tweets = []
    stopwords_set = set(stopwords.words("english"))

    for index, row in train.iterrows():
        words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
                         if 'http' not in word
                         and not word.startswith('@')
                         and not word.startswith('#')
                         and word != 'RT']
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        tweets.append((words_without_stopwords, row.sentiment))
        
        
    '''
    maxlen = 0
    wordfreq = collections.Counter()
    num_recs = 0
    for sentence in X:
        words = nltk.word_tokenize(sentence.lower())
        if len(sentence)>maxlen:
            maxlen = len(sentence)
        for word in words:
            wordfreq[word] += 1
        num_recs += 1
    print('max_len: ',maxlen)
    print('freq: ',wordfreq)
    
    maxlen = 374
    word_freq: 690522
    '''
    