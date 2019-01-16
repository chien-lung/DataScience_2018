# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 00:03:28 2018

@author: Lung
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import re

import nltk
from nltk.corpus import stopwords

import time

def preprocessData(data): 
    stopwords_set = set(stopwords.words("english"))
    sents = []
    texts = []
    for index, row in data.iterrows():
        words_lower = [e.lower() for e in row.text.split()]
        words_cleaned = [word for word in words_lower
                         if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')]
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        text = ' '.join(words_without_stopwords)
        re.sub('[^a-zA-z0-9\s]','',text)
        try:
            sent = row.sentiment
        except: #for test file
            sent = 2
        texts.append(text)
        sents.append(sent)
        
    data = pd.DataFrame({'sentiment':sents, 'text':texts})
    return data
    
if __name__=="__main__":
    trainfile="train.csv"
    testfile="test.csv"
    train = pd.read_csv(trainfile)
    train = train[['sentiment','text']]
    test = pd.read_csv(testfile)
    test = test[['text']]
    train, val = train_test_split(train, test_size=0.1, random_state=0)
    
    #preprocess data
    start_time = time.time()
    train = preprocessData(train)
    val = preprocessData(val)
    test = preprocessData(test)
    print('preprocess time: ', time.time()-start_time, 'sec.')
    
    max_fatures = 2500
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(train['text'].values)
    X_train = tokenizer.texts_to_sequences(train['text'].values)
    X_train = pad_sequences(X_train)
    
    X_val = tokenizer.texts_to_sequences(val['text'].values)
    X_val = pad_sequences(X_val, maxlen=X_train.shape[1])
    
    X_test = tokenizer.texts_to_sequences(test['text'].values)
    X_test = pad_sequences(X_test, maxlen=X_train.shape[1])
    
    #hyperparameter
    epoch = 10
    batch_size = 16
    embed_dim = 128
    lstm_out = 196
    
    #model
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length = X_train.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    
    #load model
    #model = load_model('v2_model_2.h5')
    
    #train
    Y_train = pd.get_dummies(train['sentiment']).values
    Y_val = pd.get_dummies(val['sentiment']).values
    model.fit(X_train, Y_train, epochs = epoch, batch_size=batch_size, validation_data=(X_val,Y_val))
    
    #save model
    model.save('v3_model.h5')
    
    #test
    Y_test = model.predict(X_test)
    
    #format output
    ans = Y_test[:,1]*4
    for i in range(len(ans)):
        ans[i] = '{:.2f}'.format(ans[i])
    index = np.array(range(len(ans)))
    ans_dict = {'ID':index,'sentiment':ans}
    df = pd.DataFrame(ans_dict)
    df.to_csv('answer.csv', index=None)
    
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