# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:22:23 2018

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
#import collections
#import matplotlib.pyplot as plt
def preprocessData(data):
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('@[a-zA-X0-9_]*','',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('#[a-zA-X0-9_]*','',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('https?:[/.\w\s-]*','',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
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
    train = preprocessData(train)
    val = preprocessData(val)
    test = preprocessData(test)
    
    max_fatures = 2000
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
    batch_size = 32
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
    model.save('v2_model.h5')
    
    #test
    Y_test = model.predict(X_test)
    
    #format output
    ans = Y_test[:,1]*4
    for i in range(len(ans)):
        ans[i] = '{:3.2f}'.format(ans[i])
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