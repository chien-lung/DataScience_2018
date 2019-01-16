
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import re
import time

def preprocessData(data): 
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('@[a-zA-X0-9_]*','',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('#[a-zA-X0-9_]*','',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('https?:[/.\w\s-]*','',x)))
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    return data

def pretrainEmbedding(file):
    embeddings_index = dict()
    with open(file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

if __name__=="__main__":
    trainfile="train.csv"
    testfile="test.csv"
    train = pd.read_csv(trainfile)
    train = train[['sentiment','text']]
    test = pd.read_csv(testfile)
    test = test[['text']]
    train, val = train_test_split(train, test_size=0.1, random_state=0)
    
    #parameters
    max_words = 5000
    max_len = 35
    
    #hyperparameter
    epoch = 5
    batch_size = 16
    embed_dim = 100
    lstm_out = 196
    
    #preprocess data
    print('preprocess data...')
    start_time = time.time()
    train = preprocessData(train)
    val = preprocessData(val)
    test = preprocessData(test)
    print('preprocess time: ', time.time()-start_time, 'sec.')
    
    #load pre-train word embedding
    print('loading pre-train word embedding')
    start_time = time.time()
    embeddings_index = pretrainEmbedding('glove.twitter.27B.100d.txt')
    print('loading time: ', time.time()-start_time, 'sec.')
    print('loaded %s word vectors.' % len(embeddings_index))
    
    #tokenize script
    print('tokenize script...')
    tokenizer = Tokenizer(num_words=max_words, split=' ')
    tokenizer.fit_on_texts(train['text'].values)
    
    #embedding layer weight
    print('build pre-trained embedding layer weight...')
    num_words = min(max_words, len(tokenizer.word_index))
    embedding_matrix = np.zeros((num_words, embed_dim))
    for word, i in tokenizer.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    #input
    X_train = tokenizer.texts_to_sequences(train['text'].values)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = tokenizer.texts_to_sequences(val['text'].values)
    X_val = pad_sequences(X_val, maxlen=max_len)
    X_test = tokenizer.texts_to_sequences(test['text'].values)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    #model
    model = Sequential()
    model.add(Embedding(max_words, embed_dim, weights=[embedding_matrix],
                        input_length = max_len, trainable=False))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
    print(model.summary())
    
    #load model
    #model = load_model('v4_model.h5')
    
    #train
    #Y_train = pd.get_dummies(train['sentiment']).values
    #Y_val = pd.get_dummies(val['sentiment']).values
    Y_train = train['sentiment'].values /4
    Y_val = val['sentiment'].values /4
    model.fit(X_train, Y_train, epochs = epoch, batch_size=batch_size, validation_data=(X_val,Y_val))
    
    #save model
    model.save('v5_model.h5')
    
    #test
    Y_test = model.predict(X_test)
    
    #format output
    #ans = Y_test[:,1]*4
    ans = Y_test*4
    for i in range(len(ans)):
        ans[i] = '{:.2f}'.format(ans[i])
    index = np.array(range(len(ans)))
    ans_dict = {'ID':index,'sentiment':ans}
    df = pd.DataFrame(ans_dict)
    df.to_csv('answer.csv', index=None)