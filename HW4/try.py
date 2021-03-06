# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 15:54:54 2018

@author: Lung
"""
import pandas as pd
import numpy as np
import cv2 as cv
from keras.applications.vgg16 import VGG16
from keras.applications import vgg16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.utils import to_categorical
from keras import optimizers
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from PIL import Image

def createModel(input_shape, nClass):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nClass, activation='softmax'))
     
    return model

if __name__=="__main__":
    trainfile="train.csv"
    testfile="test.csv"
    train = pd.read_csv(trainfile)
    test = pd.read_csv(testfile)
    #ans = pd.read_csv('sub.csv',header=None)
    
    label = np.array(train.iloc[:,0:1])
    train_pixels = np.array(train.iloc[:,1:])
    X_test = np.array(test.iloc[:,1:]).astype('float32')
    X_test /= 255
    
    label = to_categorical(label)
    
    train_pixels = train_pixels.reshape(train_pixels.shape[0], 28, 28, 1)
    train_pixels = train_pixels.astype('float32')
    train_pixels /= 255
    
    X_train, X_val, Y_train, Y_val = train_test_split(train_pixels, label, test_size=0.15, random_state=0)
    
    model = createModel((28,28,1), 10)
    
    #model.load_weights('model_weights.h5')
    
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=50, validation_data=(X_val, Y_val))
    
    #save
    model.save_weights('model_weights.h5')
    model.save('model.h5')

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    Y_test = model.predict(X_test, batch_size=20)
    ans = np.zeros(shape = (Y_test.shape[0], 1))
    for i in range(X_test.shape[0]):
        ans[i] = np.argmax(Y_test[i])
        
    df = pd.DataFrame(np.transpose(np.vstack((np.array(range(len(ans))), ans[:,0].astype('int32')))))
    df.to_csv('answer.csv', header=['id','label'], index=None)