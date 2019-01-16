# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 00:44:31 2018

Image Classification for fashion item
train.csv: 60000, test.csv: 10000 
attributes: label, pixel1~784
data: 0(white) ~ 255(black)

using vgg16 pretrain model

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
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
from PIL import Image

def test_and_save(X_test, save_file):
    dim = 48
    test_imgs = np.zeros(shape=(X_test.shape[0], dim, dim,3))
    for i in range(X_test.shape[0]):
        img_arr = cv.resize(X_test[i], dsize=(dim,dim), interpolation=cv.INTER_CUBIC)
        img_arr = np.repeat(img_arr.reshape(dim,dim,1), 3, axis=2)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        test_imgs[i] = img_arr
    Y_test = model.predict(test_imgs, batch_size=16)
    ans = np.zeros(shape = (Y_test.shape[0], 1))
    ans = np.argmax(Y_test, axis=1)
        
    df = pd.DataFrame(np.transpose(np.vstack((np.array(range(len(ans))), ans.astype('int32')))))
    df.to_csv(save_file, header=['id','label'], index=None)

if __name__=="__main__":
    trainfile="train.csv"
    testfile="test.csv"
    train = pd.read_csv(trainfile)
    test = pd.read_csv(testfile)
    
    label = np.array(train.iloc[:,0:1])
    train_pixels = np.array(train.iloc[:,1:])
    dim = 48
    
    label = to_categorical(label)    
    train_pixels = train_pixels.reshape(train_pixels.shape[0], 28, 28)
    train_pixels = train_pixels.astype('float32')
    #train_pixels /= 255
    X_test = np.array(test.iloc[:,1:]).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28)
    #X_test /= 255
    
    train_imgs = np.zeros(shape=(train_pixels.shape[0], dim, dim,3))
    
    for i in range(train_pixels.shape[0]):
        img_arr = cv.resize(train_pixels[i], dsize=(dim,dim), interpolation=cv.INTER_CUBIC)
        img_arr = np.repeat(img_arr.reshape(dim,dim,1), 3, axis=2)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        train_imgs[i] = img_arr
    
    #split training and validation data
    X_train, X_val, Y_train, Y_val = train_test_split(train_imgs, label, test_size=0.15 , random_state=0)
    
    #training model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(dim, dim, 3))
    
    out = vgg_model.output
    out = Flatten()(out)
    out = Dense(256, activation='relu', input_dim=1 * 1 * 512)(out)
    out = Dropout(0.5)(out)
    out = Dense(10, activation='softmax')(out)
    
    model = Model(inputs=vgg_model.input, outputs=out)
    
    #laoding model parameters
    #model.load_weights('model_weights_pre.h5')
    
    #training setting
    for layer in vgg_model.layers:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=60, validation_data=(X_val, Y_val))
    
    #save weight
    model.save_weights('model_weights_pre.h5')
    #model.save('model_pre.h5')
    
    #test
    test_and_save(X_test, 'answer_pre.csv')
    
    #fine-tune
    X_train, X_val, Y_train, Y_val = train_test_split(train_imgs, label, test_size=0.15 , random_state=1)
    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=50, validation_data=(X_val, Y_val))
    
    #save fine-tune weight
    model.save_weights('model_weights_pre_fine.h5')
    #model.save('model_pre_fine.h5')

    
    #test
    test_and_save(X_test, 'answer_pre_fine.csv')
    
    '''
    new_model = Sequential()
    new_model.add(Flatten())
    new_model.add(Dense(256, activation='relu', input_dim=1 * 1 * 512))
    new_model.add(Dropout(0.5))
    new_model.add(Dense(10, activation='softmax'))5 
    '''
    
    '''
    x = train_imgs[0]
    x = cv.resize(x, dsize=(224,224), interpolation=cv.INTER_CUBIC)
    x = np.stack((x,)*3, axis=-1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    '''

    '''
    #for include_top = True
    vgg_model.layers.pop()
    vgg_model.outputs = [vgg_model.layers[-1].output]
    vgg_model.layers[-1].outbound_nodes = []
    new_layer = Dense(10, activation='softmax')(vgg_model.output)
    model = Model(vgg_model.input, new_layer)
    for layer in vgg_model.layers:
        layer.trainable = False
    model.compile(optimizer=optimizers.RMSprop(lr=2e-4), loss='categorical_crossentropy')
    
    model.fit(X_train, Y_train, epochs=1)
    '''
    