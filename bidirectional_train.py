#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 230: Deep Learning

Joe Sommer 12/8/2019 

This script is for training a multi-layered bidirectional LSTM model to 
classify a song's release year from its lyrics. 

Reads in a pre-processed dataset of Billboard's Hot 100s' song lyrics 
and corresponding release years. Saves the model at the end to be loaded into 
a separate script for evaluating the model's performance. 
"""


import numpy as np
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Dropout, Embedding

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Import data
df = pd.read_csv('dataset_billboard.csv')

lyrics = df['Lyrics'].values
years = df['Year'].values

lyrics_train, lyrics_test, y_train, y_test = train_test_split(lyrics, years, test_size = 0.3, random_state = 1000)


n_classes=6 # 6 different decades 

years_train = tf.keras.utils.to_categorical(y_train,num_classes=n_classes)
years_test = tf.keras.utils.to_categorical(y_test,num_classes=n_classes)

max_len = 1200

# Vectorizer  
vectorizer = CountVectorizer()
vectorizer.fit(lyrics_train)
X_train = vectorizer.transform(lyrics_train)
X_test = vectorizer.transform(lyrics_test)

x_train = X_train
x_test = X_test
max_len = X_train.shape[1]


# reshape inputs to feed to LSTM
N_train = x_train.shape[0]
N_test = x_test.shape[0]


x_train_reshape = np.zeros((N_train, 1,max_len))
x_test_reshape = np.zeros((N_test, 1,max_len))

years_train_reshape = np.zeros((N_train, 1,n_classes))
years_test_reshape = np.zeros((N_test, 1,n_classes))

for i in range(N_train): 
    cur_train_example = x_train[i][:].toarray()[0] 
    cur_train_label = years_train[i][:]
    x_train_reshape[i][:][:] = cur_train_example
    years_train_reshape[i][:][:] = cur_train_label
for j in range(N_test):
    cur_test_example = x_test[j][:].toarray()[0]
    cur_test_label = years_test[j][:]
    x_test_reshape[j][:][:] = cur_test_example
    years_test_reshape[j][:][:] = cur_test_label
    
    
    

input_1 = layers.Input((1,max_len))


# Adjust layers of Bidirectional model here 
bidi_1 = Bidirectional(LSTM(10, 
                             activation='tanh',
                             input_shape=(1,max_len),
                             return_sequences=True),
                        merge_mode = 'sum')(input_1)
drop_1 = Dropout(0.5)(bidi_1)
bidi_2 = Bidirectional(LSTM(10, 
                             activation='tanh',
                             return_sequences=True),
                        merge_mode = 'sum')(drop_1)
drop_2 = Dropout(0.5)(bidi_2)

bidi_3 = Bidirectional(LSTM(10, 
                             activation='tanh'),
                        merge_mode = 'sum')(drop_2)
   
out = layers.Dense(n_classes,activation='softmax')(bidi_3)


model = tf.keras.models.Model(inputs=input_1,outputs=out)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


my_batch_size = 20
n_epochs = 130


history = model.fit(x_train_reshape, years_train, 
                    epochs=n_epochs,
                    verbose=True,
                    validation_data=(x_test_reshape, years_test),
                    batch_size = my_batch_size) 



model.save('bidirectional_3_stack.h5')


