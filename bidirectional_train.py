#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:44:02 2019

@author: Joe1
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV

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
# print(os.listdir("../CS230_Project"))

# when training on AWS p2.xlarge, this command will ensure that you're training with the GPU: 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# you can run this command in a separate terminal tab in JupyterLab to monitor and sanity check whether your training is actually using GPU:
# $ watch -n 1 nvidia-smi 


# Import data
df = pd.read_csv('dataset_clean.csv')

lyrics = df['Lyrics'].values
years = df['Year'].values

lyrics_train, lyrics_test, y_train, y_test = train_test_split(lyrics, years, test_size = 0.3, random_state = 1000)



n_classes=6

years_train = tf.keras.utils.to_categorical(y_train,num_classes=n_classes)
years_test = tf.keras.utils.to_categorical(y_test,num_classes=n_classes)

max_features = 10000
max_len = 1200

# Vectorizer  
vectorizer = CountVectorizer()
vectorizer.fit(lyrics_train)
X_train = vectorizer.transform(lyrics_train)
X_test = vectorizer.transform(lyrics_test)

x_train = X_train
x_test = X_test
max_len = X_train.shape[1]

# tokenizer 
#tok = text.Tokenizer(num_words=max_features, lower=False)
#tok.fit_on_texts(list(lyrics_train)+list(lyrics_test))
#X_train = tok.texts_to_sequences(lyrics_train)
#X_test = tok.texts_to_sequences(lyrics_test)


# Pad data 
#x_train = sequence.pad_sequences(X_train,maxlen= max_len)
#x_test = sequence.pad_sequences(X_test, maxlen = max_len)


# reshape inputs to feed to LSTM
N_train = x_train.shape[0]
N_test = x_test.shape[0]


## TODO: DELETE
#m = 0
#for i in range(N_train):
#    l = len(X_train[i])
#    if l > m:
#        m = l
#for i in range(N_test):
#    l = len(X_test[i])
#    if l > m:
#        m = l
#print('max len : {}'.format(m))



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
    

# model.add(layers.Dense(10, input_dim=input_dim, activation='relu')) just for reference 

#model = Sequential()
#model.add(Embedding(max_features, 128, input_length=max_len))
#model.add(Bidirectional(LSTM(10)))

#model.add(Bidirectional(LSTM(10, 
#                             activation='tanh',
#                             input_shape=(1,max_len))))
#model.add(Dropout(0.5))
#
#model.add(layers.Dense(10, 
#                        activation='tanh'))
#model.add(Dropout(0.5))

#model = Sequential()
#model.add(Bidirectional(LSTM(10, 
#                             activation='tanh',
#                             input_shape=(1,max_len),
#                             return_sequences=True)))
#model.add(Dropout(0.5))
#
#model.add(Bidirectional(LSTM(10, 
#                             activation='tanh',
#                             return_sequences=True)))
#model.add(Dropout(0.5))
#
#model.add(Bidirectional(LSTM(10, 
#                             activation='tanh')))
#model.add(Dropout(0.5))
#
##model.add(Bidirectional(LSTM(10, 
##                             activation='tanh')))
#
##
#model.add(layers.Dense(10, 
#                        activation='tanh'))
#model.add(layers.Dense(n_classes,activation='softmax'))
    
    
# Original Bidi-LSTM 
#model = Sequential()
#model.add(Bidirectional(LSTM(10, 
#                             activation='tanh',
#                             input_shape=(1,max_len),
#                             return_sequences=True)))
#model.add(Dropout(0.5))
#
#model.add(Bidirectional(LSTM(10, 
#                             activation='tanh',
#                             return_sequences=True)))
#model.add(Dropout(0.5))
#
#model.add(Bidirectional(LSTM(10, 
#                             activation='tanh')))
#model.add(Dropout(0.5))
#
#model.add(layers.Dense(10, 
#                        activation='tanh'))


# Attention Model

input_1 = layers.Input((1,max_len))
#LSTM 
bidi_1 = Bidirectional(LSTM(10, 
                             activation='tanh',
                             input_shape=(1,max_len),
                             return_sequences=True))(input_1)
drop_1 = Dropout(0.5)(bidi_1)
#bidi_2 = Bidirectional(LSTM(10, 
#                             activation='tanh',
#                             return_sequences=True))(drop_1)
#drop_2 = Dropout(0.5)(bidi_2)
#bidi_3 = Bidirectional(LSTM(10, 
#                             activation='tanh'))(drop_2)
#drop_3 = Dropout(0.5)(bidi_3)

drop_3 = drop_1
dense_1 = layers.Dense(10, 
                        activation='tanh')(drop_3)


att = layers.Dense(10,input_dim=10)(dense_1)
att = layers.Activation('softmax')(att)
att = layers.RepeatVector(10)(att)
att = layers.Permute((2,1))(att)

dot_out = layers.Dot(axes=1)([dense_1, att])

out = layers.Dense(n_classes,activation='softmax')(dot_out)

model = tf.keras.models.Model(inputs=input_1,outputs=out)

# Delete from up here if it doesn't work



#model.add(layers.Dense(n_classes,activation='softmax'))





model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#model.summary()

my_batch_size = 10
n_epochs = 120

#history = model.fit(x_train, years_train, 
#                    epochs=n_epochs,
#                    verbose=True,
#                    validation_data=(x_test, years_test),
#                    batch_size=my_batch_size) 
history = model.fit(x_train_reshape, years_train, 
                    epochs=n_epochs,
                    verbose=True,
                    validation_data=(x_test_reshape, years_test),
                    batch_size = my_batch_size) 



model.save('my_bidirectional_6.h5')

"""
my_bd : batch size 30, epochs 20, 
max_features = 10000
max_len = 5800 

my_bidi_fixed_2 : batdch size 30, epochs 17
max_features 10K, max-len 5800 
add dropout & dense layers 

my_bidi_3 : batch size 20, epochs 100
removed embedding, just the LSTM, activation = 'tanh' (relu before)
also using vectorizer instead of tokenizer 
adding dropout -> tr/test acc about 97-98/39-40 
add dense layer and dropout -> test acc about 40-41 

my_bidi_4: epochs 100, batch size 20, add another LSTM layer to model 3 
instead of the dense layer -> about 93/41 
add another LSTM and 2 dropouts, epochs down to 30 -> about 86/41, but increasing
epochs back up to 120 -> back down to like 37 for some reason

bidi_5 : bidi_4 but batch size = 10, also add dense layer -> about 

bidi_6: bidi_5 but attempt to add attention -> nah it sucked
attempt 2: only has lstm 

"""

