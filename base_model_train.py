#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 230: Deep Learning

Joe Sommer 12/8/2019 

This script is for training a simple, single-layer model to 
classify a song's release year from its lyrics. 

Reads in a pre-processed dataset of song lyrics 
and corresponding release years from either Billboard's Hot 100s or the
Million Song Dataset. Saves the model at the end to be loaded into 
a separate script for evaluating the model's performance. 
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

# Read in preprocessed dataset 
df = pd.read_csv('dataset_billboard.csv')   

lyrics = df['Lyrics'].values
years = df['Year'].values

lyrics_train, lyrics_test, y_train, y_test = train_test_split(lyrics, years, test_size = 0.3, random_state = 1000)


vectorizer = CountVectorizer()
vectorizer.fit(lyrics_train)

X_train = vectorizer.transform(lyrics_train)
X_test = vectorizer.transform(lyrics_test)

n_classes = 6

years_train = tf.keras.utils.to_categorical(y_train,num_classes=n_classes)
years_test = tf.keras.utils.to_categorical(y_test,num_classes=n_classes)

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(n_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


my_batch_size = 10
n_epochs = 75

history = model.fit(X_train, years_train, 
                    epochs=n_epochs,
                    verbose=True,
                    validation_data=(X_test, years_test),
                    batch_size=my_batch_size)

# Change name depending on which dataset is being trained on 
model.save('base_model_small_set.h5')

