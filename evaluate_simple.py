#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 230: Deep Learning

Joe Sommer 12/8/2019 

This script is for evaluating a pre-trained single-layer 
model that classifies a song's release year from its lyrics. 

It sets up an untrained model, reads in the pre-trained model's weights, then 
evaluates its performance. It also displays a confusion matrix showing the 
results on a test set. 
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


# Load weights from saved models 
model.load_weights('base_model_big_set.h5')            ## CHOOSE MODEL TO EVALUATE HERE

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Evaluate Model
print('\nEvaluating Training Accuracy...')
loss_train, accuracy_train = model.evaluate(X_train, years_train, verbose=False)
print('\nEvaluating Testing Accuracy...\n')
loss_test, accuracy_test = model.evaluate(X_test, years_test, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy_train)) 
print("Testing Accuracy:  {:.4f}\n".format(accuracy_test)) 

# Get confusion matrix 
from sklearn.metrics import confusion_matrix

print('Making predictions...')
years_pred = model.predict(X_test)
print('Generating Confusion Matrix...')
print('NOTE: rows: actual class, cols: predicted class')
confus_mat = confusion_matrix(np.argmax(years_test,axis=1), np.argmax(years_pred,axis=1))
print('Confusion Matrix: \n{}'.format(confus_mat))

