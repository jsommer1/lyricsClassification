#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For evaluating base model performance
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

# you can run this command in a separate terminal tab in JupyterLab to 
#monitor and sanity check whether your training is actually using GPU:
# $ watch -n 1 nvidia-smi 


# choose dataset 
df = pd.read_csv('dataset_clean.csv')   

lyrics = df['Lyrics'].values
years = df['Year'].values

lyrics_train, lyrics_test, y_train, y_test = train_test_split(lyrics, years, test_size = 0.3, random_state = 1000)


vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(lyrics_train)

X_train = vectorizer.transform(lyrics_train)
X_test = vectorizer.transform(lyrics_test)

# number of classes is 6 if grouping by decade, 11 if grouping by 5 years
n_classes = 6

years_train = tf.keras.utils.to_categorical(y_train,num_classes=n_classes)
years_test = tf.keras.utils.to_categorical(y_test,num_classes=n_classes)

input_dim = X_train.shape[1]

# Set up model 

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(n_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# Load weights from saved models 
model.load_weights('my_bidirectional.h5')                   ## CHOOSE MODEL TO EVALUATE HERE

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Evaluate Model
print('\nEvaluating Training Accuracy...')
loss_train, accuracy_train = model.evaluate(x_train, years_train, verbose=False)
print('\nEvaluating Testing Accuracy...\n')
loss_test, accuracy_test = model.evaluate(x_test, years_test, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy_train)) # Prev 0.9941
print("Testing Accuracy:  {:.4f}\n".format(accuracy_test)) # prev 0.3777

# Get confusion matrix 
from sklearn.metrics import confusion_matrix

print('Making predictions...')
years_pred = model.predict(x_test)
print('Generating Confusion Matrix...')
confus_mat = confusion_matrix(np.argmax(years_test,axis=1), np.argmax(years_pred,axis=1))
print('Confusion Matrix: \n{}'.format(confus_mat))

