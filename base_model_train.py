#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains the base model (just some Dense layers). 
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

# you can run this command in a separate terminal tab in JupyterLab to monitor and sanity check whether your training is actually using GPU:
# $ watch -n 1 nvidia-smi 

df = pd.read_csv('dataset_clean_bow.csv')   

lyrics = df['Lyrics'].values
years = df['Year'].values

lyrics_train, lyrics_test, y_train, y_test = train_test_split(lyrics, years, test_size = 0.3, random_state = 1000)


vectorizer = CountVectorizer()
vectorizer.fit(lyrics_train)

X_train = vectorizer.transform(lyrics_train)
X_test = vectorizer.transform(lyrics_test)

# number of classes is 6 if grouping by decade, 11 if grouping by 5 years
n_classes = 6

years_train = tf.keras.utils.to_categorical(y_train,num_classes=n_classes)
years_test = tf.keras.utils.to_categorical(y_test,num_classes=n_classes)

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.8)) ## 


model.add(layers.Dense(n_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

my_batch_size = 10
n_epochs = 100

history = model.fit(X_train, years_train, 
                    epochs=n_epochs,
                    verbose=True,
                    validation_data=(X_test, years_test),
                    batch_size=my_batch_size)

model.save('my_NN_no_resamp_3.h5')

"""
Model 1: batch size 10, 75 epochs  (run on BOW dataset) 
    -> really skewed distribution! prob will be "accurate", but confusion matrix will prob show lots of classifications as 4 

Model 2: batch size 10, 20 epochs (stop before overfitting), run on small dataset

Model 3: batch size 10, 100 epochs, add dropout only, run on big set

MOdel 4: model 3 but with dropout of 0.8 instead 

"""

