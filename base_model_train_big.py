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

df = pd.read_csv('dataset_clean_bow.csv')  # TODO: ADD FILE NAME HERE 

lyrics = df['Lyrics'].values
years = df['Year'].values

lyrics_train, lyrics_test, y_train, y_test = train_test_split(lyrics, years, test_size = 0.3, random_state = 1000)

from sklearn.utils import resample 

print('\n\nResampling data to even distributions...')

# recombine for downsampling
tr_len = lyrics_train.shape[0]
lyr_tr = lyrics_train.reshape((tr_len,1))
y_tr = y_train.reshape((tr_len,1))
tr_set = np.concatenate([lyr_tr,y_tr],axis=1)
# tr_set[tr_set[:,1] == 0]   <-- like this 

# separate into classes
class_0 = tr_set[tr_set[:,1]==0]
class_1 = tr_set[tr_set[:,1]==1]
class_2 = tr_set[tr_set[:,1]==2]
class_3 = tr_set[tr_set[:,1]==3]
class_4 = tr_set[tr_set[:,1]==4]
class_5 = tr_set[tr_set[:,1]==5]

n_class_5 = class_5.shape[0]

# Downsample classes 0 thru 4 to # samples in class 5
n_samp = 5000
class_0_RS = resample(class_0, replace=True, n_samples=n_samp, random_state = 27)
class_1_RS = resample(class_1, replace=True, n_samples=n_samp, random_state = 27)
class_2_RS = resample(class_2, replace=True, n_samples=n_samp, random_state = 27)
class_3_RS = resample(class_3, replace=True, n_samples=n_samp, random_state = 27)
class_4_RS = resample(class_4, replace=True, n_samples=n_samp, random_state = 27)

## Upsample class 0,1,5 to 10K samples 
#class_1_RS = resample(class_1, replace=True, n_samples=n_samp, random_state = 27)
#class_0_US = resample(class_0, replace=True, n_samples=n_samp, random_state = 27)
class_5_RS = resample(class_5, replace=True, n_samples=n_samp, random_state = 27)

# Recombine resampled datasets 
tr_set_resamp = np.concatenate([class_0_RS, class_1_RS, class_2_RS, class_3_RS, class_4_RS, class_5_RS],axis=0)

lyrics_train_resamp = tr_set_resamp[:,0]
y_train_resamp = tr_set_resamp[:,1]

print('\nData done resampling\n\n')

vectorizer = CountVectorizer()
vectorizer.fit(lyrics_train_resamp)

X_train = vectorizer.transform(lyrics_train_resamp)
X_test = vectorizer.transform(lyrics_test)

# number of classes is 6 if grouping by decade, 11 if grouping by 5 years
n_classes = 6

years_train = tf.keras.utils.to_categorical(y_train_resamp,num_classes=n_classes)
years_test = tf.keras.utils.to_categorical(y_test,num_classes=n_classes)

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

# Model type 2
model.add(Dropout(0.5))
#model.add(layers.Dense(10, 
#                        bias_regularizer=regularizers.l1(0.01),
#                        kernel_regularizer=regularizers.l2(0.01),
#                        activation='relu'))
#model.add(Dropout(0.5))
## 
##Model type 3
#model.add(layers.Dense(10, 
#                        bias_regularizer=regularizers.l1(0.01),
#                        kernel_regularizer=regularizers.l2(0.01),
#                        activation='relu'))
#model.add(Dropout(0.5))
#

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


model.save('my_NN_bigger_data_resampled.h5')

"""
MODEL TYPE: 1 Dense layer, 1 softmax layer 

Model 1: batch size 10, 75 epochs, downsample & upsample to 10K 

Model 2: batch size 10, 200 epochs, downsample to 10K 

base model w/ no resampling is in the other file base_model_train.py

Model 3: b.s. 10, 75 epochs, downsample everything to size of class 5 

MODEL TYPE: Dense -> dropout -> Dense -> Droput -> Softmax 

Model 4: Same as model 3, including downsampling 

Model 5: Model 4 but add another Dense->Dropout layer
    
model 6: batch: 10, epochs: 100, just dense-> dropout, stop words, no resamp

my_NN_bigger_data_resampled : resample everything to 5K, batch 10, epochs 100, don't remove stop words

"""

