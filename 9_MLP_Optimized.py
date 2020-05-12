#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Team 18

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import time
import os
import psutil

warnings.filterwarnings("ignore")

#Reading the file
pd.options.display.float_format = "{:.6f}".format
df = pd.read_csv('Dataset_Without_Outliers.csv')
z = np.abs(stats.zscore(df.drop(columns=['Class'])))
print(z)
threshold = 3
print(np.where(z > 3))
print(z[0][3])
df = df[(z < 3).all(axis=1)]
df.shape
df.head(5)
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=.30, random_state=50)

###### Standard Scaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)


####Label Encoding
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)
encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_Y_test = encoder.transform(Y_test)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y_test = np_utils.to_categorical(encoded_Y_test)

####Base Model
start = time.time()
model_base = Sequential()
model_base.add(Dense(18, input_dim=19, activation='relu'))
model_base.add(Dense(8, activation='relu'))
model_base.add(Dense(4, activation='softmax'))
model_base.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model_base.fit(X_train, dummy_y, epochs=100, validation_split=0.20, batch_size=10)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'cv'], loc='upper left')
plt.show()

y_pred = model_base.predict(X_test)

y_test_class = np.argmax(dummy_y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
#Accuracy of the predicted values
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))
print(accuracy_score(y_test_class,y_pred_class))

end = time.time()
print('Time complexity for MLP base model is', end-start,' seconds')
process = psutil.Process(os.getpid())
print('Memory consumed for MLP base model is',process.memory_info().rss, 'bytes')

#####Optimized Model

start = time.time()
model = Sequential()
model.add(Dense(18, input_dim=19, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, dummy_y, epochs=150, validation_split=0.20, batch_size=10)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'cv'], loc='upper left')
plt.show()

y_pred = model.predict(X_test)

y_test_class = np.argmax(dummy_y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
#Accuracy of the predicted values
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))
print(accuracy_score(y_test_class,y_pred_class))

end = time.time()
print('Time complexity for MLP optimized model is', end-start,' seconds')
process = psutil.Process(os.getpid())
print('Memory consumed for MLP optimized model is',process.memory_info().rss, 'bytes')

