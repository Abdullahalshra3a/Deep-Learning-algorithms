from collections import Counter
from keras import backend as K
from keras.layers import *
import numpy as np
from keras.utils import np_utils, to_categorical

import argparse
import os
from os import walk
from sklearn import preprocessing
from sklearn.metrics import log_loss, auc, roc_curve

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from numpy import array
import pickle
import re
import glob
import datetime
import tensorflow as tf
import itertools
import math
import random
from collections import Counter
import keras_metrics
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.preprocessing import Normalizer

from numpy.random import seed

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping



from numpy.random import seed



from keras.utils.data_utils import get_file
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM,SimpleRNN

SEED = 123 #used to help randomly select the data points








from numpy.random import seed
seed(7)









SEED = 123 #used to help randomly select the data points


early_stopping_monitor = EarlyStopping(patience=2)


## Import dependencies

## Set random seeds for reproducibility
np.random.seed(123)
random.seed(123)

import time


#using the specfied columns
COLUMN_TO_STANDARDIZE = [
                'Protocol',
        		'Flow Duration',
     
                'Flow Byts/s',	
                'Flow Pkts/s',	
     
                'Fwd Pkts/s',	
                'Bwd Pkts/s',	
                
                'Label'



]









# uncomment if you want to merge all the data together, so you can read it from one excel file
# wd = "./"
# cicids_files = "*.csv"

# print("Reading inSDN data...")
# # files = glob.glob('C:/Users/bbb/PycharmProjects/RNN/dataset/*.csv')
# files = glob.glob('/home/ahmad/Desktop/inSDN/InSDN_DatasetCSV/*.csv')

# print(files)
# cicids_data = []
# for ff in files:
#     cicids_data.append(pd.read_csv(ff, encoding="Latin1", usecols=COLUMN_TO_STANDARDIZE))
# cicids_data = pd.concat(cicids_data)


# # cicids_data = cicids_data.rename(columns={' Label': 'Label'})

# dataframe=cicids_data.copy()


# print(dataframe)
# print('sucess')

# dataframe.to_csv('data.csv')
# #

#reading data from data.csv, usecols for reading only specified features
dataframe = pd.read_csv("data.csv",usecols=COLUMN_TO_STANDARDIZE)


#choose which classes you want in your dataset, what it does exactly is filtering and taking only rows with specified label names
dataframe = dataframe.loc[(dataframe.Label == "DDoS") | (dataframe.Label == "Normal")]


#print availbe classes after filtering
print(dataframe['Label'].count())

#to calculate how many DoS, Normal, DDoS, probe, web attack 
# print("DoS labels count is ", dataframe[dataframe['Label']=='DDoS'].count())

labeltrain=dataframe['Label']

#change normal class to 0 and DDoS to 1
newlabel_train=labeltrain.replace({'Normal':0,'DDoS':1 })

#put the values back in the values of the dataframe labels
dataframe['Label']=newlabel_train



#here I removed labels from the columns to standardize because labels are either 0 or 1 and they don't need standrdizing
COLUMN_TO_STANDARDIZE = [
                'Protocol',
        		'Flow Duration',
               
                'Flow Byts/s',	
                'Flow Pkts/s',	
          
                'Fwd Pkts/s',	
                'Bwd Pkts/s',	
                

]


#they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.
dataframe[COLUMN_TO_STANDARDIZE] = preprocessing.StandardScaler().fit_transform(dataframe[COLUMN_TO_STANDARDIZE])











#Dividing attack and normal traffic
dataframeAttack = dataframe.loc[(dataframe.Label == 1)]
dataframeNormal = dataframe.loc[(dataframe.Label == 0)]


#Dividing the attack traffic 80% for training and 20% for testing
training_df_attack, testing_df_attack = train_test_split(dataframeAttack , test_size=0.2)

#Diving the normal traffic 80% for training and 20% for testing
training_df_normal, testing_df_normal = train_test_split(dataframeNormal , test_size=0.2)


#appending the train traffics of normal and attacks together
dataframe_train = training_df_attack.append(training_df_normal)

#appending the test traffics of normal and attacks together
dataframe_test = testing_df_attack.append(testing_df_normal)


#using shuffle for training data, it is recommended to avoid having the normal traffic or attack traffic in a sequence
from sklearn.utils import shuffle
dataframe_train = shuffle(dataframe_train)
dataframe_train = shuffle(dataframe_train)
dataframe_train = shuffle(dataframe_train)
dataframe_train = shuffle(dataframe_train)
dataframe_train = shuffle(dataframe_train)


dataframe_train = dataframe_train.reset_index()
del dataframe_train['index']



# uncomment to check the number of DDoS samples in training and testing dataframes
print("DDoS labels count ON THE DATAFRAME_TRAIN is ", dataframe_train[dataframe_train['Label']==1].count())
print("DDoS labels count ON THE DATAFRAME_TEST is ", dataframe_test[dataframe_test['Label']==1].count())



#extracting the features and labels from training dataframe
x_train,y_train=dataframe_train,dataframe_train.pop('Label').values

#added new
y_train = pd.get_dummies(y_train)

x_train=x_train.values

#changing datatype from float64 to float32
x_train = np.array(x_train).astype(np.float32)

y_train = np.array(y_train).astype(int)


#extracting the features and labels from testing dataframe
x_test,y_test=dataframe_test,dataframe_test.pop('Label').values


x_test=x_test.values

x_test = np.array(x_test).astype(np.float32)

y_test = pd.get_dummies(y_test)
print('y_test after getting the dummies are',y_test)
y_test=y_test.values
y_test = np.array(y_test).astype(int)

#number of selected features
n_features = 6
timesteps = 1

#changeing the dimension of the training and testing data to 3 dimension to fit as an input to the neural network
x_train_3d = x_train.reshape(x_train.shape[0], timesteps, n_features)
x_test_3d = x_test.reshape(x_test.shape[0], timesteps, n_features)




# initializing the model
model = Sequential()


# replace GRU with SimpleRNN if you want to use RNN

#adding input layer and GRU layer in one step
model.add(GRU(6, input_shape=(x_train_3d.shape[1:]), activation='tanh', return_sequences=True))
#adding dropout to avoid overfit
model.add(Dropout(0.2))

model.add(GRU(4, activation='tanh',return_sequences=True))
model.add(Dropout(0.1))

model.add(GRU(2, activation='tanh'))
model.add(Dropout(0.2))

#adding output layer with softmax activation function with units 0 for normal and 1 for attack
model.add(Dense(2,activation='softmax'))


#using an adam optimizer
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='mse',
    optimizer=opt,
    metrics=['accuracy'],
)

model.summary()

#initializing time instance to calculate the trianing time
start_time = time.time()

#training my model with 30 epochs and using testing data for validation
history = model.fit(x_train_3d,
          y_train,
          epochs=30,
            batch_size = 25,
          validation_data=(x_test_3d, y_test))


print("--- %s seconds ---" % (time.time() - start_time))



predictions = model.predict(x_test_3d)



#this step is necessary if you used to predict the labels of a 3 dimensional data
predicted = np.argmax(predictions, axis = 1)
# predicted = predictions
print("predicted labels are ",predicted)
print(predicted.dtype)
print(predicted.shape)


true_lbls = np.argmax(y_test, axis=1)
print(true_lbls.dtype)
print(true_lbls.shape)



print("true labels are",true_lbls)



#plot history: accuracy
plt.plot(history.history['val_loss'])
plt.title('Validation loss history DDoS GRU')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.show()

# Plot history: Accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation accuracy history DDoS GRU')
plt.ylabel('Accuracy value (%)')
plt.xlabel('No. epoch')
plt.show()





#calculating metrics 
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
accuracy = accuracy_score(true_lbls,predicted)
print('accuracy_score is',accuracy)
precision = precision_score(true_lbls,predicted)
print("precision is ", precision )
recall = recall_score(true_lbls,predicted)
print("recall is", recall )
f1Score = f1_score(true_lbls,predicted)
print("f1_score is",f1Score)




test_scores = model.evaluate(x_test_3d, y_test, verbose=2)
print(test_scores)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])