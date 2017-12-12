
import numpy as np
import pandas as pd
import os
import re
import glob
import csv

np.random.seed(1337)

from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split

from dbn.tensorflow import SupervisedDBNClassification

PREPROCESSED_DATA_FOLDER = 'processed_images/'

# Model_Location = 'train/'#8_2_Network_5_5/' #raw_input('Location of Model == ')
Model_Location = raw_input('Location of Model == ')

def get_dataset(tz):
    
    global PREPROCESSED_DATA_FOLDER
    files = PREPROCESSED_DATA_FOLDER+'*Zone{}.npy'.format(tz)
    print('Loading in '+files)
    f = glob.glob(files)
    full_Y = np.genfromtxt('converted_stage1_labels.csv',delimiter=',')
    Y = full_Y[:,tz-1]
    X = np.empty([len(f),62500])
    for i in range(len(f)):
        tmp = np.load(f[i])
        tmp = np.reshape(tmp,[1,62500])
        X[i,:] = tmp

    return X,Y


f = open(Model_Location+'Output_Acc_Sum.txt','w')
for tz in range(0,17):
    print('\nChecking Accuracy of NN for zone {}'.format(tz+1))
    filename = Model_Location+'Matt_Net_Zone_{}.pkl'.format(tz+1)
    print('Loading '+filename)
    My_Net = SupervisedDBNClassification.load(filename)
    X,Y = get_dataset(tz+1)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
    Yp = My_Net.predict(X_test)
    score = accuracy_score(Y_test,Yp)
    print('NN for Zone {} accuracy == {}'.format(tz+1,score))
    f.write('Zone, {}, accuracy, {}\n'.format(tz+1,score))

f.close()
    
