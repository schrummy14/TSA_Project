
import numpy as np
import pandas as pd
import tsahelper as tsa
import os
import re
import glob
import csv

from timeit import default_timer as timer
import my_helpers as myh

np.random.seed(1337)

from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection import train_test_split

from dbn.tensorflow import SupervisedDBNClassification

# global INPUT_FOLDER
# global PREPROCESSED_DATA_FOLDER
# global STAGE1_LABELS
# global THREAT_ZONE
# global BATCH_SIZE
#
# global IMAGE_DIM
# global LEARNING_RATE
# global N_TRAIN_STEP
#
# global TRAIN_PATH
# global MODEL_PATH
# global MODEL_NAME

INPUT_FOLDER = '/home/mschramm/Documents/Classes/Stat_430/Project/TSA/tester/tsa_datasets/stage1/aps'
PREPROCESSED_DATA_FOLDER = 'processed_images/'
STAGE1_LABELS= '/home/mschramm/Documents/Classes/Stat_430/Project/TSA/tester/tsa_datasets/stage1_labels.csv'
THREAT_ZONE = 1
BATCH_SIZE = 16

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1

TRAIN_PATH = 'train/'
MODEL_PATH = 'model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format(
    'matt_net_v1',LEARNING_RATE,IMAGE_DIM,IMAGE_DIM,
    THREAT_ZONE))

TRAIN_TEST_SPLIT_RATIO = 0.2

def preprocess_tsa_data():

    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'],df['Zone']=df['Id'].str.split('_',1).str
    SUBJECT_LIST = df['Subject'].unique()

    # SUBJECT_LIST = ['00360f79fd6e02781457eda48f85da90','0043db5e8c819bffc15261b1f1ac5e42',
    #                 '0050492f92e22eed3474ae3a6fc907fa','006ec59fa59dd80a64c85347eef810c7',
    #                 '0097503ee9fa0606559c56458b281a08','011516ab0eca7cad7f5257672ddde70e']


    batch_num = 1
    threat_zone_examples = []
    start_time = timer()

    for subject in SUBJECT_LIST:

        print('-------------------------------------------------------------')
        print('t+>{:5.3f} |Reading subject #:{}'.format(timer()-start_time,subject))
        print('-------------------------------------------------------------')

        images = tsa.read_data(INPUT_FOLDER+'/'+subject+'.aps')
        images = images.transpose()

        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list,tsa.zone_crop_list)):
            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            label = np.array(tsa.get_subject_zone_label(tz_num,
                                                        tsa.get_subject_labels(STAGE1_LABELS,subject)))

            for img_num, img in enumerate(images):
                print('Threat Zone:Image -> {}:{}'.format(tz_num,img_num))
                print('Threat Zone Label -> {}'.format(label))

                if threat_zone[img_num] is not None:
                    print('-> reorienting base image')
                    base_img = np.flipud(img)
                    print('-> shape {}| mean = {}'.format(base_img.shape,
                                                          base_img.mean()))

                    print('-> rescaling image')
                    rescaled_img = tsa.convert_to_grayscale((base_img))
                    print('-> shape {}| mean = {}'.format(rescaled_img.shape,
                                                          rescaled_img.mean()))

                    print('-> making high contrast')
                    high_contrast_img = tsa.spread_spectrum(rescaled_img)
                    print('-> shape {}| mean = {}'.format(high_contrast_img.shape,
                                                          high_contrast_img.mean()))

                    masked_img = tsa.roi(high_contrast_img,threat_zone[img_num])

                    print('-> cropping image')
                    cropped_img = tsa.crop(masked_img,crop_dims[img_num])
                    print('-> shape {}| mean = {}'.format(cropped_img.shape,
                                                          cropped_img.mean()))

                    print('-> normalizing image')
                    normalized_img = tsa.normalize(cropped_img)
                    print('-> shape {}| mean = {}'.format(normalized_img.shape,
                                                          normalized_img.mean()))

                    zeroed_img = tsa.zero_center(normalized_img)
                    print('-> shape {}| mean = {}'.format(zeroed_img.shape,
                                                          zeroed_img.mean()))

                    # threat_zone_examples.append([[tz_num],zeroed_img,label])

                    np.save(PREPROCESSED_DATA_FOLDER + subject + 'Zone{}.npy'.format(
                        tz_num + 1), zeroed_img)

                    # com_img = np.reshape(zeroed_img,[1,250*250])

                else:
                    print('-> No view...')

                print('----------------------weeee-----------------------')


def data_set_to_np_array(chunk_size,tz):
    global PREPROCESSED_DATA_FOLDER

    f = glob.glob(PREPROCESSED_DATA_FOLDER+'*Zone{}.npy'.format(tz))
    if(len(f)==0):
        preprocess_tsa_data()
        X,Y = data_set_to_np_array(chunk_size,tz)
        return X,Y

    per_array = np.random.permutation(len(f))

    full_Y = np.genfromtxt('converted_stage1_labels.csv', delimiter=',')
    Y = full_Y[per_array[0:(chunk_size)],tz-1]

    big_array = np.empty([chunk_size,62500])
    for i in range(chunk_size):
        tmp = np.load(f[per_array[i]])
        tmp = np.reshape(tmp,[1,62500])
        big_array[i,:] = tmp

    return big_array,Y.astype(int)

def get_dataset(tz):
    
    global PREPROCESSED_DATA_FOLDER
    f = glob.glob(PREPROCESSED_DATA_FOLDER+'*Zone{}.npy'.format(tz))
    full_Y = np.genfromtxt('converted_stage1_labels.csv',delimiter=',')
    Y = full_Y[:,tz-1]
    X = np.empty([len(f),62500])
    for i in range(len(f)):
        tmp = np.load(f[i])
        tmp = np.reshape(tmp,[1,62500])
        X[i,:] = tmp

    return X,Y


for tz in range(5,6):
    print('\nWorking on Zone == {}'.format(tz))
    Matt_Net = SupervisedDBNClassification(hidden_layers_structure=[80, 160, 40],
                                       learning_rate_rbm=0.05,
                                       learning_rate=0.1,
                                       n_epochs_rbm=100,
                                       n_iter_backprop=400,
                                       batch_size=8,
                                       activation_function='relu',
                                       dropout_p=0.2)
    # Split Data
    X,Y = get_dataset(tz)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
    print('Size of training set == {}, Size of testing set == {}\n'.format(len(X_train),len(X_test)))

    start_time = timer()
    tot_start = start_time
    Matt_Net.pre_train(X_train)
    print('Time to pretrain == {:5.3f} seconds\n'.format(timer()-start_time))

    start_time = timer()
    Matt_Net.fit(X_train,Y_train,False)
    print('Time to fit == {:5.3f} seconds\n'.format(timer()-start_time))
    print('Total time == {:5.3f} seconds\n'.format(timer()-tot_start))

    Matt_Net.save('train/Matt_Net_Zone_{}.pkl'.format(tz))

    Y_pred = Matt_Net.predict(X_test)
    start_time = timer()
    score = accuracy_score(Y_test,Y_pred)
    print('Done, time to predict == {:5.3}\nAccuracy == {} for zone {}\n'.format(timer()-start_time,score,tz))
    
    del Matt_Net


