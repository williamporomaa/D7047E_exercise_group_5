#Import necessary libraries

#------------------------------------------------------
# Tensor flow
import tensorflow as tf
print(tf.__version__)

# Keras imports
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU
from keras.layers import Flatten, Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.utils import np_utils
#-------------------------------------------------------

from TrainFeeder import TrainFeeder

import pandas as pd
import nltk
import spacy
import sklearn
import csv
import os
import random

# Utilities imports
import numpy as np
import re
import json
from argparse import Namespace
from collections import Counter
import string
from tqdm import tqdm_notebook

# Science imports (sklearn)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#Define important constants
X = []
y = []

Data_path = '/home/gyokov/HateSpeech/HASOC/Data/'

rSeed = 42
np.random.seed(rSeed) 

vocabSize = 5000
maxLength = 100

#Corpus = 'OLID'
Corpus = 'HASOC'

if Corpus == 'OLID':
    Train_name = 'OLID_Train_ATUSER_URL_EmojiRemoved_Pedro.txt'
    Test_name = 'OLID_TEST_ATUSER_URL_EmojiRemoved_Pedro.txt'
    save_dir = '/home/gyokov/HateSpeech/HASOC/Journal_Models/OLID/'

if Corpus == 'HASOC':
    Train_name = 'HASOC_Train_ATUSER_URL_EmojiRemoved_Pedro.txt'
    Test_name = 'HASOC_Test_ATUSER_URL_EmojiRemoved_Pedro.txt'
    save_dir = '/home/gyokov/HateSpeech/HASOC/Journal_Models/HASOC/'
    
#Read the task from files
#Preparation of tweets
def simplePreparationX(X, vocabSize, max_length): #prepare the data as Keras likes
    encoded_docs = [one_hot(d, vocabSize) for d in X] #encode the tweets
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post') #pad the to 100 keras expect all fed items to be the same size
    return padded_docs

#Preparation of labels
def simplePreparationY(y):
    y_encoded = [] # for y create a list  
    binary = True
    for elem in y:
        if elem == 'HATE': # HATE is encoded as 0 
            y_encoded.append(0)
        elif elem == 'PRFN': #PRFN as 1 
            y_encoded.append(1)
        elif elem == 'TIN': #TIN as 0 
            y_encoded.append(0)
        elif elem == 'UNT': #UNT as 1 
            y_encoded.append(1)
        elif elem == 'NOT': #NOT as 0
            y_encoded.append(0)
        elif elem == 'HOF': #HOF as 1
            y_encoded.append(1)
        #This is just here if I don't want to tackle the limited task 2, but an extended version, where
        #NONE is still there, so there are four classes instead of 3
        elif elem == 'NONE':
            y_encoded.append(3)
        else:
            y_encoded.append(2) #OFFN as 2
            binary = False
    y_encoded = np_utils.to_categorical(y_encoded) #like is better to use one-hot vector for categories do that 2 => [0, 0, 1]    
    return y_encoded


dfHateFile = pd.read_csv(Data_path+Train_name, sep='\t')
#Shuffle the train set
dfHateFile = dfHateFile.sample(frac=1,random_state=rSeed).reset_index(drop=True)

#--------------------------------
#Prepare the tasks from the files
#Now we only work with task1, so the others are not important anymore

Task1_features = simplePreparationX(dfHateFile['tweet'],vocabSize,maxLength) 
Task1_labels = simplePreparationY(dfHateFile['task_1'])

dfTestFile = pd.read_csv(Data_path+Test_name, sep='\t')
Test_features = simplePreparationX(dfTestFile['tweet'],vocabSize,maxLength)
Task_test_labels = simplePreparationY(dfTestFile['task_1'])

#----------------------------------------------------
#Create train/dev splits
Task_train_features, Task_dev_features, Task_train_labels, Task_dev_labels = train_test_split(Task1_features,Task1_labels, test_size=0.01) #split the X,y 3 ways classifiers
Task_train_feeder = TrainFeeder(Task_train_features, Task_train_labels)

Task_classes = ['NOT', 'HOF']

#Define model
def kerasModelsFunctions(monitor='val_acc', verbose=2, patience=5, restore_best_weights=True): #a neat helper that ends trainig if it is increasing
    early_stop = EarlyStopping(monitor='val_acc', verbose=verbose, patience=patience)
    return early_stop

def kerasModelCreating(maxLength, vocabSize, emb_dim, dropout_rate, lstm_dim, dense_dim, param_class=3): #first model a sequental model one in one out
    # define the model
    model2 = Input(shape=(maxLength,)) #create the input layer 
    embed = Embedding(vocabSize, emb_dim, input_length=maxLength)(model2) #embedding after alays in this order
    conv1 = Conv1D(64, kernel_size=8, activation='tanh')(embed) #convolution of 1D tanh activated 6ngrams
    conv2 = Conv1D(32, kernel_size=6, activation='tanh')(conv1) #convolution of 1D tanh activated 6ngrams    
    
    pool1 = MaxPooling1D(pool_size=4)(conv1) #pool of 2
    bidir1 = Bidirectional(LSTM(lstm_dim[0], return_sequences=True))(pool1) #stack some Bidirectional LSTM return sequeces is to be able to pass them to another RNN
    bidir2 = Bidirectional(LSTM(lstm_dim[1], return_sequences=True))(bidir1) # 2 less units
    bidir3 = Bidirectional(LSTM(lstm_dim[2]))(bidir2) # last layer of RNN dont call return sequences if the next is a FC layer
    hidden1 = Dense(dense_dim[0], activation='relu')(bidir3) #was flatten the connected previous layer now is the BiLSTM
    drop1 = Dropout(dropout_rate)(hidden1) # drop connected next
    hidden2 = Dense(dense_dim[1], activation='relu')(drop1)# another dense less units
    drop2 = Dropout(dropout_rate)(hidden2) # again drop varying this parameter changes the accuracy also slightly
    hidden3 = Dense(dense_dim[2], activation='relu')(drop2)# Fully Connected layer less unites
    drop3 = Dropout(dropout_rate)(hidden3) #drop less units    
    out = Dense(param_class, activation='softmax')(drop3) #classification layer 3 classes    
    net = Model(inputs=model2, outputs=out) #create the model
    #print(net.summary())
    
    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) #compile the model being a statci framework need to happen

    return net

#Define methods for training
def train_epoch(KerasModel, train_feeder, batch_size):
    finished_epochs = train_feeder.epochs_completed
    #While the epoch is not finished
    while finished_epochs == train_feeder.epochs_completed:
        #Get a batch
        current_batch_features, current_batch_labels = train_feeder.next_batch(batch_size)        
        KerasModel.train_on_batch(current_batch_features, current_batch_labels)  
        
def train_epoch_ps(KerasModel, train_feeder, batch_size, lambda_param):
    finished_epochs = train_feeder.epochs_completed
    #While the epoch is not finished
    while finished_epochs == train_feeder.epochs_completed:
        #Get a batch
        current_batch_features, current_batch_labels = train_feeder.next_batch_ps(batch_size, lambda_param)        
        KerasModel.train_on_batch(current_batch_features, current_batch_labels)      

def eval_epoch(KerasModel, validation_features, validation_labels, metric):
    predictions = KerasModel.predict(validation_features)
    if (metric == 'accuracy'):
        return sklearn.metrics.accuracy_score(np.argmax(validation_labels,axis=1),np.argmax(predictions,axis=1))
    if (metric == 'macroF1'):
        return sklearn.metrics.f1_score(np.argmax(validation_labels,axis=1),np.argmax(predictions,axis=1), average = 'macro')
    
def eval_epoch_ltd(KerasModel, validation_features, validation_labels, metric):
    predictions = KerasModel.predict(validation_features)    
    predicted_classes = np.argmax(predictions,axis=1)
    print('Original prediction')
    print(predicted_classes)
    print(predicted_classes.shape)
    for i in range(0,predicted_classes.shape[0]):
        if predicted_classes[i] > 2:
            predicted_classes[i] = 1
        else:
            predicted_classes[i] = 0
    print('Original prediction')
    print(predicted_classes)
    true_classes = np.argmax(validation_labels,axis=1)
    print(true_classes)
    for i in range(0,true_classes.shape[0]):
        if true_classes[i] > 2:
            true_classes[i] = 1
        else:
            true_classes[i] = 0
    print(true_classes)
    if (metric == 'macroF1'):
        return sklearn.metrics.f1_score(true_classes,predicted_classes, average = 'macro')

#Defining model parameters
emb_dim = 300
dropout_rate = 0.4
lstm_dim = [800, 600, 200]
dense_dim = [400, 400, 200]
batch_size = 64
dropout_rates = [0.4]
max_epoch = 3

#Actual training...
for dropout_rate in dropout_rates:
    print(dropout_rate)
    net_no = 1
    save_file_base = Corpus + '_maxLength' + str(maxLength) + '_vocabSize' + str(vocabSize) + '_emb_dim' + str(emb_dim) + 'LSTM_' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_dropout' + repr(dropout_rate) + '_batchSize' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_valid01' + '_NetNo' + str(net_no)
    save_file = save_file_base + '.h5'
    save_file_result = save_file_base + '_result.txt'
    save_file_final_result = save_file_base + '_final_result.txt'

    Task_Model = kerasModelCreating(maxLength, vocabSize, emb_dim, dropout_rate, lstm_dim, dense_dim, Task_train_labels.shape[1])
    validation_now = eval_epoch(Task_Model, Task_dev_features, Task_dev_labels, 'macroF1')
    print('Epoch no: 0')
    print('Validation score: ', validation_now)
    with open(save_dir + save_file_result, "a+") as fid:
        fid.write('Epoch no: 0')
        fid.write('\n')
        fid.write('Validation score: ' + repr(validation_now))
        fid.write('\n')

    #If we don't have a final result yet, train, otherwise, retest
    if (not os.path.exists(save_dir + save_file_final_result)):
        best_validation = 0.0
        wrong_steps = 0
        for epoch_no in range(0,max_epoch):
            train_epoch(Task_Model, Task_train_feeder, batch_size)
            validation_now = eval_epoch(Task_Model, Task_dev_features, Task_dev_labels, 'macroF1')
            print('Epoch no: ', epoch_no+1)
            print('Validation score: ', validation_now)
            with open(save_dir + save_file_result, "a+") as fid:
                fid.write('Epoch no: ' + str(epoch_no+1))
                fid.write('\n')
                fid.write('Validation score: ' + repr(validation_now))
                fid.write('\n')
            if validation_now > best_validation:
                best_validation = validation_now
                wrong_steps = 0
                Task_Model.save_weights(save_dir + save_file)
            else:
                wrong_steps += 1
                Task_Model.load_weights(save_dir + save_file)
            if wrong_steps > 2:
                Task_Model.load_weights(save_dir + save_file)
                Task_train_feeder.reset()
                break
    else:
        Task_Model.load_weights(save_dir + save_file)

    validation_final = eval_epoch(Task_Model, Task_dev_features, Task_dev_labels, 'macroF1')
    print('Final validation: ', validation_final)
    test_final = eval_epoch(Task_Model, Test_features, Task_test_labels, 'macroF1')
    print('Final test: ', test_final)
                
    with open(save_dir + save_file_final_result, "a+") as fid:    
        fid.write('Validation score: ' + repr(validation_final))
        fid.write('\n')
        fid.write('Test score: ' + repr(test_final))
        fid.write('\n')    