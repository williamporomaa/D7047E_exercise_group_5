#Important: since in this version I am writing everything to OneDrive, I try to wrap all write operations in structures like so:
#for attempt in range(10):
#    try:
#        # do thing
#    except:
#        # perhaps reconnect, etc.
#    else:
#        break
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
from keras import optimizers
#Counting parameters
from keras.utils.layer_utils import count_params

import keras.backend as K
#-------------------------------------------------------

from TrainFeeder import TrainFeeder

import pandas as pd
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

#tensorboard imports
from tensorboard.plugins.hparams import api as hp
import tensorflow.summary as summary

#Garbage collection
import gc

#Define important constants
X = []
y = []

rSeed = 42
np.random.seed(rSeed) 

#Define parameters that do not depend on the random search
Fold_max = 5
Corpus = 'HateBaseAgree_OLID_HASOC'
lambda_now = 0.0
optMethod = 'adam'
max_epoch = 100
if Corpus == 'HateBaseAgree_OLID_HASOC':
    log_dir = '/home/gyokov/OneDrive/HateSpeech/SpringerNature/Tensorboard/HateBaseAgree_OLID_HASOC/'
if Corpus == 'OLID':
    log_dir = '/home/gyokov/OneDrive/HateSpeech/SpringerNature/Tensorboard/OLID/'
if Corpus == 'HASOC':
    log_dir = '/home/gyokov/OneDrive/HateSpeech/SpringerNature/Tensorboard/HASOC_extensive/'
if Corpus == 'OLID_HASOC':
    log_dir = '/home/gyokov/OneDrive/HateSpeech/SpringerNature/Tensorboard/OLID_HASOC_extensive/'
maxLength = 128

#Define tensorboard metrics
METRIC_MacroF1 = []
METRIC_WeightedF1 = []
METRIC_MacroF1_Test = []
METRIC_WeightedF1_Test = []
for Fold_now in range(0,Fold_max):
    METRIC_MacroF1.append('Validation macro f1-score in fold ' + str(Fold_now+1))
    METRIC_WeightedF1.append('Validation weighted f1-score in fold ' + str(Fold_now+1))
    METRIC_MacroF1_Test.append('Test macro f1-score in fold ' + str(Fold_now+1))
    METRIC_WeightedF1_Test.append('Test weighted f1-score in fold ' + str(Fold_now+1))
METRIC_MacroF1_avg = 'Validation average macro f1 score'
METRIC_WeightedF1_avg = 'Validation average weighted f1 score'
METRIC_MacroF1_Test_avg = 'Test average macro f1 score'
METRIC_WeightedF1_Test_avg = 'Test average weighted f1 score'
METRIC_Parameter_count = 'No. of parameters'



#Parameters for the Neural Network training
if (not os.path.isdir(log_dir)):
    os.mkdir(log_dir)
HP_VocabSize = hp.HParam('vocabulary_size', hp.Discrete([5000,10000,20000,30000,40000,50000]))   
HP_BatchSize = hp.HParam('batch size', hp.Discrete([32,64]))
HP_CNN_ONE = hp.HParam('CNN layer 1 size', hp.Discrete([64,128])) 
HP_CNN_TWO = hp.HParam('CNN layer 2 size', hp.Discrete([0,32,64]))
HP_CNN_POOL = hp.HParam('Pooling size', hp.Discrete([2,4]))
HP_LSTM_ONE = hp.HParam('LSTM layer 1 size', hp.Discrete([600,800,1000]))
HP_LSTM_TWO = hp.HParam('LSTM layer 2 size', hp.Discrete([0,400,600,800]))
HP_LSTM_THREE = hp.HParam('LSTM layer 3 size', hp.Discrete([0,100,200,400]))
HP_DENSE_ONE = hp.HParam('Dense layer 1 size', hp.Discrete([200,400,600]))
HP_DENSE_TWO = hp.HParam('Dense layer 2 size', hp.Discrete([0,100,200,400]))
HP_DENSE_THREE = hp.HParam('Dense layer 3 size', hp.Discrete([0,100,200]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2,0.3,0.4,0.5]))
HP_eDIM = hp.HParam('embedding dimension', hp.Discrete([100,150,200,250,300,350,400]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([0.002,0.001,0.0005]))
HPARAMS = [HP_VocabSize,HP_BatchSize,HP_CNN_ONE,HP_CNN_TWO,HP_CNN_POOL,HP_LSTM_ONE,HP_LSTM_TWO,HP_LSTM_THREE,HP_DENSE_ONE,HP_DENSE_TWO,HP_DENSE_THREE,HP_DROPOUT,HP_eDIM,HP_LR]
#If the logging directory is empty, create the summary
if (len(os.listdir(log_dir))==0):
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
            hparams=[HP_VocabSize,HP_BatchSize,HP_CNN_ONE,HP_CNN_TWO,HP_CNN_POOL,HP_LSTM_ONE,HP_LSTM_TWO,HP_LSTM_THREE,HP_DENSE_ONE,HP_DENSE_TWO,HP_DENSE_THREE,HP_DROPOUT,HP_eDIM,HP_LR],
            metrics=[hp.Metric(METRIC_MacroF1[0], display_name = 'Macro F1-score at each step in fold 1'), hp.Metric(METRIC_MacroF1[1], display_name = 'Macro F1-score at each step in fold 2'),hp.Metric(METRIC_MacroF1[2], display_name = 'Macro F1-score at each step in fold 3'),hp.Metric(METRIC_MacroF1[3], display_name = 'Macro F1-score at each step in fold 4'),hp.Metric(METRIC_MacroF1[4], display_name = 'Macro F1-score at each step in fold 5'), hp.Metric(METRIC_MacroF1_avg, display_name = 'Final Macro F1-score, as average of folds'),hp.Metric(METRIC_WeightedF1[0], display_name = 'Weighted F1-score at each step in fold 1'), hp.Metric(METRIC_WeightedF1[1], display_name = 'Weighted F1-score at each step in fold 2'),hp.Metric(METRIC_WeightedF1[2], display_name = 'Weighted F1-score at each step in fold 3'),hp.Metric(METRIC_WeightedF1[3], display_name = 'Weighted F1-score at each step in fold 4'),hp.Metric(METRIC_WeightedF1[4], display_name = 'Weighted F1-score at each step in fold 5'), hp.Metric(METRIC_WeightedF1_avg, display_name = 'Final Weighted F1-score, as average of folds'),hp.Metric(METRIC_MacroF1_Test[0], display_name = 'Test macro F1-score at each step in fold 1'), hp.Metric(METRIC_MacroF1_Test[1], display_name = 'Test macro F1-score at each step in fold 2'),hp.Metric(METRIC_MacroF1_Test[2], display_name = 'Test macro F1-score at each step in fold 3'),hp.Metric(METRIC_MacroF1_Test[3], display_name = 'Test macro F1-score at each step in fold 4'),hp.Metric(METRIC_MacroF1_Test[4], display_name = 'Test macro F1-score at each step in fold 5'), hp.Metric(METRIC_MacroF1_Test_avg, display_name = 'Final Test macro F1-score, as average of folds'),hp.Metric(METRIC_WeightedF1_Test[0], display_name = 'Test weighted F1-score at each step in fold 1'), hp.Metric(METRIC_WeightedF1_Test[1], display_name = 'Test weighted F1-score at each step in fold 2'),hp.Metric(METRIC_WeightedF1_Test[2], display_name = 'Test weighted F1-score at each step in fold 3'),hp.Metric(METRIC_WeightedF1_Test[3], display_name = 'Test weighted F1-score at each step in fold 4'),hp.Metric(METRIC_WeightedF1_Test[4], display_name = 'Test weighted F1-score at each step in fold 5'), hp.Metric(METRIC_WeightedF1_Test_avg, display_name = 'Final Test weighted F1-score, as average of folds'), hp.Metric(METRIC_Parameter_count, display_name = 'No. of parameters')],
        )

Data_path = '/home/gyokov/HateSpeech/HASOC/Data/' 
    
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
        #elif elem == 'NONE':
        #    y_encoded.append(3)
        else:
             print(elem)
        #    y_encoded.append(2) #OFFN as 2
        #    binary = False
    y_encoded = np_utils.to_categorical(y_encoded) #like is better to use one-hot vector for categories do that 2 => [0, 0, 1]    
    return y_encoded

def GetFeatures(Corpus_now, Fold_now):
    if Corpus_now == 'HateBaseAgree_OLID_HASOC':
        Train_name = 'HateBaseAgree_OLID_HASOC8020_Train' + str(Fold_now) + '_ATUSER_URL_EmojiRemoved_Pedro.txt'
        Dev_name = 'HASOC8020_Dev' + str(Fold_now) + '_ATUSER_URL_EmojiRemoved_Pedro.txt'
        Test_name = 'HASOC_Test_ATUSER_URL_EmojiRemoved_Pedro.txt'
        save_dir = '/home/gyokov/OneDrive/HateSpeech/SpringerNature/Models/CNNLSTM/HateBase_OLID_HASOC/'
    if Corpus_now == 'OLID':
        Train_name = 'OLID_Train_ATUSER_URL_EmojiRemoved_Pedro.txt'
        Dev_name = 'OLID_TEST_ATUSER_URL_EmojiRemoved_Pedro.txt'
        Test_name = 'OLID_TEST_ATUSER_URL_EmojiRemoved_Pedro.txt'
        save_dir = '/home/gyokov/OneDrive/HateSpeech/SpringerNature/Models/CNNLSTM/OLID/'
    if Corpus_now == 'HASOC':
        Train_name = 'HASOC8020_Train' + str(Fold_now) + '_ATUSER_URL_EmojiRemoved_Pedro.txt'
        print(Train_name)
        Dev_name = 'HASOC8020_Dev' + str(Fold_now) + '_ATUSER_URL_EmojiRemoved_Pedro.txt'
        Test_name = 'HASOC_Test_ATUSER_URL_EmojiRemoved_Pedro.txt'
        save_dir = '/home/gyokov/OneDrive/HateSpeech/SpringerNature/Models/CNNLSTM/HASOC/'
        print(Train_name)
        print(Dev_name)
        print(Test_name)
    if Corpus_now == 'OLID_HASOC':
        Train_name = 'OLID_HASOC8020_Train' + str(Fold_now) + '_ATUSER_URL_EmojiRemoved_Pedro.txt'
        Dev_name = 'HASOC8020_Dev' + str(Fold_now) + '_ATUSER_URL_EmojiRemoved_Pedro.txt'
        Test_name = 'HASOC_Test_ATUSER_URL_EmojiRemoved_Pedro.txt'
        save_dir = '/home/gyokov/OneDrive/HateSpeech/SpringerNature/Models/CNNLSTM/OLID_HASOC/'

    dfHateFile_train = pd.read_csv(Data_path+Train_name, sep='\t')
    #Shuffle the train set
    dfHateFile_train = dfHateFile_train.sample(frac=1,random_state=rSeed).reset_index(drop=True)
    dfHateFile_dev = pd.read_csv(Data_path+Dev_name, sep='\t')
    dfHateFile_test = pd.read_csv(Data_path+Test_name, sep='\t')

    #--------------------------------
    #Prepare the tasks from the files
    #Now we only work with task1, so the others are not important anymore
    
    Train_features = simplePreparationX(dfHateFile_train['tweet'],vocabSize,maxLength) 
    Train_labels = simplePreparationY(dfHateFile_train['task_1'])
    Train_feeder = TrainFeeder(Train_features, Train_labels)
    Dev_features = simplePreparationX(dfHateFile_dev['tweet'],vocabSize,maxLength) 
    Dev_labels = simplePreparationY(dfHateFile_dev['task_1'])
    Test_features = simplePreparationX(dfHateFile_test['tweet'],vocabSize,maxLength)
    Test_labels = simplePreparationY(dfHateFile_test['task_1'])
    return Train_feeder, Dev_features, Dev_labels, Test_features, Test_labels, save_dir,




#Define model
def kerasModelsFunctions(monitor='val_acc', verbose=2, patience=5, restore_best_weights=True): #a neat helper that ends trainig if it is increasing
    early_stop = EarlyStopping(monitor='val_acc', verbose=verbose, patience=patience)
    return early_stop

def kerasModelCreating(maxLength, vocabSize, emb_dim, dropout_rate, cnn_no, cnn_dim, cnn_pool, lstm_no, lstm_dim, dense_no, dense_dim, optimizer_method, param_class=2): #first model a sequental model one in one out
    # define the model
    model2 = Input(shape=(maxLength,)) #create the input layer 
    embed = Embedding(vocabSize, emb_dim, input_length=maxLength)(model2) #embedding after alays in this order
    for layer in range(0,cnn_no):
        if layer == 0:
            conv = Conv1D(cnn_dim[layer], kernel_size=8, activation='tanh')(embed) #convolution of 1D tanh activated 8grams
            pool = MaxPooling1D(pool_size=cnn_pool)(conv) #pool of 1
        else:        
            conv = Conv1D(cnn_dim[layer], kernel_size=6, activation='tanh')(pool) #convolution of 1D tanh activated 6ngrams        
            pool = MaxPooling1D(pool_size=cnn_pool)(conv) #pool of 2
    for layer in range(0,lstm_no):
        if (layer == 0):
            if (lstm_no > 1):
                bidir = Bidirectional(LSTM(lstm_dim[layer], return_sequences=True))(pool) #stack some Bidirectional LSTM return sequeces is to be able to pass them to another RNN
            else:
                bidir = Bidirectional(LSTM(lstm_dim[layer]))(pool) #stack some Bidirectional LSTM return sequeces is to be able to pass them to another RNN
        elif (layer < (lstm_no-1)):
            bidir = Bidirectional(LSTM(lstm_dim[layer], return_sequences=True))(bidir) #stack some Bidirectional LSTM return sequeces is to be able to pass them to another RNN
        else: 
            bidir = Bidirectional(LSTM(lstm_dim[layer]))(bidir)
    for layer in range(0,dense_no):
        if layer == 0:
            hidden = Dense(dense_dim[layer], activation='relu')(bidir) #was flatten the connected previous layer now is the BiLSTM
            drop = Dropout(dropout_rate)(hidden) # drop connected next
        else:
            hidden = Dense(dense_dim[0], activation='relu')(drop) #was flatten the connected previous layer now is the BiLSTM
            drop = Dropout(dropout_rate)(hidden) # drop connected next    
    out = Dense(param_class, activation='softmax')(drop) #classification layer 3 classes    
    net = Model(inputs=model2, outputs=out) #create the model
    #print(net.summary())
    if (optimizer_method == 'sgd'):
        net.compile(optimizer=optimizers.SGD(lr=0.01, nesterov = True, clipvalue=0.5), loss='categorical_crossentropy', metrics=['acc']) #compile the model being a statci framework need to happen
    else: 
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
    if (metric == 'weightedF1'):
        return sklearn.metrics.f1_score(np.argmax(validation_labels,axis=1),np.argmax(predictions,axis=1), average = 'weighted')
    if (metric == 'probability'):
        return predictions
    
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




#Actual training...
Hparam_rand_seed = 70
rng = random.Random(Hparam_rand_seed)
for session_num in range(72,81):
    hparams = {h:h.domain.sample_uniform(rng) for h in HPARAMS}
    vocabSize = hparams[HP_VocabSize]    
    batch_size = hparams[HP_BatchSize]
    emb_dim = hparams[HP_eDIM]
    dropout_rate = hparams[HP_DROPOUT]
    cnn_no = 2
    lstm_no = 3
    dense_no = 3
    if hparams[HP_CNN_TWO] == 0:
        cnn_no = 1
    cnn_dim = [hparams[HP_CNN_ONE],hparams[HP_CNN_TWO]]
    cnn_pool = hparams[HP_CNN_POOL]
    if hparams[HP_LSTM_THREE] == 0:
        lstm_no = 2
    if hparams[HP_LSTM_TWO] == 0:
        hparams[HP_LSTM_THREE] = 0
        lstm_no = 1
    lstm_dim = [hparams[HP_LSTM_ONE],hparams[HP_LSTM_TWO],hparams[HP_LSTM_THREE]]
    if hparams[HP_DENSE_THREE] == 0:
        dense_no = 2
    if hparams[HP_DENSE_TWO] == 0:
        hparams[HP_DENSE_THREE] = 0
        dense_no = 1
    dense_dim = [hparams[HP_DENSE_ONE],hparams[HP_DENSE_TWO],hparams[HP_DENSE_THREE]]
    LR = hparams[HP_LR]
    for net_no in range(1,4):
        run_name = 'run-'+str(session_num)+'-'+str(net_no)
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        logging_file = log_dir + run_name
        writer = summary.create_file_writer(logging_file)
        for attempt in range(10):
            try:
                with writer.as_default():
                    hp.hparams(hparams)
            except:
                print('Creating hparams file failed at attempt ' + str(attempt+1))
            else:
                break
        with writer.as_default():
            hp.hparams(hparams)
        average_validation_final = 0.0
        average_weighted_validation_final = 0.0
        average_test_final = 0.0
        average_weighted_test_final = 0.0            
        for Fold_no in range(1,Fold_max+1):
            right_steps = 0
            current_LR = LR
            Task_train_feeder, Task_dev_features, Task_dev_labels, Task_test_features, Task_test_labels, save_dir = GetFeatures(Corpus, Fold_no)    
            if (optMethod == 'sgd'):
                if (Fold_no > 0):
                    save_file_base = 'RandS' + str(Hparam_rand_seed) + '_' + Corpus + '8020' + '_Fold' + str(Fold_no) + '_' + optMethod + '_LR' + repr(LR)[2:]  + 'halving_Length' + str(maxLength) + '_Vocab' + str(vocabSize) + '_EmbD' + str(emb_dim) + '_CNN' + str(cnn_dim[0]) + 'X' + str(cnn_dim[1]) + '_LSTM' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_DO' + repr(dropout_rate) + '_batch' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_Net' + str(net_no)
                else:
                    save_file_base = Corpus + '_' + optMethod + '_LR' + repr(LR)[2:]  + 'halving_Length' + str(maxLength) + '_Vocab' + str(vocabSize) + '_EmbD' + str(emb_dim) + '_CNN' + str(cnn_dim[0]) + 'X' + str(cnn_dim[1]) + '_LSTM' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_DO' + repr(dropout_rate) + '_batch' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_Net' + str(net_no)
            elif (optMethod == 'adam'):
                if (Fold_no > 0):
                    if (lambda_now > 0.0):
                        print('Probabilistic sampling used in naming the files')
                        save_file_base = 'RandS' + str(Hparam_rand_seed) + '_' + Corpus + '8020' + '_PS' + repr(lambda_now) + '_Fold' + str(Fold_no) + '_Length' + str(maxLength) + '_Vocab' + str(vocabSize) + '_EmbD' + str(emb_dim) + '_CNN' + str(cnn_dim[0]) + 'X' + str(cnn_dim[1]) + '_LSTM' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_DO' + repr(dropout_rate) + '_batch' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_Net' + str(net_no)
                    else:
                        save_file_base = 'RandS' + str(Hparam_rand_seed) + '_' + Corpus + '8020' + '_Fold' + str(Fold_no) + '_LR' + repr(LR)[2:]  + '_Length' + str(maxLength) + '_Vocab' + str(vocabSize) + '_EmbD' + str(emb_dim) + '_CNN' + str(cnn_dim[0]) + 'X' + str(cnn_dim[1]) + '_LSTM' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_DO' + repr(dropout_rate) + '_batch' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_Net' + str(net_no)
                else:
                    if (lambda_now > 0.0):
                        print('Probabilistic sampling used in naming the files')
                        save_file_base = 'RandS' + str(Hparam_rand_seed) + '_' + Corpus + '_PS' + repr(lambda_now) + '_Length' + str(maxLength) + '_Vocab' + str(vocabSize) + '_EmbD' + str(emb_dim) + '_CNN' + str(cnn_dim[0]) + 'X' + str(cnn_dim[1]) + '_LSTM' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_DO' + repr(dropout_rate) + '_batch' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_Net' + str(net_no)                    
                    else:
                        save_file_base = 'RandS' + str(Hparam_rand_seed) + '_' + Corpus + '_Length' + str(maxLength) + '_Vocab' + str(vocabSize) + '_EmbD' + str(emb_dim) + '_CNN' + str(cnn_dim[0]) + 'X' + str(cnn_dim[1]) + '_LSTM' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_DO' + repr(dropout_rate) + '_batch' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_Net' + str(net_no)                    
            else: 
                if (Fold_no > 0):
                    save_file_base = 'RandS' + str(Hparam_rand_seed) + '_' + Corpus + '8020' + '_Fold' + str(Fold_no) + '_' + optMethod + '_Length' + str(maxLength) + '_Vocab' + str(vocabSize) + '_EmbD' + str(emb_dim) + '_CNN' + str(cnn_dim[0]) + 'X' + str(cnn_dim[1]) + '_LSTM' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_DO' + repr(dropout_rate) + '_batch' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_Net' + str(net_no)
                else:
                    save_file_base = 'RandS' + str(Hparam_rand_seed) + '_' + Corpus + '_' + optMethod + '_Length' + str(maxLength) + '_Vocab' + str(vocabSize) + '_EmbD' + str(emb_dim) + '_CNN' + str(cnn_dim[0]) + 'X' + str(cnn_dim[1]) + '_LSTM' + str(lstm_dim[0]) + 'X' + str(lstm_dim[1]) + 'X' + str(lstm_dim[2]) + '_Dense' + str(dense_dim[0]) + 'X' + str(dense_dim[1]) + 'X' + str(dense_dim[2]) + '_DO' + repr(dropout_rate) + '_batch' + str(batch_size) + '_maxEpoch' + str(max_epoch) + '_Net' + str(net_no)
            save_file = save_file_base + '.h5'            
            save_test_pred = save_file_base + '_TestOut'
            save_file_result = save_file_base + '_Res.txt'
            save_file_final_result = save_file_base + '_EndRes.txt'
            
            Task_Model = kerasModelCreating(maxLength, vocabSize, emb_dim, dropout_rate, cnn_no, cnn_dim, cnn_pool, lstm_no, lstm_dim, dense_no, dense_dim, optMethod, 2)
            #If we are on the first fold, and first net, save the model summary
            if (net_no == 1 and Fold_no == 1):
                for attempt in range(10):
                    try:
                        with open(save_dir + save_file_base + '_model.txt', 'w+') as fid:
                            Task_Model.summary(line_length=255, print_fn=lambda x: fid.write(x + '\n'))                                
                    except Exception as e:
                        print(e)
                        print('Saving model summary failed at attempt ' + str(attempt+1))
                        if attempt==9:
                            raise e
                    else:
                        break                 
                #The first parameter we can save already, that is the size of the model
                for attempt in range(10):
                    try:
                        with writer.as_default():
                            summary.scalar(METRIC_Parameter_count, count_params(Task_Model.trainable_weights), step = 0)                        
                    except:
                        print('Writing to tensorboard files failed at attempt ' + str(attempt+1))
                    else:
                        break
            #If we don't have a final result yet, train, otherwise, go to the next version
            if (not os.path.exists(save_dir + save_file_final_result)):
                if (optMethod == 'sgd' or optMethod == 'adam'):
                    K.set_value(Task_Model.optimizer.learning_rate, current_LR)                
                #validation_now = eval_epoch(Task_Model, Task_dev_features, Task_dev_labels, 'macroF1')
                #print('Epoch no: 0')
                #print('Validation score: ', validation_now)
                #with open(save_dir + save_file_result, "a+") as fid:
                #    fid.write('Epoch no: 0')
                #    fid.write('\n')
                #    fid.write('Validation score: ' + repr(validation_now))
                #    fid.write('\n')
                #Task_Model.save_weights(save_dir + save_file)
                #best_validation = validation_now
                best_validation = 0.0
                wrong_steps = 0
                for epoch_no in range(0,max_epoch):
                    print('Learning_rate: ', K.eval(Task_Model.optimizer.learning_rate))
                    #Here, let's try to catch memory errors, and see if we can free up memory using the garbage collector...
                    for attempt in range(5):
                        try:
                            # do thing
                            if (lambda_now > 0.0):
                                print('Training with probabilistic sampling')
                                train_epoch_ps(Task_Model, Task_train_feeder, batch_size, lambda_now)
                            else:
                                print('Training without probabilistic sampling')
                                train_epoch(Task_Model, Task_train_feeder, batch_size)
                        except Exception as e:
                            print(e)
                            print('Training failed, most likely due to memory problems. Engaging the garbage collector and trying again. Attempt: ' + str(attempt+1))
                            gc.collect()
                            # perhaps reconnect, etc.
                            if (attempt==4):
                                raise e
                        else:
                            break

                    validation_now = eval_epoch(Task_Model, Task_dev_features, Task_dev_labels, 'macroF1')
                    weighted_validation_now = eval_epoch(Task_Model, Task_dev_features, Task_dev_labels, 'weightedF1')
                    print('Epoch no: ', epoch_no+1)
                    print('Validation score: ', validation_now)
                    for attempt in range(10):
                        try:
                            with open(save_dir + save_file_result, "a+") as fid:
                                fid.write('Epoch no: ' + str(epoch_no+1))
                                fid.write('\n')
                                fid.write('Validation score: ' + repr(validation_now))
                                fid.write('\n')
                        except:
                            print('Writing to results file failed at attempt ' + str(attempt+1))
                        else:
                            break                    
                    if validation_now > best_validation:
                        best_validation = validation_now
                        wrong_steps = 0
                        for attempt in range(10):
                            try:
                                Task_Model.save_weights(save_dir + save_file)
                            except Exception as e:
                                print(e)
                                print('Saving the model failed at attempt ' + str(attempt+1))
                            else:
                                break
                        #Evaluate the test set as well...
                        test_macro_now = eval_epoch(Task_Model, Task_test_features, Task_test_labels, 'macroF1')
                        test_weighted_now = eval_epoch(Task_Model, Task_test_features, Task_test_labels, 'weightedF1')
                        for attempt in range(10):
                            try:
                                with writer.as_default():
                                    summary.scalar(METRIC_MacroF1[Fold_no-1], best_validation, step = right_steps)
                                    summary.scalar(METRIC_WeightedF1[Fold_no-1], weighted_validation_now, step = right_steps)
                                    summary.scalar(METRIC_MacroF1_Test[Fold_no-1], test_macro_now, step = right_steps)
                                    summary.scalar(METRIC_WeightedF1_Test[Fold_no-1], test_weighted_now, step = right_steps)
                            except:
                                print('Saving tensorboard results failed at attempt' + str(attempt+1))
                            else:
                                break
                        right_steps += 1
                    else:
                        wrong_steps += 1
                        for attempt in range(10):
                            try:
                                Task_Model.load_weights(save_dir + save_file)
                            except:
                                print('Loading the model failed at attempt ' + str(attempt+1))
                            else:
                                break                        
                        if (optMethod == 'sgd' or optMethod == 'adam'):
                            current_LR = current_LR / 2
                            K.set_value(Task_Model.optimizer.learning_rate, current_LR)
                            print('Current learning rate: ', K.eval(Task_Model.optimizer.learning_rate))
                        if (optMethod == 'adam_then_sgd'):
                            optMethod = 'sgd'
                            Task_Model = kerasModelCreating(maxLength, vocabSize, emb_dim, dropout_rate, cnn_no, cnn_dim, cnn_pool, lstm_no, lstm_dim, dense_no, dense_dim, optMethod, 2)
                            for attempt in range(10):
                                try:
                                    Task_Model.load_weights(save_dir + save_file)
                                except:
                                    print('Loading the model failed at attempt ' + str(attempt+1))
                                else:
                                    break
                            current_LR = current_LR / 2
                            K.set_value(Task_Model.optimizer.learning_rate, current_LR)
                            print('Current learning rate: ', K.eval(Task_Model.optimizer.learning_rate))
                    if wrong_steps > 2:
                        for attempt in range(10):
                            try:
                                Task_Model.load_weights(save_dir + save_file)
                            except:
                                print('Loading the model failed at attempt ' + str(attempt+1))
                            else:
                                break
                        Task_train_feeder.reset()
                        break
            #If the final result already exists, we already trained networks with those parameters, so break            
            else:                
                break
                #for attempt in range(10):
                #    try:
                #        Task_Model.load_weights(save_dir + save_file)
                #    except:
                #        print('Loading the model failed at attempt ' + str(attempt+1))
                #    else:
                #        break
            
            validation_final = eval_epoch(Task_Model, Task_dev_features, Task_dev_labels, 'macroF1')
            average_validation_final = average_validation_final + (validation_final/Fold_max)
            weighted_validation_final = eval_epoch(Task_Model, Task_dev_features, Task_dev_labels, 'weightedF1')
            average_weighted_validation_final = average_weighted_validation_final + (weighted_validation_final/Fold_max)
            print('Final validation: ', validation_final)
            test_final = eval_epoch(Task_Model, Task_test_features, Task_test_labels, 'macroF1')
            average_test_final = average_test_final + (test_final/Fold_max)
            weighted_test_final = eval_epoch(Task_Model, Task_test_features, Task_test_labels, 'weightedF1')
            average_weighted_test_final = average_weighted_test_final + (weighted_test_final/Fold_max)
            print('Final test: ', test_final)            
            for attempt in range(10):
                try:
                    with open(save_dir + save_file_final_result, "a+") as fid:    
                        fid.write('Validation score: ' + repr(validation_final))
                        fid.write('\n')
                        fid.write('Test score: ' + repr(test_final))
                        fid.write('\n')                
                except:
                    print('Writing to result file failed at attempt ' + str(attempt+1))
                else:
                    break  
            #Put the test results for each Fold in the tensorboard file ()
            for attempt in range(10):
                try:
                    with writer.as_default():
                        summary.scalar(METRIC_MacroF1_Test[Fold_no-1], test_final, step = 0)
                        summary.scalar(METRIC_WeightedF1_Test[Fold_no-1], weighted_test_final, step = 0)
                except:
                    print('Writing to tensorboard files failed at attempt ' + str(attempt+1))
                else:
                    break      
            Probabilities = eval_epoch(Task_Model, Task_test_features, Task_test_labels, 'probability')
            Classes = np.argmax(Probabilities,axis=1)
            d = {'NOT':Probabilities[:,0], 'HOF':Probabilities[:,1], 'Label':Classes}
            df = pd.DataFrame.from_dict(d)
            pred_file = save_dir + save_test_pred
            for attempt in range(10):
                try:
                    df.to_pickle(pred_file)
                except:
                    print('Pickling file failed at attempt ' + str(attempt+1))
                else:
                    break
            #If the final validation is not good enough (but not necessarily so bad), we don't need to keep the model
            if (validation_final < 0.5):
                os.remove(save_dir + save_file)
            #If the result from the current fold was very bad, there is no need to do it for all the loops
            if (validation_final < 0.3):
                break
        #This part only applies if we went through all the folds, thus Fold_no is the maximum fold now (which is usually five)
        if (Fold_no == Fold_max):
            for attempt in range(10):
                try:
                    with writer.as_default():
                        summary.scalar(METRIC_MacroF1_avg, average_validation_final, step = 0)
                        summary.scalar(METRIC_WeightedF1_avg, average_weighted_validation_final, step = 0)
                        summary.scalar(METRIC_MacroF1_Test_avg, average_test_final, step = 0)
                        summary.scalar(METRIC_WeightedF1_Test_avg, average_weighted_test_final, step = 0)
                except:
                    print('Writing to tensorboard files failed at attempt ' + str(attempt+1))
                else:
                    break
            #If the first average result was very bad, we don't need to train all 5 folds 3 times, sufficient to do it just once        
            if (average_validation_final < 0.5):
                break
        #If we did not reach the last fold, we had so very bad results that we should just break it now
        else:
            break