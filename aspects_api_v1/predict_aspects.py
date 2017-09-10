
# coding: utf-8

# In[1]:

from __future__ import print_function

# from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten,LSTM
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D
# from keras.datasets import imdb
# from keras import optimizers
from keras import regularizers
# from data_helper_keras_v1 import load_data
# import data_helper_keras_v2
from keras.models import model_from_json


# In[2]:

import os, re
import numpy as np
import pandas as pd
from collections import Counter
import itertools


import json


params_hotels = {
    "batch_size":50,
    "dropout_keep_prob": 0.5,
    "embedding_dim": 50,
    "evaluate_every": 100,
    "filter_sizes": [3,4,5],
    "hidden_unit": 100,
    "l2_reg_lambda": 0.00,
    "max_pool_size": 3,
    "non_static": False,
    "num_epochs": 5,
    "num_filters": 50,
    "max_sentence_length" : 25,
    "vocab_size":10000,
    "shuffle":1,
    "test_size":0.10,
    "lstm_output_size":50,
    'max_data':100000
}

params_restaurants = {
    "batch_size":50,
    "dropout_keep_prob": 0.5,
    "embedding_dim": 50,
    "evaluate_every": 100,
    "filter_sizes": [3,4,5],
    "hidden_unit": 100,
    "l2_reg_lambda": 0.00,
    "max_pool_size": 3,
    "non_static": False,
    "num_epochs": 5,
    "num_filters": 50,
    "max_sentence_length" : 30,
    "vocab_size":10000,
    "shuffle":1,
    "test_size":0.10,
    "lstm_output_size":50,
    'max_data':100000
}
def load_aspects_model(path = "/home/sruteeshkumar/personal/flask_demo/aspects_models/",domain ='hotels'):
    
    if domain=='hotels':
        # load json and create model
        with open(path+'tripadvisor_aspects_vocabulary_v2.json') as fout:
            vocabulary = json.load(fout)
            
        with open(path+'tripadvisor_aspects_labels_v2.json') as fout2:
            labels = json.load(fout2)
            

        json_file = open(path+"tripadvisor_aspects_detection_v2.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(path+"tripadvisor_aspects_detection_v2.h5")
        print("Loaded model from disk")
        return loaded_model,vocabulary,labels

    elif domain=='restaurants':
        # load json and create model
        with open(path+'restaurant_aspects_vocabulary.json') as fout:
            vocabulary = json.load(fout)
            
        with open(path+'restaurant_aspects_labels.json') as fout2:
            labels = json.load(fout2)

            
        json_file = open(path+"restaurant_aspects_detection.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(path+"restaurant_aspects_detection.h5")
        print("Loaded model from disk")
        return loaded_model,vocabulary,labels



