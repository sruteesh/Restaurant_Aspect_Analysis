
# coding: utf-8

# In[2]:

import os
import re
import sys
import json
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
# import gensim as gs
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn
from sklearn.utils import shuffle

# In[23]:

regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
#     r"(?:\\u+[\w_]+[\w\'_\-]*[\w_]+)",
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'www.?(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs 
#     r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r"(?:[0-9a-z][0-9a-z'\-_]+[0-9a-z])", # numbers with - and '
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

hastags_words = [u'disappointed',u'love',u'wtf',u'frustrated']
stop_words = [u'.',u'|', u'\xa0',u'-',u'!',u')',u'/',u'-',u',',u':',"'"]
replace_words ={
    u'\u2019' : "'",
    u'\u2026' : '',
    u'&amp;' : '&',
    u'???' : '?'
}
replace_words_1 = { 
    u'??' : '?'
}

logging.getLogger().setLevel(logging.INFO)

# training_config = 'training_config.json'
# params = json.loads(open(training_config).read())


import re
import nltk

# vocab_words = []

def  lapply(_list, fun, **kwargs):
    return [fun(item, **kwargs) for item in _list]


def clean_str_1(tokens):
    tokens = lapply(tokens, lambda x: re.sub('\#.*', '', x))
    tokens = lapply(tokens, lambda x: re.sub('\@.*', '', x))
    tokens = lapply(tokens, lambda x: re.sub('[^A-z]', '', x))
    tokens = lapply(tokens, lambda x: re.sub('[?|$|.|!]', '', x))
    tokens = lapply(tokens, lambda x: re.sub('^.*http.*', '', x))
    tokens = [token for token in tokens if len(token)>3 and token not in stoplist]
    return tokens


def stop_words_list():
    '''
        A stop list specific to the observed timelines composed of noisy words
        This list would change for different set of timelines
    '''
    return ['amp','get','got','hey','hmm','hoo','hop','iep','let','ooo','par',
            'pdt','pln','pst','wha','yep','yer','aest','didn','nzdt','via',
            'one','com','new','like','great','make','top','awesome','best',
            'good','wow','yes','say','yay','would','thanks','thank','going',
            'new','use','should','could','really','see','want','nice',
            'while','know','free','today','day','always','last','put','live',
            'week','went','wasn','was','used','ugh','try','kind', 'http','much',
            'need', 'next','app','ibm','appleevent','using','youv','thi']

stoplist  = set(  nltk.corpus.stopwords.words("english")
                    + nltk.corpus.stopwords.words("french")
                    + nltk.corpus.stopwords.words("german")
                    + stop_words_list())

def preprocess(tweet):
    #timeline_tokens = [nltk.word_tokenize(i) for i in timeline_texts]
    tweets_tokens = clean_str_1(tweet.lower().split())

    # processed_tweets = [word for word in tweets_tokens if word not in stoplist]
    return ' '.join(tweets_tokens)

def clean_sentence(sentence):
    cleaned_words = []
    sentence = sentence.lower().decode('utf8')
    # print (type(replace_words.iteritems()))
    for k,v in list(replace_words.items()) + list(replace_words_1.items()):
        if sentence.find(k) != -1:
            sentence = sentence.replace(k,v)

    for word in tokens_re.findall(sentence) :
        word = word.strip()
        if word.startswith(('http')) :
            cleaned_words.append('URL')
        elif word.startswith(('#')) :
            if word[1:] in hastags_words:
                cleaned_words.append(word[1:])
            else:
                cleaned_words.append('HASHTAG')
        elif word.startswith(('@')) :
            cleaned_words.append('MENTION')
        elif word not in stop_words:
            cleaned_words.append(word)
    return ' '.join(cleaned_words)


def clean_str(s):
	s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()


# In[37]:

def load_embeddings(vocabulary):
    word_embeddings = {}
    # print (type(vocabulary))
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    # print (word,word_embeddings[word])
    return word_embeddings

def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """Pad setences during training or prediction"""
    if forced_sequence_length is None: # Train
        sequence_length = max(len(x) for x in sentences)
    else: # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []

    if len(sentences)==0:
        num_padding = sequence_length
        padded_sentences.append([padding_word] * num_padding)
        return padded_sentences

    else:
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
                logging.info('This sentence has to be cut off because it is longer than trained sequence length')
                padded_sentence = sentence[0:sequence_length]
            else:
                padded_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(padded_sentence)
        return padded_sentences

def build_vocab(sentences,vocab_size = None):
    word_counts = Counter(itertools.chain(*sentences))
    if vocab_size:
        vocabulary_inv = [word[0] for word in word_counts.most_common(vocab_size-1)]
    else:
        vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary_inv = ['<PAD/>'] + vocabulary_inv
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_data(filename,config):
    global params,vocabulary,label_dict,labels, co_lab
    params= config

    print (filename)
    if ".zip" in filename:
        df = pd.read_csv(filename, compression='zip')
    else:
        df = pd.read_csv(filename)

    selected = ['Category', 'Descript']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))

    labels = sorted(list(set(df[selected[0]].tolist())))

    # if params['shuffle'] :

    # co_lab = {ll:0 for ll in labels}
    if params['max_data']:
        co_lab = {ll:0 for ll in labels}
        logging.critical("Data Frame's are going normalize to \n{}\n".format(params['max_data']))
        def check_count(row):
            if co_lab[row['Category']] > params['max_data']:
                return False
            else :
                co_lab[row['Category']] += 1
                return True
        df = df[df.apply(lambda row: check_count(row), axis=1)]
        df = shuffle(df)
        logging.critical("Data Shuffled")
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    logging.critical('The Outputs Classes are    \n{}\n'.format(label_dict))
    x_raw = df[selected[1]].apply(lambda x: preprocess(x).split(' ')).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    # print (x_raw[0])

    vocabulary, vocabulary_inv = build_vocab(x_raw,params['vocab_size'])

    x = [[vocabulary[word] for word in sentence if vocabulary.get(word,0)] for sentence in x_raw]
    del(x_raw)
    # print (x[0])
    x = np.array(pad_sentences(x,forced_sequence_length = params['max_sentence_length'],padding_word=0))
    y = np.array(y_raw)

    idx = len(x) - int(len(x) * params['test_size'])

    x_train, y_train = np.array(x[:idx]), np.array(y[:idx])
    x_test, y_test = np.array(x[idx:]), np.array(y[idx:])

    label_d = {word: index for index, word in enumerate(labels)}
    y_r = df[selected[0]].apply(lambda y_1: label_d[y_1]).tolist()

    del(df)
    logging.critical('The data size for x_train,y_train,x_test,y_test {} {} {} {}'.format(x_train.shape,y_train.shape,x_test.shape,y_test.shape))
    # logging.critical(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    return x_train,y_train,x_test,y_test,vocabulary,np.array(y_r[:idx]),np.array(y_r[idx:]),labels,label_d

def test_data(sentence,vocabulary,params):
    sentence = preprocess(sentence).split(' ')
    words = [[vocabulary[word] for word in sentence if vocabulary.get(word,0)]]
    # print words
    return np.array(pad_sentences(words,forced_sequence_length = params['max_sentence_length'],padding_word=0))

def  lapply(_list, fun, **kwargs):
    return [fun(item, **kwargs) for item in _list]


def test_csv(infile,vocabulary,params,not_file=False):
    
    if not_file==True:
        df = infile
    else:
        df = pd.read_csv(infile)

    if 'Category' in df.columns:
        selected = ['Category', 'Descript']
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1)
        print(df.head())
        df = df.dropna(axis=0, how='any', subset=selected)
        df = df.reindex(np.random.permutation(df.index))

        labels = sorted(list(set(df[selected[0]].tolist())))

        label_dict = {word: index for index, word in enumerate(list(set(labels)))}

        x_raw = [lapply(item, lambda x: preprocess(x).split()) for item in df[selected[1]].values]
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

        x=[]
        for rev in x_raw:
            sent_index=[]
            for sent in rev:
                sent_index.append([vocabulary[word] for word in sent if vocabulary.get(word,0)])
            x.append(sent_index)

        padded_x = []
        for rev in x:
            padded_x.append(np.array(pad_sentences(rev,forced_sequence_length = params['max_sentence_length'],padding_word=0)))

        y = np.array(y_raw)

        label_d = {word: index for index, word in enumerate(labels)}
        y_r = df[selected[0]].apply(lambda y_1: label_d[y_1]).tolist()
        # print (np.array(y_r[:10]))
        return padded_x,y, np.array(y_r),df
    else:
        selected = ['Descript']
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1)
        df = df.dropna(axis=0, how='any', subset=selected)
        df = df.reindex(np.random.permutation(df.index))

        x_raw = [lapply(item, lambda x: preprocess(x).split()) for item in df[selected[0]].values]

        # x = [[vocabulary[word] for word in sentence if vocabulary.get(word,0)] for sentence in x_raw]
        # x = [[[vocabulary[word] for word in sent if vocabulary.get(word,0)] for sent in rev] for rev in x_raw]
        x=[]
        for rev in x_raw:
            print(rev)
            sent_index=[]
            for sent in rev:
                sent_index.append([vocabulary[word] for word in sent if vocabulary.get(word,0)])
            x.append(sent_index)

        padded_x = []
        for rev in x:
            padded_x.append(np.array(pad_sentences(rev,forced_sequence_length = params['max_sentence_length'],padding_word=0)))

        y =y_r= []

        return padded_x,y,y_r, df



def test_csv_2(infile,vocabulary,label_dict,params,not_file=False):
    
    if not_file==True:
        df = infile
    else:
        df = pd.read_csv(infile)

    if 'Category' in df.columns:
        selected = ['Category', 'Descript']
        non_selected = list(set(df.columns) - set(selected))
        df = df.drop(non_selected, axis=1)
        print(df.head())
        # df = df.dropna(axis=0, how='any', subset=selected)
        df = df.reindex(np.random.permutation(df.index))

        x_raw = df[selected[1]].apply(lambda x: preprocess(x).split()).tolist()
        y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()


        x = [[vocabulary[word] for word in sentence if vocabulary.get(word,0)] for sentence in x_raw]
        x = np.array(pad_sentences(x,forced_sequence_length = params['max_sentence_length'],padding_word=0))

        y = np.array(y_raw)
        y_r = []
        # print (np.array(y_r[:10]))
        return x,y, np.array(y_r),df



# if __name__ == "__main__":
#     # print(clean_sentence("video screenshots be cuter than the actual picture 💀😂😭💯 #istg@gsg???? ."))
#     print(preprocess("@Lilou_Caron @Yayonne11 @apelletier007 On t'aiiiiiime. ðŸ˜"))
# # #     train_file = '/home/common/siva/Datasets/semEval2016/twitter_5_class_train.csv.zip'
# # #     params = {
# # #         "batch_size": 10,
# # #         "dropout_keep_prob": 0.4,
# # #         "embedding_dim": 32,
# # #         "evaluate_every": 100,
# # #         "filter_sizes": (5),
# # #         "hidden_unit": 500,
# # #         "l2_reg_lambda": 0.0,
# # #         "max_pool_size": 2,
# # #         "non_static": False,
# # #         "num_epochs": 2,
# # #         "num_filters": 256,
# # #         "max_sentence_length" : 50,
# # #         "vocab_size":5000,
# # #         "shuffle":1,
# # #         "test_size":0.001,
# # #         "lstm_output_size":50
# # #     }
# # #     # params = json.loads(open(training_config).read())
# # #     load_data(train_file,params)
# # #     # print test_data("i love india")
# #     df = pd.read_csv('/home/common/siva/multi-class-text-classification-cnn-rnn_V1/data/new_data_11_category_v1.csv', )
# #     # print df.columns
# #     for row in df.iterrows():
# #         # print row
# #         # print type(row)
# #         try:
# #             if isinstance(row[1]['Descript'],str):
# #                 clean_sentence(row[1]['Descript'])
# #             else:
# #                 print row
# #         except Exception as e:
# #             print (e)
# #             print(row[1])

#     params = {
#         "batch_size":50,
#         "dropout_keep_prob": 0.5,
#         "embedding_dim": 50,
#         "evaluate_every": 100,
#         "filter_sizes": [3,4,5],
#         "hidden_unit": 100,
#         "l2_reg_lambda": 0.00,
#         "max_pool_size": 3,
#         "non_static": False,
#         "num_epochs": 5,
#         "num_filters": 50,
#         "max_sentence_length" : 30,
#         "vocab_size":10000,
#         "shuffle":1,
#         "test_size":0.10,
#         "lstm_output_size":50,
#         'max_data':100000
#     }
#     # infile = '/home/common/siva/Datasets/imdb.csv.zip'

#     # infile = '/home/common/siva/Datasets/imdb.csv.zip'
#     import json
#     with open('../sruteesh/tripadvisor_aspects_vocabulary.json') as fout:
#         vocabulary = json.load(fout)
        
#     with open('../sruteesh/tripadvisor_aspects_labels.json') as fout2:
#         labels = json.load(fout2)

#     infile = '/home/common/siva/sruteesh/hotel_reviews_test_sample.csv'
#     # infile = '/home/common/siva/Datasets/semEval2016/twitter_5_class_test (copy).csv'
#     # infile = '/home/common/siva/multi-class-text-classification-cnn-rnn_V1/data/new_data_test.csv'
#     xx,yy,yy_1,test_df = test_csv(infile,vocabulary,params)
