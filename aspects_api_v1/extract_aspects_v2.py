
# coding: utf-8
from __future__ import division

import nltk
import numpy as np
import sys
import pandas as pd

from nltk.stem.wordnet import WordNetLemmatizer

from aspects_api_v1.process_data.tokenizers import MyPottsTokenizer
from aspects_api_v1.process_data.asp_extractors import SentenceAspectExtractor
import aspects_api_v1.predict_aspects as predict_aspects
from aspects_api_v1.predict_aspects import params_hotels
from aspects_api_v1.predict_aspects import params_restaurants
import aspects_api_v1.data_helper_keras_v2 as data_helper_keras_v2



# from process_data.tokenizers import MyPottsTokenizer
# from process_data.asp_extractors import SentenceAspectExtractor
# from predict_aspects import params_hotels
# from predict_aspects import params_restaurants# import predict_aspects as predict_aspects
# import data_helper_keras_v2 as data_helper_keras_v2

predict_aspects_model_restaurants,vocabulary_restaurants,labels_restaurants = predict_aspects.load_aspects_model(domain='restaurants')
predict_aspects_model_hotels,vocabulary_hotels,labels_hotels = predict_aspects.load_aspects_model(domain='hotels')



# 


# print(labels)
labels_inv_hotels = {v:k for k,v in labels_hotels.items()}
labels_inv_restaurants = {v:k for k,v in labels_restaurants.items()}



def  lapply(_list, fun, **kwargs):
    return [fun(item, **kwargs) for item in _list]

def max_2(row,domain):
    if domain=='hotels':
        labels_inv = labels_inv_hotels
    if domain=='restaurants':
        labels_inv = labels_inv_restaurants

    top_2 = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
    return (labels_inv[top_2[0]],round(float(row[top_2[0]]),3)),(labels_inv[top_2[1]],round(float(row[top_2[1]]),3))


from collections import Counter,defaultdict
from operator import itemgetter
SENTENCE_LEN_THRESHOLD =30
class Restaurant(object):
    """
    """
    SENT_TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')

    
    def __init__(self, review_df):
        
        # Create the list of Reviews for this Restaurant
        try:
#             self.reviews = [self.sentence_tokenize(str(review_row['review_text'])) for _,review_row in review_df.iterrows()]
            review_df['review_sents'] = review_df['review_text'].apply(lambda row: self.sentence_tokenize(str(row)))
        except Exception as e:
            print (str(e))
            print (str(review_row['review_text']))
            
        self.reviews = review_df['review_sents'] 
        self.raw_df = review_df
            

    def __iter__(self):

        return self.reviews.__iter__()
    
    def sentence_tokenize(self, reviews_text):
        
        #Convert the raw text of a review to a list of sentence objects. 
        return [Sentence(sent) for sent in  Restaurant.SENT_TOKENIZER.tokenize(reviews_text)]
    

   
    
def encode_decode(row):
    if row is not None:
        return row.encode('ascii','ignore')
        # return row.decode('ascii','ignore')

def group_reviews(_list,test_reviews):
    j=0
    grouped_list =[]
    for item in test_reviews:
        i = len(item)
        grouped_list.append(_list[j:i+j])
        j = i+j
    return grouped_list

def predict_review_aspects(test_reviews,domain):
    
    test_reviews_list = lapply(test_reviews, lambda x: x.tolist())
    if domain=='hotels':
        predict_aspects_model= predict_aspects_model_hotels
        params = params_hotels
    if domain=='restaurants':
        predict_aspects_model = predict_aspects_model_restaurants
        params = params_restaurants

    test_reviews_list_list = sum(test_reviews_list,[])
    if len(test_reviews_list_list)==1:
        test_reviews_list_list = np.array(test_reviews_list_list[0]).reshape(1,params['max_sentence_length'])
    try:
        probs = predict_aspects_model.predict_proba(test_reviews_list_list,batch_size=5)

    except Exception as e:
        print(e)
        print (rev)
    # probs = [lapply(test_reviews, lambda x: predict_aspects_model.predict_proba(x,batch_size=5))]
    # probs=[]

    probs = group_reviews(probs,test_reviews)
    probs_1 = [[lapply(item, lambda x: round(x,3)) for item in rev] for rev in probs]
    probs_2 = [lapply(rev, lambda x: max_2(x,domain)) for rev in probs_1]


    # probs_slow = group_reviews(probs_slow,test_reviews)
    # probs_1_slow = [[lapply(item, lambda x: round(x,3)) for item in rev] for rev in probs_slow]
    # probs_2_slow = [lapply(rev, lambda x: max_2(x,domain)) for rev in probs_1_slow]


    n_sents = float(len(sum(probs_2,[])))
    
    print ('analysing %d review_sentences from %d reviews'%(n_sents,len(probs_2)))
    return probs_2


def get_aspects(input_test_data,domain):
    
    if type(input_test_data)==list:
        test_reviews =input_test_data
        data_df = pd.DataFrame(test_reviews)
    elif ".csv" in str(input_test_data):
        data_df = pd.read_csv(input_test_data)
        print ("Reading the %d from the csv file"%len(data_df))
    else:
        print("Input List of Reviews or Reviews in CSV format only")
        raise ValueError("Input List of Reviews or Reviews in CSV format only")

    columns = list(data_df.columns)
    columns[-1] = 'review_text'
    data_df.columns = columns

    # data_df['review_text'] = data_df['review_text'].apply(lambda row: encode_decode(row.lower())) # for python-2
    data_df['review_text'] = data_df['review_text'].apply(lambda row: row.lower()) # for python-3

    try:
#             self.reviews = [self.sentence_tokenize(str(review_row['review_text'])) for _,review_row in review_df.iterrows()]
        data_df['Descript'] = data_df['review_text'].apply(lambda row: Restaurant.SENT_TOKENIZER.tokenize(row))
    except Exception as e:
        print (str(e))
        pass

    print(data_df.head())
    if domain=='hotels':
        x,y,y_r, data_df = data_helper_keras_v2.test_csv(data_df,vocabulary_hotels,params_hotels,not_file=True)
    if domain=='restaurants':
        x,y,y_r, data_df = data_helper_keras_v2.test_csv(data_df,vocabulary_restaurants,params_restaurants,not_file=True)

    
    predicted_aspects = predict_review_aspects(x,domain=domain)
    data_df['predicted_aspects'] = predicted_aspects

    valid_aspect_reviews=[]
    for _,item in data_df.iterrows():
        valid_aspect_sents =[]
        for i,sent in enumerate(item['Descript']):
            print(sent,item['predicted_aspects'][i])
            valid_aspect_sents.append({'category':item['predicted_aspects'][i],
                                     'review_sentence':sent})
        valid_aspect_reviews.append(valid_aspect_sents)

    print(type(valid_aspect_reviews))
    return valid_aspect_reviews

if __name__ == "__main__":
    # print(clean_sentence("video screenshots be cuter than the actual picture 💀😂😭💯 #istg@gsg???? ."))
    print(get_aspects(["italian contemporary style – and the best lobby-lounge in town – conservatorium is infused into a 19th-century bank building, bang in between museumplein and amsterdam’s chicest fashion street."],domain='hotels'))
    # get_aspects('/home/common/siva/sruteesh/rest_reviews_test.csv')