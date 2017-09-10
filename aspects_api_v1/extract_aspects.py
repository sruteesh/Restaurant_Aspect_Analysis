
# coding: utf-8

# In[28]:


#Sentence Class - NLTK Sentence object
#Process the sentence(Tokenize, Lemmatize, and POS Tag)
from __future__ import division

import nltk
import numpy as np
import sys
import pandas as pd

from nltk.stem.wordnet import WordNetLemmatizer

from aspects_api_v1.process_data.tokenizers import MyPottsTokenizer
from aspects_api_v1.process_data.asp_extractors import SentenceAspectExtractor

class Sentence(object):
    """
    Class corresponding to a sentence in a review. Stores/manages word tokenization,
    part of speech (POS)tagging and lemmatization as well as some components
    of the final analysis such as aspect extraction. 
    """

    # Tokenizer for converting a raw string (sentence) to a list of strings (words)
    WORD_TOKENIZER = MyPottsTokenizer(preserve_case=False)

    # Lemmatizer
    LEMMATIZER = WordNetLemmatizer()

    # Featurizer
    # 	FEATURIZER = MetaFeaturizer([SubjFeaturizer(), LiuFeaturizer()]) #combine two featurizer objects

    # Aspect Extractor
    ASP_EXTRACTOR = SentenceAspectExtractor()

    def __init__(self, raw):
        """
        INPUT: string (raw text of sentence), (optional) Review object
        Stores raw sentence in attribute and performs/stores
        tokenization and POS tagging via class-variable tokenizer/tagger. 
        """

        self.raw = raw #string
        # self.tokenized = self.word_tokenize(raw) #list of strings
        print(type(self.raw))
        print(self.raw)

        self.tokenized = nltk.word_tokenize(self.raw) #list of strings
        self.pos_tagged = self.pos_tag(self.tokenized) #list of tuples
        self.lemmatized = self.lemmatize(self.pos_tagged) #list of tuples
        
        # compute and store aspects for this sentence
        self.aspects = self.compute_aspects()

    def word_tokenize(self, raw):

        return Sentence.WORD_TOKENIZER.tokenize(raw)

    def pos_tag(self, tokenized_sent):

        return nltk.pos_tag(tokenized_sent)

    def lemmatize(self, pos_tagged_sent):

        lemmatized_sent = []

        # Logic to use POS tag if possible
        for wrd, pos in pos_tagged_sent:
            try: 
                lemmatized_sent.append((Sentence.LEMMATIZER.lemmatize(wrd, pos), pos))
            except KeyError:
                lemmatized_sent.append((Sentence.LEMMATIZER.lemmatize(wrd), pos))

        return lemmatized_sent

    def compute_aspects(self):

        return Sentence.ASP_EXTRACTOR.get_sent_aspects(self)

    def has_aspect(self, asp_string):

        # re-tokenize the aspect
        asp_toks = asp_string.split()
#         print asp_toks
#         print self.tokenized
        # return true if all the aspect tokens are in this sentence 
        print(self.tokenized)
        print(asp_toks)
        print(all([tok in self.tokenized for tok in asp_toks]))
        return all([tok in self.tokenized for tok in asp_toks])

    def encode(self):

        return {'text': self.raw}

    def __str__(self):

        return self.raw


# In[29]:


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
    

    #Get Aspect Based Summary 
    def aspect_based_summary(self,extract_aspects=True,aspects=None):

        if extract_aspects:
            aspects = self.extract_aspects()
            asp_dict = dict([(aspect, self.aspect_summary(aspect)) for aspect in aspects])
            asp_dict = self.filter_asp_dict(asp_dict) # final filtering
        else:
            print(aspects)
            asp_dict = dict([(aspect, self.aspect_summary(aspect)) for aspect in aspects])
            print(asp_dict)
        return {'aspect_summary': asp_dict,}

    #Extract Aspects
    #Club Single aspects with Multi aspects if same [Ex: Ambience and Great Ambience --> Great Ambience] 
    #Take Only Relevant (Frequent Aspects) and discard others
    def extract_aspects(self, single_word_thresh=0.012, multi_word_thresh=0.003):


        # Get all the candidate aspects in each sentence
        
        asp_sents = [sent.aspects for rev in self.reviews for sent in rev]
        n_sents = float(len(asp_sents))
        
        print ('analysing %d review_sentences from %d reviews'%(n_sents,len(self.reviews)))
        
        single_asps = [] #list of lists (aspects)
        multi_asps = [] #list of lists
        # create single-word and multi-word aspect lists
        for sent in asp_sents: 
            for asp in sent:
                if len(asp) >0:
                    if 'and' in asp:
                        asp = ' '.join(asp)
                        for item in asp.split(' and'):
                            item = item.split()
                            if len(item) == 1:
                                single_asps.append(" ".join(item))
                            elif len(item) > 1:
                                multi_asps.append(" ".join(item))
                            else:
                                assert(False), "something wrong with aspect extraction" # shouldn't happen

                    else:    
                        if len(asp) == 1:
                            single_asps.append(" ".join(asp))
                        elif len(asp) > 1:
                            multi_asps.append(" ".join(asp))
                        else:
                            assert(False), "something wrong with aspect extraction" # shouldn't happen

        #Get sufficiently-common single- and multi-word aspects
        single_asps = [(asp, count) for asp, count in Counter(single_asps).most_common(30) if (count/n_sents) > single_word_thresh]
        multi_asps = [(asp, count) for asp, count in Counter(multi_asps).most_common(30) if (count/n_sents) > multi_word_thresh]
        
        #filter redundant single-word aspects
        single_asps = self.filter_single_asps(single_asps, multi_asps)
        # the full aspect list, sorted by frequency
        all_asps =  [asp for asp,_ in sorted(single_asps + multi_asps, key=itemgetter(1))]

        return all_asps

    #For the aspect get all the sentences which have this aspect and aslo their meta-data(rating,id,timestamp)
    def aspect_summary(self, aspect):
        master_sents=[]
        aspect_sents = self.get_sents_by_aspect(aspect)

        for row,sent in aspect_sents:
            sents={}
#             if len(sent.tokenized) > SENTENCE_LEN_THRESHOLD:
#                 continue #filter really long sentences
#             sents['rating'] = row['rating']
#             sents['review_id'] = row['id']
#             sents['timestamp'] = row['timestamp']
            sents['aspect'] = aspect
            sents['review_sentence'] = sent.raw
            master_sents.append(sents)
        return master_sents

    #Get sentences which have a particular aspect
    def get_sents_by_aspect(self, aspect):
        
        
        return [(review_row,sent) for _,review_row in self.raw_df.iterrows() for sent in review_row['review_sents'] if sent.has_aspect(aspect)] 
    #Club Single aspects with Multi aspects if same [Ex: Ambience and Great Ambience --> Great Ambience] 
    
    def filter_single_asps(self, single_asps, multi_asps):

        return [(sing_asp, count) for sing_asp,count in single_asps if not any([sing_asp in mult_asp for mult_asp,_ in multi_asps])]

    def filter_asp_dict(self, asp_dict, num_valid_threshold = 2):

        return dict([(k, v) for k,v in asp_dict.iteritems() if len(asp_dict[k]) > num_valid_threshold])
    
    
    


# In[30]:


entity_dict = {'service':['staff','room service','people','front desk','hotel staff','service'],
              'location':['street','location','place','area','view'],
              'food':['breakfast','food'],
              'cost':['price','cost'],
              'hotel':['restuarant','hotel'],
              'rooms':['lobby','pool','door','floor','bathroom'],
              'ambience':['ambience','decor']}


# In[31]:


entity_dict_inv={}
for asp in sum(entity_dict.values(),[]):
    entity_dict_inv.update({asp:k for k,v in entity_dict.items() if asp in v})


# In[32]:


def encode_decode(row):
    if row is not None:
        return row.encode('ascii','ignore') 


# In[46]:


def get_aspects(test_reviews):
    
    data_df = pd.DataFrame(test_reviews)

    columns = list(data_df.columns)
    columns[-1] = 'review_text'
    data_df.columns = columns

    data_df['review_text'] = data_df['review_text'].apply(lambda row: row.lower())

    #Call the Restaurant Object to get Aspects for the reviews
    restaurant_summary = Restaurant(data_df)
    aspect_reviews = sum(restaurant_summary.aspect_based_summary(extract_aspects=False,aspects=entity_dict_inv.keys())['aspect_summary'].values(),[])

    valid_aspect_reviews =[]

    if len(aspect_reviews)==0:
       valid_aspect_reviews.append({'aspect':None,'category':None,'review_sentence':None})
    else:
        for item in aspect_reviews:
            if item['aspect'] in entity_dict_inv:
                valid_aspect_reviews.append({'aspect':item['aspect'],
                                             'category':entity_dict_inv[item['aspect']],
                                             'review_sentence':item['review_sentence']})
    return valid_aspect_reviews


   