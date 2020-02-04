import re
import json
import nltk
import numpy as np
import pandas as pd
import en_core_web_sm
from pprint import pprint
from collections import Counter
from copy import deepcopy as dc
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import config as cfg
from encoders.encoder import Encoder


pd.set_option('display.max_rows', None)

class TFIDFEncoder(Encoder):
    def __init__(self):
        pass
        
    def generate_n_grams(self, doc, n_grams):
        doc = doc.lower()    
        doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc)

        tokens = [token for token in doc.split(" ") if token != ""]

        ngrams = zip(*[tokens[i:] for i in range(n_grams)])
        return [" ".join(ngram) for ngram in ngrams]
    
    def create_bag_of_words_as_list(self, docs, n_grams=1):
        bag_of_words = set()
        doc_word_counter_dict = {}

        for doc_id, doc in enumerate(docs):
            if n_grams == 1:
                split_doc = doc.split(' ')
                doc_word_counter_dict[doc_id] = Counter(split_doc)
                bag_of_words.update(split_doc)
            else:
                n_gram_phrases = self.generate_n_grams(doc, n_grams=n_grams)
                doc_word_counter_dict[doc_id] = Counter(n_gram_phrases)
                bag_of_words.update(n_gram_phrases)

        stopwords = sw.words('english')
        bag_of_words -= set(stopwords)

        clean_bag_of_words = [word for word in bag_of_words if len(word) > 2]

        return list(clean_bag_of_words), doc_word_counter_dict
    
    def create_doc_word_matrix(self, docs, bag_of_words_dict, doc_word_ctr_dict):
        global_doc_word_matrix = {doc_idx: [0] * len(bag_of_words_dict) for doc_idx in range(len(docs))}

        for doc_idx in range(len(docs)):
            for word_idx, word in bag_of_words_dict.items():
                global_doc_word_matrix[doc_idx][word_idx] += doc_word_ctr_dict[doc_idx][word]

        return global_doc_word_matrix
    
    def create_term_frequency_matrix(self, base_matrix):
        tf_matrix = dc(base_matrix)

        for doc_id in tf_matrix:
            num_words_sum = sum(tf_matrix[doc_id])
            for word_idx, _ in enumerate(tf_matrix[doc_id]):
                tf_matrix[doc_id][word_idx] /= num_words_sum

        return tf_matrix
    
    def create_idf_matrix(self, base_matrix):
        idf_matrix = dc(base_matrix)

        for doc_id in idf_matrix:
            for word_idx, _ in enumerate(idf_matrix[doc_id]):
                docs_with_word = sum([1 for i in range(len(idf_matrix)) if idf_matrix[i][word_idx] > 0])
                idf_matrix[doc_id][word_idx] = np.log(len(idf_matrix) / docs_with_word)

        return idf_matrix
    
    def encode_multiple(self, data):
        bag_of_words_as_list, doc_word_ctr_dict = self.create_bag_of_words_as_list(data, n_grams=cfg.N_GRAMS)
        bag_of_words_enumerated_dict = dict(enumerate(bag_of_words_as_list))
        
        matrix = self.create_doc_word_matrix(data, bag_of_words_enumerated_dict, doc_word_ctr_dict)
        tf_matrix = self.create_term_frequency_matrix(matrix)
        idf_matrix = self.create_idf_matrix(matrix)
        
        tfidf_matrix = np.multiply(np.array(list(tf_matrix.values())), np.array(list(idf_matrix.values()))).tolist()

        return tfidf_matrix
