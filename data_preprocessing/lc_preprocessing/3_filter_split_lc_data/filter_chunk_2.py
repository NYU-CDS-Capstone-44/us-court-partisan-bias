import pandas as pd
import numpy as np
import csv
from gensim.models import Word2Vec
import nltk
from nltk import word_tokenize
import torch
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import sys
import time
import re

import warnings as wrn
wrn.filterwarnings('ignore')

SEED = 7
torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True
csv.field_size_limit(sys.maxsize)

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

# Load in the dataset and drop nulls
df = pd.read_csv('/vast/amh9750/lc_output_chunk_2.csv') 

df_null_decision_text = df[df['decision_text'].isnull()]
df_null_decision_text.to_csv('lc_null_decision_text_chunk_2.csv', index=False)

df.dropna(subset=['decision_text'], inplace=True)

def clean_encoding_errors(text):
    # Replace known encoding errors
    text = re.sub(r'Ã\x82Â', "", text)
    text = re.sub(r'Ã¢Â\x80Â\x98', "‘", text)
    text = re.sub(r'Ã¢Â\x80Â\x99', "’", text)
    text = re.sub(r'Ã¢Â\x80Â\x9c', "“", text)
    text = re.sub(r'Ã¢Â\x80Â\x9d', "”", text)
    text = re.sub(r'Ã¢Â\x80Â\x94', "—", text)
    text = re.sub(r'Ã¢Â\x80Â¦', "…", text)

    return text

df['decision_text'] = df['decision_text'].apply(clean_encoding_errors)

# Load in the w2v model 
w2v_model = Word2Vec.load("/scratch/amh9750/us-court-partisan-bias/w2v_embedding_model.bin")

# Define function for getting indices
def word2token(word):
    """
    Returns the index of the word in the Word2Vec embeddings model defined globally
    param: word (str)
    return: index (int)
    """
    try:
        return w2v_model.wv.key_to_index[word]
    except KeyError:
        return 0

def tokenize_vectorize(data):
    """
    Tokenizes a single document and turns the document into a vector of indices, adding padding/truncating where necessary 
    param: data (pandas series of size (num_opinions, 1))
    param: max_sequence_length (int)
    return: padded_word_vectors (np array of size (num_opinions, num_words))
    """
    tokenized_data = word_tokenize(data.lower())
    token_count = len(tokenized_data)
    #word_vectors = [word2token(word) for word in tokenized_data]
    #padded_word_vectors = pad_sequences([word_vectors], maxlen=512, padding='post', value=0.0, dtype = float)
    #return padded_word_vectors
    return token_count


# Tokenize, vectorize, and pad data
start = time.time()
text = df['decision_text']
X = text.parallel_apply(tokenize_vectorize)
end = time.time()
df['token_count'] = X
print('tokenize and get length: ', end-start)

df_large = df[df['token_count'] >= 35]
df_small = df[df['token_count'] < 35]

df_large.to_csv('lc_greater_than_35_chunk_2.csv', index=False)
df_small.to_csv('lc_less_than_35_chunk_2.csv', index=False)
