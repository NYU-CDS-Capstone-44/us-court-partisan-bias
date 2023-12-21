import pandas as pd
import numpy as np
np.object = object
np.int = int
np.float = float
np.bool = bool



import torch
import torchtext
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
#import spacy
import os
import nltk
import sys
from nltk import word_tokenize
from tqdm import tqdm
#from tqdm.notebook import tqdm
from IPython.display import display
#from transformers import AutoTokenizer
#import gensim
from gensim.models import Word2Vec
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
import keras_preprocessing
from keras_preprocessing.sequence import pad_sequences
import torch
from keras.utils import to_categorical

tqdm.pandas()

import warnings as wrn
wrn.filterwarnings('ignore')

SEED = 7
torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True
nltk.download('punkt')  # Download the Punkt tokenizer

# Max Sequence Length
MAX_SEQUENCE_LENGTH = 4073

# Number of Nodes
NUMBER_OF_NODES = 50

# Number of LSTM layers
NUMBER_OF_LAYERS = 4

#Dropout
DROPOUT_AMOUNT = 0.05

# Specify the file path where you want to save the logs
log_file_path = f'/scratch/amr10211/capstone/topics/model_outputs/log_file_{MAX_SEQUENCE_LENGTH}trunc_{NUMBER_OF_NODES}nodes_{NUMBER_OF_LAYERS}layers_names_removed_summary_{DROPOUT_AMOUNT}drop_removed_numbers_TOPIC.txt'

# Redirect stdout and stderr to the log file
sys.stdout = open(log_file_path, 'w')
sys.stderr = open(log_file_path, 'w')

# Remove Justice Names from text
def remove_justice_names_and_numbers(text):
    '''Removing Justice names and Mr. & Ms.'''

   # Split the text into words
    words = text.split()

    # Find the index of 'Justice'    
    justice_index = [i for i, word in enumerate(words) if word == 'Justice']
    if justice_index:
        justice_index = justice_index[0]
    else:
        justice_index = -1

    # Remove numeric values
    words = [word for word in words if not word.isdigit()]

    # Additional words to remove
    words_to_remove = ['mr.', 'ms.']

    # Remove the word after 'Justice'
    if justice_index != -1 and justice_index < len(words) - 1:
        del words[justice_index + 1]

    # Convert specified words to lowercase
    for i in range(len(words)):
        if words[i].lower() in words_to_remove:
            words[i] = ''

    # Join the words back into a string
    return ' '.join(words)


# Get W2V index for token
def word2token(word):
    try:
        return w2v_model.wv.key_to_index[word]
    except KeyError:
        return 0


# Tokenize and Vectorize Data
def tokenize_vectorize(data, y, model, MAX_SEQUENCE_LENGTH, min_tokens=35):
    start = time.time()
    # Tokenize the documents
    tokenized_data =[word_tokenize(doc.lower()) for doc in data]
    token = time.time()
    print('tokenize: ', token - start)
    print('Number of docs before filter:', len(tokenized_data))

    # Remove samples with less than min_tokens
    filtered_data = [(doc, label) for doc, label in zip(tokenized_data, y) if len(doc) >= min_tokens]
    tokenized_data, filtered_labels = zip(*filtered_data)

    filter_time = time.time()
    print('filter: ', filter_time - token)
    print('Number of docs after filter:', len(tokenized_data))

    # Assuming 'model' is Word2Vec model
    vector_size = model.vector_size  # Get the size of the word vectors

    # Convert tokens to word vectors
    word_vectors = [[word2token(word) for word in doc] for doc in tokenized_data]

    vector = time.time()
    print('vectorize: ', vector-token)

    padded_word_vectors = pad_sequences(word_vectors, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=0.0, dtype = float)

    return padded_word_vectors, np.array(filtered_labels)


# Read data
train = pd.read_csv('/home/amr10211/us-court-partisan-bias/summarized-opinions-data-sc-topic.csv')
#train = pd.read_csv('/vast/amr10211/sc-train.csv')
val = pd.read_csv('/vast/amr10211/sc-val-topic.csv')
test = pd.read_csv('/vast/amr10211/sc-test-topic.csv')

# Extract cluster IDs from val and test DataFrames
val_cluster_ids = val['cluster_id'].unique()
test_cluster_ids = test['cluster_id'].unique()

# Filter rows in train DataFrame based on cluster IDs to prevent leakage
train = train[~train['cluster_id'].isin(val_cluster_ids) & ~train['cluster_id'].isin(test_cluster_ids)]

#Remove Justice names and numbers from train
train['decision_text'] = train['decision_text'].apply(remove_justice_names_and_numbers)

# Load Word2Vec model
w2v_model = Word2Vec.load("/vast/amr10211/w2v_embedding_model.bin")

# Tokenize and Vectorize data
X_train, y_train = tokenize_vectorize(train['decision_text'], train['issue_area'], w2v_model, MAX_SEQUENCE_LENGTH)
X_val, y_val = tokenize_vectorize(val['decision_text'], val['issue_area'], w2v_model, MAX_SEQUENCE_LENGTH)

# Subtract 1 from the labels to start from 0
y_train = y_train - 1
y_val = y_val - 1

#Converting labels categorical 
y_train = to_categorical(y_train, num_classes=14)
y_val = to_categorical(y_val, num_classes=14)

# Set w2v weights, vocab size, and embedding size 
w2v_weights = w2v_model.wv.vectors
vocab_size, embedding_size = w2v_weights.shape
print("Vocabulary Size: {} - Embedding Dim: {}".format(vocab_size, embedding_size))

# Check if GPU is available
if tf.test.gpu_device_name():
    print('GPU found')
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found, running on CPU")
model_start = time.time()

# Keras Embedding layer with Word2Vec weights initialization
model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[w2v_weights],
                    input_length=MAX_SEQUENCE_LENGTH,
                    mask_zero=True,
                    trainable=False))

model.add(Bidirectional(LSTM(NUMBER_OF_NODES, return_sequences=True)))
model.add(Dropout(DROPOUT_AMOUNT))
model.add(Bidirectional(LSTM(NUMBER_OF_NODES, return_sequences=True)))
model.add(Dropout(DROPOUT_AMOUNT))
model.add(Bidirectional(LSTM(NUMBER_OF_NODES, return_sequences=True)))
model.add(Dropout(DROPOUT_AMOUNT))
model.add(Bidirectional(LSTM(NUMBER_OF_NODES)))
model.add(Dropout(DROPOUT_AMOUNT))
model.add(Dense(14, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=250,
                    validation_data=(X_val, y_val), verbose=1)
model.save(f'/scratch/amr10211/capstone/topics/RNN_LSTM_{MAX_SEQUENCE_LENGTH}trunc_{NUMBER_OF_NODES}nodes_{NUMBER_OF_LAYERS}layers_names_removed_summary_{DROPOUT_AMOUNT}drop_removed_numbers_TOPIC.h5')
model_end = time.time()

print('model train:', model_end - model_start)
