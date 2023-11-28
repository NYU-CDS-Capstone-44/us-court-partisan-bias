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
tqdm.pandas()
   
import warnings as wrn
wrn.filterwarnings('ignore')

SEED = 7
torch.manual_seed(SEED)
torch.backends.cuda.deterministic = True
nltk.download('punkt')  # Download the Punkt tokenizer

# Max Sequence Length
MAX_SEQUENCE_LENGTH = 512

# Number of Nodes
NUMBER_OF_NODES = 10

# Number of LSTM layers
NUMBER_OF_LAYERS = 2

# Specify the file path where you want to save the logs
log_file_path = f'/home/amr10211/us-court-partisan-bias/RNN_LSTM_models/log_file_{MAX_SEQUENCE_LENGTH}trunc_{NUMBER_OF_NODES}nodes_{NUMBER_OF_LAYERS}layers.txt'

# Redirect stdout and stderr to the log file
sys.stdout = open(log_file_path, 'w')
sys.stderr = open(log_file_path, 'w')


# Get W2V index for token
def word2token(word):
    try:
        return w2v_model.wv.key_to_index[word]
    except KeyError:
        return 0


# Tokenize and Vectorize Data

def tokenize_vectorize(data, model, MAX_SEQUENCE_LENGTH):
    start = time.time()
    # Tokenize the documents
    tokenized_data =[word_tokenize(doc.lower()) for doc in data]
    token = time.time()
    print('tokenize: ', token-start)

    # Assuming 'model' is Word2Vec model
    vector_size = model.vector_size  # Get the size of the word vectors

    # Convert tokens to word vectors
    word_vectors = [[word2token(word) for word in doc] for doc in tokenized_data]

    vector = time.time()
    print('vectorize: ', vector-token)
    
    padded_word_vectors = pad_sequences(word_vectors, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=0.0, dtype = float)

    return padded_word_vectors

# Read data
train = pd.read_csv('/vast/amr10211/sc-train.csv')
val = pd.read_csv('/vast/amr10211/sc-val.csv')
test = pd.read_csv('/vast/amr10211/sc-test.csv')

# Load Word2Vec model
w2v_model = Word2Vec.load("/vast/amr10211/w2v_embedding_model.bin")

# Tokenize and Vectorize data
X_train = tokenize_vectorize(train['decision_text'], w2v_model, MAX_SEQUENCE_LENGTH)
X_val = tokenize_vectorize(val['decision_text'], w2v_model, MAX_SEQUENCE_LENGTH)
y_train = np.array(train['scdb_decision_direction'])
y_val = np.array(val['scdb_decision_direction'])
y_train = (y_train == 1.0).astype(int)
y_val = (y_val == 1.0).astype(int)

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
model.add(Bidirectional(LSTM(NUMBER_OF_NODES)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=250,
                    validation_data=(X_val, y_val), verbose=1)
model.save(f'/home/amr10211/us-court-partisan-bias/RNN_LSTM_models/RNN_LSTM_{MAX_SEQUENCE_LENGTH}trunc_{NUMBER_OF_NODES}nodes_{NUMBER_OF_LAYERS}layers.h5')
model_end = time.time()
print('model train:', model_end - model_start)
