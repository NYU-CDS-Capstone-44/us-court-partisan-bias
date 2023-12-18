import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from tqdm import tqdm
import time 
import nltk
from nltk import word_tokenize

# Get W2V index for token
def word2token(word):
    try:
        return w2v_model.wv.key_to_index[word]
    except KeyError:
        return 0


# Tokenize and Vectorize Data
def tokenize_vectorize(ID, data, y, model, MAX_SEQUENCE_LENGTH, min_tokens=35):
    start = time.time()
    # Tokenize the documents
    tokenized_data =[word_tokenize(doc.lower()) for doc in data]
    token = time.time()
    print('tokenize: ', token - start)
    print('Number of docs before filter:', len(tokenized_data))

    # Remove samples with less than min_tokens
    filtered_data = [(id_val, doc, label) for id_val, doc, label in zip(ID, tokenized_data, y) if len(doc) >= min_tokens]
    ID, tokenized_data, filtered_labels = zip(*filtered_data)

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

    return ID, padded_word_vectors, np.array(filtered_labels)


# Max Sequence Length
MAX_SEQUENCE_LENGTH = 4073

# Load the trained RNN LSTM model
rnn_lstm_model_path = '/best_RNN_LSTM_model/RNN_LSTM_4073trunc_100nodes_2layers_names_removed_summary_0.05drop15epochs.h5'
model = load_model(rnn_lstm_model_path)

# Load the Word2Vec model
w2v_model = Word2Vec.load("/vast/amr10211/w2v_embedding_model.bin")

# Load the test set
test = pd.read_csv('/vast/amr10211/sc-test.csv')

# Tokenize and vectorize data
ID, X_test, y_test = tokenize_vectorize(test['id'], test['decision_text'], test['scdb_decision_direction'], w2v_model, MAX_SEQUENCE_LENGTH)
y_test = (y_test == 1.0).astype(int)

# Create an empty DataFrame for test results
test_results_df = pd.DataFrame(index=range(len(X_test)))

# Make predictions
start = time.time()
predicted_probs = model.predict(X_test, verbose=True)
test_results_df['id'] = ID
test_results_df['predicted_probability_rnn_lstm'] = predicted_probs
predicted_class = np.where(predicted_probs > 0.5, 1, 0)
test_results_df['predicted_label_rnn_lstm'] = predicted_class
test_results_df['actual_label'] = y_test
end = time.time()
print('make predictions: ', end - start)
test_results_df.to_csv('/best_RNN_LSTM_model/RNN_LSTM_test_results_15epochs.csv', index=False)

# Calculate overall accuracy
accuracy = np.mean(test_results_df['predicted_label_rnn_lstm'] == test_results_df['actual_label'])
print(f'Overall Accuracy (RNN LSTM): {accuracy * 100:.2f}%')


