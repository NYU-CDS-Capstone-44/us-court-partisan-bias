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
from keras.utils import to_categorical

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
rnn_lstm_model_path = 'RNN_LSTM_4073trunc_50nodes_4layers_names_removed_summary_0.05drop_removed_numbers_TOPIC.h5'
model = load_model(rnn_lstm_model_path)

# Load the Word2Vec model
w2v_model = Word2Vec.load("/vast/amr10211/w2v_embedding_model.bin")

# Load the test set
test = pd.read_csv('/vast/amr10211/sc-test-topic.csv')

# Tokenize and vectorize data
ID, X_test, y_test = tokenize_vectorize(test['id'], test['decision_text'], test['issue_area'], w2v_model, MAX_SEQUENCE_LENGTH)

# Subtract 1 from the labels to start from 0
y_test = y_test - 1

# Create an empty DataFrame for test results
test_results_df = pd.DataFrame(index=range(len(X_test)))

# Make predictions
start = time.time()
predicted_probs = model.predict(X_test, verbose=True)
test_results_df['id'] = ID
for i in range(14):
    test_results_df[f'predicted_probability_class_{i + 1}_rnn_lstm_topic'] = predicted_probs[:, i]
test_results_df['predicted_label_rnn_lstm_topic'] = np.argmax(predicted_probs, axis=1)
test_results_df['actual_label_topic'] = y_test
end = time.time()
print('make predictions: ', end - start)
test_results_df.to_csv('RNN_LSTM_test_results_TOPIC.csv', index=False)

# Calculate overall accuracy
accuracy = np.mean(test_results_df['predicted_label_rnn_lstm_topic'] == test_results_df['actual_label_topic'])
print(f'Overall Accuracy (RNN LSTM): {accuracy * 100:.2f}%')


