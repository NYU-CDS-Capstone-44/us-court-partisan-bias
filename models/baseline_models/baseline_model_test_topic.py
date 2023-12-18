import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
import pickle
from sklearn.metrics import accuracy_score


# Load Word2Vec model
w2v_model = Word2Vec.load("/home/amr10211/us-court-partisan-bias/w2v_embedding_model.bin")

def tokenize_vectorize_mean(data, model):
    """
    Tokenizes and converts the words in each document into their Word2Vec embeddings
    Takes the mean across word embeddings to give a 2D array rather than a 3D array
    param: data (pandas Series)
    param: model (Word2Vec model)
    return: word_vectors (list of arrays)
    """
    # Tokenize the documents
    tokenized_data =[word_tokenize(doc.lower()) for doc in data]

    # Convert tokens to word vectors and take the mean across words to get document vector representations
    word_vectors = [np.mean([model.wv[token] for token in doc if token in model.wv], axis=0) for doc in tokenized_data]

    return word_vectors

# Load data
test = pd.read_csv('/vast/amr10211/sc-test-topic.csv')

# Extract text column as a Series
X_test = test['decision_text']

# Extract label column as a Series
y_test = test['issue_area']

X = X_test
y = y_test

# Vectorize documents
X = tokenize_vectorize_mean(X, w2v_model)
X = np.array(X)

# Change labels to 0-14
y = y - 1

print('finished tokenizing and vectorizing')

# Load the trained model
with open('baseline_model_topic.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions on the test set
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Save predictions and probabilities to a CSV file
class_labels = [f'Probability_Class_{i}' for i in range(model.classes_.shape[0])]
probabilities_df = pd.DataFrame(probabilities, columns=class_labels)
test_with_predictions_and_probabilities = pd.concat([test, pd.Series(predictions, name='Predictions'), probabilities_df], axis=1)
test_with_predictions_and_probabilities.to_csv('baseline_test_topic_predictions.csv', index=False)


# Calculate and save test accuracy to a text file
test_accuracy = accuracy_score(y, predictions)
with open('baseline_test_topic_accuracy.txt', 'w') as file:
    file.write(f'Test Accuracy: {test_accuracy:.4f}')

print('Inference and evaluation completed.')

