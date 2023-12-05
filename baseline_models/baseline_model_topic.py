import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
import pickle
#from keras.utils import to_categorical

SEED = 7

# Load Word2Vec model
w2v_model = Word2Vec.load("w2v_embedding_model.bin")

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
train = pd.read_csv('/vast/amr10211/sc-train-topic.csv')
val = pd.read_csv('/vast/amr10211/sc-val-topic.csv')
test = pd.read_csv('/vast/amr10211/sc-test-topic.csv')

# Extract text column as a Series
X_train = train['decision_text']
X_val = val['decision_text']
X_test = test['decision_text']

# Extract label column as a Series
y_train = train['issue_area']
y_val = val['issue_area']
y_test = test['issue_area']

# Concatenate train and val data for cross validation in GridSearchCV
X = pd.concat([X_train, X_val], axis=0)
y = pd.concat([y_train, y_val], axis=0)

# Vectorize documents
X = tokenize_vectorize_mean(X, w2v_model)
X = np.array(X)

# Subtract 1 from the labels to start from 0
y_train = y_train - 1
y_val = y_val - 1

#Converting labels categorical 
#y_train = to_categorical(y_train, num_classes=14)
#y_val = to_categorical(y_val, num_classes=14)


print('finished tokenizing and vectorizing')

# Initialize model
model = GradientBoostingClassifier(random_state=7)

# Define parameter grid to search over 
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of boosting stages (trees)
    'max_depth': [3, 6, 9],  # Maximum depth of individual trees
}

print('starting grid search')

# Initialize and run GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=10)
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

model_filename = "baseline_model_topic.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

# Write best hyperparameters out to a text file
output_file = "baseline_model_topic.txt"

with open(output_file, 'w') as file:
    file.write("Best Hyperparameters:\n")
    for param, value in best_params.items():
        file.write(f"{param}: {value}\n")
    file.write(f"\nBest Accuracy: {best_accuracy:.4f}\n")


