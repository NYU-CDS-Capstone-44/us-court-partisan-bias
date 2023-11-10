
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv('/vast/amr10211/sc-train.csv')
val = pd.read_csv('/vast/amr10211/sc-val.csv')
test = pd.read_csv('/vast/amr10211/sc-test.csv')

X_train = train['decision_text']
X_val = val['decision_text']
X_test = test['decision_text']

y_train = train['scdb_decision_direction']
y_val = val['scdb_decision_direction']
y_test = test['scdb_decision_direction']

X = pd.concat([X_train, X_val], axis=0)
y = pd.concat([y_train, y_val], axis=0)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_test = vectorizer.transform(X_test)

model = GradientBoostingClassifier(random_state=7)

param_grid = {
    'n_estimators': [100, 200, 300, 400],  # Number of boosting stages (trees)
    'max_depth': [3, 4, 5, 6],  # Maximum depth of individual trees
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=10)
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

output_file = "baseline_model.txt"

with open(output_file, 'w') as file:
    file.write("Best Hyperparameters:\n")
    for param, value in best_params.items():
        file.write(f"{param}: {value}\n")
    file.write(f"\nBest Accuracy: {best_accuracy:.4f}\n")

print(f"Best hyperparameters and accuracy written to {output_file}")
