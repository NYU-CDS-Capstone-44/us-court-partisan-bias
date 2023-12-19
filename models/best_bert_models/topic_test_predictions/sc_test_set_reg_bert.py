from datasets import Dataset
import datasets
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from huggingface_hub import notebook_login
from transformers import TrainingArguments, Trainer
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras.utils import to_categorical

seed = 7
torch.manual_seed(seed)
np.random.seed(seed)

# Make sure to login to huggingface on the terminal 

# Specify hyperparameter version
version = '32batch_3epoch_5e5lr_01wd'

# Load data
test = pd.read_csv('/vast/amh9750/sc-test-topic.csv')
test = test.drop(test.columns[0], axis=1)
issue_area = test['issue_area'] - 1
test['labels'] = issue_area.astype(int)

def tokenize_function(example):
    tokenized_example = tokenizer(
        example["decision_text"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
    )
    return tokenized_example

# Check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device={device}")

# Load the model back in from huggingface
finetune_model_name = f'annabellehuether/topic-bert-base-uncased-supreme-court-{version}'

finetune_model = AutoModelForSequenceClassification.from_pretrained(finetune_model_name)
finetune_model.to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Make the test set a Dataset
test_dataset = Dataset.from_pandas(test)

# Tokenize the test data
tokenized_test = test_dataset.map(
    tokenize_function,
    remove_columns=["id", "cluster_id", "type", "decision_text","date_filed", "issue_area"]
)

tokenized_test.set_format("torch")

# Set up a trainer object to predict on test set
predictions_trainer = Trainer(
    model=finetune_model,
    args=TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=32,
        seed = 7
    ),
)

results = predictions_trainer.predict(tokenized_test)

predictions = np.argmax(results.predictions, axis=1)

true_labels = tokenized_test["labels"]

precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')
accuracy = accuracy_score(true_labels, predictions)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

results.metrics

file_name = f'topic_classification_bert_metrics_{version}.txt'

with open(file_name, 'w') as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Accuracy: {accuracy}")


predicted_labels = torch.argmax(torch.tensor(results.predictions), dim=1).tolist()
probabilities = torch.nn.functional.softmax(torch.tensor(results.predictions), dim=1).tolist()

test_results = test
test_results = test_results.drop(columns=['decision_text'])

# Add predicted labels and probabilities to the test DataFrame
test_results['predicted_label'] = predicted_labels

for i in range(14):
    test_results[f'probability_class_{i}'] = [prob[i] for prob in probabilities]

test_results.to_csv(f'topic_classification_bert_test_results_{version}.csv')


