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

# Remove Justice Names from text
def remove_justice_names(text):
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
    #words = [word for word in words if not word.isdigit()]

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


# Load data
val = pd.read_csv('/vast/amh9750/sc-val-topic.csv')
val = val.drop(val.columns[0], axis=1)
issue_area = val['issue_area'] - 1
val['labels'] = issue_area.astype(int)

test = pd.read_csv('/vast/amh9750/sc-test-topic.csv')
test = test.drop(test.columns[0], axis=1)
issue_area = test['issue_area'] - 1
test['labels'] = issue_area.astype(int)

train = pd.read_csv('/scratch/amh9750/us-court-partisan-bias/summarized-opinions-data-sc-topic.csv')
train = train.drop(train.columns[0], axis=1)

# Extract cluster IDs from val and test DataFrames
val_cluster_ids = val['cluster_id'].unique()
test_cluster_ids = test['cluster_id'].unique()

# Filter rows in train DataFrame based on cluster IDs to prevent leakage
train = train[~train['cluster_id'].isin(val_cluster_ids) & ~train['cluster_id'].isin(test_cluster_ids)]

#Remove Justice names from train
train['decision_text'] = train['decision_text'].apply(remove_justice_names)

issue_area = train['issue_area'] - 1
train['labels'] = issue_area.astype(int)

# Convert val and train to Datasets
val_dataset = Dataset.from_pandas(val)
train_dataset = Dataset.from_pandas(train)

# Combine into one Dataset dictionary
dd = datasets.DatasetDict({"train":train_dataset,"validation":val_dataset})

# Load in the autotokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    tokenized_example = tokenizer(
        example["decision_text"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
    )
    return tokenized_example

# Tokenize the examples
tokenized_datasets = dd.map(
    tokenize_function,
    remove_columns=["id", "cluster_id", "type", 'decision_text','date_filed', "issue_area"]
)

# Extract correct datasets for fine tuning
tokenized_datasets.set_format("torch")
train_data = tokenized_datasets["train"]
eval_data = tokenized_datasets["validation"]

# Load in model
model_str = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_str, num_labels=14)

# Check for GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device={device}")
model.to(device)

# Define training arguments
model_name = f"topic-bert-base-uncased-supreme-court-{version}"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=3,
                                  learning_rate=5e-5,
                                  weight_decay=0.01,
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=32,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  push_to_hub=True,
                                  log_level="error",
                                  seed=7
                                  )

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define the trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    compute_metrics=compute_metrics,
)

# Fine tune the model
trainer.train()

# Push the model to huggingface
trainer.push_to_hub(commit_message="Training completed!")

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
    remove_columns=["id", "cluster_id", "type", "decision_text","date_filed", "scdb_decision_direction"]
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

# Get predictions on the test set
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

file_name = f'topic_bert_metrics_{version}.txt'

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
test_results['probability'] = [prob[label] for prob, label in zip(probabilities, predicted_labels)]

test_results.to_csv(f'topic_bert_test_results_{version}.csv')
