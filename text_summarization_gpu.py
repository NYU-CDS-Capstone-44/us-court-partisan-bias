from textsum.summarize import Summarizer
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

# How many GPUs are there?
print(torch.cuda.device_count())

# Is PyTorch using a GPU?
print(torch.cuda.is_available())


summarizer = pipeline(
    "summarization",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
    device=0 if torch.cuda.is_available() else -1,
)

# Load all data
train = pd.read_csv('/vast/mcn8851/sc-train.csv')
val = pd.read_csv('/vast/mcn8851/sc-val.csv')
test = pd.read_csv('/vast/mcn8851/sc-test.csv')

# Concat into single df
df = pd.concat([train, val, test], ignore_index=True)

#Test subset
#df = df.iloc[:5,:]

def generate_summary(row):
    sequence = row['decision_text']
    result = summarizer(sequence)
    return result[0]["summary_text"]


# Replace 'decision_text' with 'decision_text_summary'
df['decision_text_summary'] = df.apply(generate_summary, axis=1)
df_summary = df.drop(columns=['decision_text'])

# Write to csv
df_summary.to_csv('/vast/mcn8851/gpu-summary-text-opinions-data-sc.csv', index=False) #Update File Path
