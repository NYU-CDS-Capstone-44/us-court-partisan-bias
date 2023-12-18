from textsum.summarize import Summarizer
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

# How many GPUs are there?
print('Num GPUs requested: ', torch.cuda.device_count())

# Is PyTorch using a GPU?
print('PyTorch using a GPU?: ', torch.cuda.is_available())


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

# Print some debugging information
print("Length of original DataFrame:", len(df))

#Test subset
#df = df.iloc[:50,:]

def generate_summary(row):
    sequence = row['decision_text']
    result = summarizer(sequence)
    return result[0]["summary_text"]


batch_size = 5  # Adjust the batch size based on your GPU memory
num_batches = len(df) // batch_size

i = 0

try:
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_df = df.iloc[start_idx:end_idx, :].copy()

        try:
            # Conditionally apply the summarizer based on the length of 'decision_text'
            mask = batch_df['decision_text'].apply(lambda x: len(x.split()) > 512)
            batch_df.loc[mask, 'decision_text'] = batch_df[mask].apply(generate_summary, axis=1)

            # Update the original DataFrame with the modified batch DataFrame
            df.iloc[start_idx:end_idx, :] = batch_df

        except Exception as e:
            # Error handling for batch processing
            print(f"Error processing batch {i} (indices {start_idx}-{end_idx}): {e}")

        # Clear batch-specific GPU memory
        torch.cuda.empty_cache()

    # Save the DataFrame after processing all batches (outside the loop)
    print("Length of final DataFrame with summaries:", len(df))
    df.to_csv("/vast/mcn8851/processed-summarized-opinions-data-sc.csv", index=False)

except Exception as e:
    # Handle unexpected errors
    print(f"An unexpected error occurred: {e}")

