from textsum.summarize import Summarizer
import pandas as pd
import numpy as np
import torch

model_name = "pszemraj/long-t5-tglobal-base-16384-book-summary"
summarizer = Summarizer(
    model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
    token_batch_length=4096, # tokens to batch summarize at a time, up to 16384
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


# Load first round summarized df
df = pd.read_csv('/vast/mcn8851/processed-summarized-opinions-data-sc.csv') # Set file path to correct df

# Print some debugging information
print("Length of first round summarized DataFrame:", len(df))

def generate_summary(row):
    sequence = row['decision_text']
    result = summarizer.summarize_string(sequence)
    return result

batch_size = 1  # Adjust the batch size based on your GPU memory
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
    df.to_csv("/vast/mcn8851/summarized-opinions-data-sc.csv", index=False)

except Exception as e:
    # Handle unexpected errors
    print(f"An unexpected error occurred: {e}")

