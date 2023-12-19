import pandas as pd 

chunk_size = 500000

chunk_1 = pd.read_csv('/scratch/amh9750/capstone/final_lc_csv/lower_court_chunk_1.csv', chunksize=chunk_size)

base_filename = 'lower_court_chunk_1_chunk'

# Iterate over chunks and export to CSV
for i, chunk in enumerate(chunk_1):
    # Generate a unique filename for each chunk
    filename = f"{base_filename}_{i + 1}.csv"

    # Export the chunk to CSV
    chunk.to_csv(filename, index=False)

print(f"Exported {i + 1} CSV files.")
