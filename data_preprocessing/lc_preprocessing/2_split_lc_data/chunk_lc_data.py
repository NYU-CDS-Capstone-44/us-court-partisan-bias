import pandas as pd 

chunk_size = 1000000

csv_reader = pd.read_csv('/vast/amh9750/text-opinions-data-lower-court.csv', chunksize=chunk_size)

base_filename = '/vast/amh9750/lc_output_chunk'

# Iterate over chunks and export to CSV
for i, chunk in enumerate(csv_reader):
    # Generate a unique filename for each chunk
    filename = f"{base_filename}_{i + 1}.csv"

    # Export the chunk to CSV
    chunk.to_csv(filename, index=False)

print(f"Exported {i + 1} CSV files.")
