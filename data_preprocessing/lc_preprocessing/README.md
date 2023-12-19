# Lower Court Preprocessing
## Preprocessing files are to be run in the following order:
1. `1_bulk_lc_data`
    * `data_lc.py`: Filters for lower court opinions, filters for `type` in correct types, filters for all opinions that happened after 1930, adds the imputed row for `judges` from clusters, adds the `date_filed` from clusters
    * `data_lc_text.py`: Converts all opinions to plain text 
2. `2_split_lc_data`
    * `chunk_lc_data.py`: Splits the lower court data into 5 chunks
3. `3_filter_split_lc_data`
    * `filter_chunk_1.py`: Same file but for all 5 chunks, removes opinions with null text, fixes encoding errors, filters for opinions with more than 35 tokens 
4. `4_filter_final_split_lc_data`
    * `1_remove_multiple_authors`: 1 file for each chunk that filters for only the opinions imputed with `judges` from clusters where `judges` only has one name populated
    * `2_split_chunks_again`: 1 file for each chunk that splits the first 4 large chunks into smaller chunks to facilitate inference
   
The resulting files are utilized in the inference pipeline.

`missing_data_analysis` includes some EDA on the opinions in the lower court that were missing authors, opinion text, or had less than 35 tokens
