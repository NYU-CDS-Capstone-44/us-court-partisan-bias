import pandas as pd 

chunk_1_text = pd.read_csv('/scratch/amh9750/capstone/chunk_filter_preprocess/lc_greater_than_35_chunk_1.csv')
chunk_2_text = pd.read_csv('/scratch/amh9750/capstone/chunk_filter_preprocess/lc_greater_than_35_chunk_2.csv')
chunk_3_text = pd.read_csv('/scratch/amh9750/capstone/chunk_filter_preprocess/lc_greater_than_35_chunk_3.csv')
chunk_4_text = pd.read_csv('/scratch/amh9750/capstone/chunk_filter_preprocess/lc_greater_than_35_chunk_4.csv')
chunk_5_text = pd.read_csv('/scratch/amh9750/capstone/chunk_filter_preprocess/lc_greater_than_35_chunk_5.csv')

text = pd.concat([chunk_1_text, chunk_2_text, chunk_3_text, chunk_4_text, chunk_5_text], ignore_index=True)

authors = text[text['author_str'].notnull()| text['author_id'].notnull()]

random_sample = authors.sample(n=100, random_state=7)

clusters = pd.read_csv('/vast/amh9750/opinions-cluster-data-lc.csv')

merged_df = pd.merge(random_sample, clusters, how='left', left_on='cluster_id', right_on='id')

new_sample = merged_df[['id_x', 'cluster_id', 'docket_id', 'case_name', 'case_name_full', 'decision_text']]

new_sample = new_sample.rename(columns={'id_x': 'id'})

new_sample.to_csv('/scratch/amh9750/lower_court_random_sample.csv', index=False)
