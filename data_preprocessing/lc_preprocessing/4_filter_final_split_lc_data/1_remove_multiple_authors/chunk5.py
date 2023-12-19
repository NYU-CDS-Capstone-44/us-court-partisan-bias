import pandas as pd

chunk = pd.read_csv('/scratch/amh9750/capstone/chunk_filter_preprocess/lc_greater_than_35_chunk_5.csv')

def impute_author_str_w_judge(row):
    if pd.notnull(row['author_str']):
        return row['author_str']
    elif pd.isnull(row['author_id']) and pd.notnull(row['judges']):
        judge_list = row['judges'].split(', ')
        if len(judge_list) == 1:
            return judge_list[0]
    return None

chunk['imputed_column'] = chunk.apply(impute_author_str_w_judge, axis=1)

new_chunk = chunk[(chunk['imputed_column'].notnull()) | (chunk['author_id'].notnull())]

new_chunk.to_csv('lower_court_chunk_5.csv', index=False)
