import pandas as pd
import time

#Set Chunk Number
#Set Chunk Number
chunk1 = 3
chunk2 = 1
print('Chunk = ', chunk1,chunk2)

print('Reading in docket and cluster data: ', time.time())
#Load in Docket Data 
columns_to_load = ['id', 'court_id']  # Adjust this list based on your actual column names
try:
    # Reading the CSV file into a DataFrame
    filtered_docket_df = pd.read_csv('/vast/amr10211/dockets-2023-08-31-filtered-withna.csv',
                                     usecols=columns_to_load,
                                     error_bad_lines=False,
                                     warn_bad_lines=True)
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")

#Load in Cluster Data
columns_to_load = ['id', 'docket_id']
cluster_df =  pd.read_csv('/vast/amr10211/opinions-cluster-data-lc.csv', usecols=columns_to_load)

print('Reading in inference data and merging: ', time.time())

#Load Inference Data
inference_df = pd.read_csv(f'/scratch/amh9750/capstone/bert_inference/inference_results_legal_BERT_chunk_{chunk1}_chunk_{chunk2}.csv') #UPDATE

#Merge Inference with Cluster to get Docket_ID and then Docket to get Court_ID
inference_df = inference_df.rename(columns={'id': 'opinion_id'})
cluster_df = cluster_df.rename(columns={'id': 'cluster_id'})
inference_df = inference_df.merge(cluster_df[['cluster_id','docket_id']], how='left', on='cluster_id')
inference_df = inference_df.merge(filtered_docket_df, how='left', left_on='docket_id', right_on='id')

print('Reading in remaining data and formatting: ', time.time())
#Read Remaining Data
people_df = pd.read_csv('/vast/amr10211/people-db-people-2023-08-31.csv.bz2')
people_positions_df = pd.read_csv('/vast/amr10211/people-db-positions-2023-08-31.csv.bz2')
president_df = pd.read_csv('/vast/amr10211/president_metadata.csv')

#Reformat IDs
people_df = people_df.rename(columns={'id': 'person_id'})
people_positions_df = people_positions_df.merge(people_df, how='left', on='person_id')

# Merge DataFrames based on 'author_id' and 'person_id'
merged_df = pd.merge(inference_df, people_df, left_on='author_id', right_on='person_id', how='left')

# Fill missing values in 'imputed_author_str' with 'name_last'
merged_df['imputed_author_str'] = merged_df['author_str'].fillna(merged_df['name_last'])

# Drop redundant 'person_id' column if needed
merged_df = merged_df.drop('person_id', axis=1)

# Update the original inference_df with the changes
inference_df['imputed_author_str'] = merged_df['imputed_author_str']

# Convert 'date_start' and 'date_termination' columns to datetime objects
people_positions_df['date_start'] = pd.to_datetime(people_positions_df['date_start'], format='%Y-%m-%d', errors='coerce')
people_positions_df['date_termination'] = pd.to_datetime(people_positions_df['date_termination'], format='%Y-%m-%d', errors='coerce')
inference_df['date_filed'] = pd.to_datetime(inference_df['date_filed'], format='%Y-%m-%d', errors='coerce')

# Filter people for only justices and presidents
people_positions_df = people_positions_df[people_positions_df['position_type'].str.contains('jus|jud|mag|pres', case=False, regex=True, na=False)]

# Filter for termination date greater than 1930 and null (not terminated)
people_positions_df = people_positions_df[(people_positions_df['date_termination'].dt.year >= 1930) | pd.isnull(people_positions_df['date_termination'])]
president_df['date_start'] = pd.to_datetime(president_df['date_start'])
president_df['date_termination'] = pd.to_datetime(president_df['date_termination'])

# Change author name column name
inference_df['imputed_author_str'] = inference_df['imputed_author_str'].fillna(inference_df['imputed_column'])
inference_df = inference_df.drop('imputed_column', axis=1)

#Make blank person_id columns
inference_df['imputed_person_id'] = inference_df['author_id']


print('Finding person ids: ', time.time())

def find_person_id(row, people_df, people_positions_df):
    '''Logic to impute person_id given multiple criteria in row'''
    # 1. If row['author_id'] exists then return row['author_id']
    # print(row['imputed_author_str'])
    if pd.notnull(row['author_id']):
        return row['author_id']

    # 2. If there is only one people_df['name_last'] that matches row['author_str1']
    name_last_match = people_df[people_df['name_last'] == row['imputed_author_str']]
    name_last_match_count = name_last_match.groupby('name_last').size()
    if name_last_match_count.get(row['imputed_author_str'], 0) == 1:
        return int(name_last_match['person_id'].values[0])

    # 3. If there are multiple people_df['name_last'] that matches row['author_str1']
    multiple_name_last_match = name_last_match_count[name_last_match_count > 1].index
    for name_last in multiple_name_last_match:
        candidate_set = people_positions_df[people_positions_df['name_last'] == name_last]

        # 3a. Check if row['date_filed'] year is between people_positions_df['date_start'] and people_positions_df['date_termination']
        date_match = candidate_set[
            (
                (candidate_set['date_start'].dt.year.le(row['date_filed'].year) | candidate_set['date_start'].isnull()) &
                (
                    candidate_set['date_termination'].dt.year.ge(row['date_filed'].year) | candidate_set['date_termination'].isnull()
                ) &
                (candidate_set['name_last'] == row['imputed_author_str'])
            )
        ]
        if len(date_match) == 1:
            return int(date_match['person_id'].values[0])
        elif len(date_match) > 1:
            candidate_set = date_match

        # 3b. Check if row['imputed_court_id'] = people_df['imputed_court_id']
        court_id_match = candidate_set[(candidate_set['court_id'] == row['imputed_court_id']) & (candidate_set['name_last'] == row['imputed_author_str'])]
        if len(court_id_match) == 1:
            # print('YAY! found from court id')
            return court_id_match['person_id'].values[0]
        elif len(court_id_match) > 1:
            candidate_set = court_id_match

        # 3c. Check if there is a match based on date range and last name
        court_date_name_match = candidate_set[
            ((candidate_set['date_start'].le(row['date_filed']) | candidate_set['date_start'].isnull()) &
             (candidate_set['date_termination'].ge(row['date_filed']) | candidate_set['date_termination'].ge(row['date_filed']) | candidate_set['date_termination'].isnull()) &
             (candidate_set['name_last'] == row['imputed_author_str']))
        ]

        if len(court_date_name_match) == 1:
            return court_date_name_match['person_id'].values[0]

    # 4. If there are zero people_df['name_last'] that matches row['author_str1'], state error
    return None  # You may want to return a default value or handle the error as per your needs

def fill_missing_person_id(row, df, proximity_range=1000):
    '''Logic to impute person_id given neighbors with same author_str'''
    if pd.isnull(row['imputed_person_id']):
        start_index = max(0, row.name - proximity_range)
        end_index = min(len(df), row.name + proximity_range + 1)

        nearby_data = df.loc[start_index:end_index]
        matching_row = nearby_data.loc[~pd.isnull(nearby_data['imputed_person_id']) & (nearby_data['imputed_author_str'] == row['imputed_author_str'])]

        if not matching_row.empty:
            # Use the imputed_person_id of the first matching row found in the proximity
            return matching_row.iloc[0]['imputed_person_id']

    return row['imputed_person_id']

def fill_missing_court_id(row, df, proximity_range=1000):
    '''Logic to impute court_id given neighbors with same author_str'''
    if pd.isnull(row['imputed_court_id']):
        start_index = max(0, row.name - proximity_range)
        end_index = min(len(df), row.name + proximity_range + 1)

        nearby_data = df.loc[start_index:end_index]
        matching_row = nearby_data.loc[~pd.isnull(nearby_data['imputed_court_id']) & (nearby_data['imputed_author_str'] == row['imputed_author_str'])]

        if not matching_row.empty:
            # Use the court_id of the first matching row found in the proximity
            return matching_row.iloc[0]['imputed_court_id']

    return row['imputed_court_id']

# Apply the functions above to each row in the DataFrame
inference_df['imputed_court_id'] = inference_df['court_id']
inference_df['imputed_court_id'] = inference_df.apply(lambda row: fill_missing_court_id(row, inference_df), axis=1)
inference_df['imputed_person_id'] = inference_df.apply(find_person_id, axis=1, people_df=people_df, people_positions_df = people_positions_df)
inference_df['imputed_person_id'] = pd.to_numeric(inference_df['imputed_person_id'], errors='coerce').astype('Int64')
inference_df['imputed_person_id'] = inference_df.apply(lambda row: fill_missing_person_id(row, inference_df), axis=1)

# Create the new Unique IDs based on conditions
inference_df['unique_person_id'] = inference_df.apply(lambda row:
    str(row['imputed_person_id']) if not pd.isnull(row['imputed_person_id'])
    else f"{row['imputed_author_str']}_{row['imputed_court_id']}" if not pd.isnull(row['imputed_author_str']) and not pd.isnull(row['imputed_court_id'])
    else str(row['imputed_author_str']), axis=1
)

print('Finding start dates and termination dates: ', time.time())

inference_df['year_filed'] = inference_df['date_filed'].dt.year

position_date_df = pd.merge(
    inference_df,
    people_positions_df[['person_id', 'appointer_id', 'date_start', 'date_termination']],
    how='left',
    left_on='imputed_person_id',
    right_on='person_id'
)
position_date_df = position_date_df.reset_index(drop=True)


# Filtering by date if date granularity for start and termination, or year if year granularity (jan 1 date)
position_date_df = position_date_df[
    (
        (
            (
                (
                    (position_date_df['date_start'].dt.month != 1) & 
                    (position_date_df['date_start'].dt.day != 1) & 
                    (
                    (position_date_df['date_termination'].dt.month != 1) & 
                    (position_date_df['date_termination'].dt.day != 1) |
                    (position_date_df['date_termination'].isnull())     
                    )    
                ) & (
                    (position_date_df['date_filed'] >= position_date_df['date_start']) & 
                    (position_date_df['date_filed'] <= position_date_df['date_termination'])
                )
            ) | (
                (
                    (position_date_df['date_start'].dt.month == 1) & 
                    (position_date_df['date_start'].dt.day == 1) & 
                    (
                    (position_date_df['date_termination'].dt.month == 1) & 
                    (position_date_df['date_termination'].dt.day == 1) |
                    (position_date_df['date_termination'].isnull())
                    )    
                    
                ) & 
                    (position_date_df['year_filed'] >= position_date_df['date_start'].dt.year) & 
                    (position_date_df['year_filed'] <= position_date_df['date_termination'].dt.year)
                 )
             )
         )
    | (position_date_df['imputed_person_id'].isnull()) 
    | (position_date_df['date_start'].isnull()) 
    | (position_date_df['date_termination'].isnull())
]


# Dropping the temporary 'year_filed' column
position_date_df = position_date_df.drop(columns=['year_filed'])
position_date_df = position_date_df[['opinion_id','appointer_id','date_start','date_termination']].drop_duplicates().reset_index(drop=True)

#Merge back with original inference_df to get back missing rows
visualization_df = inference_df.merge(position_date_df, how='left', on=['opinion_id'])
# Sort by 'date_start' in ascending order
visualization_df = visualization_df.drop_duplicates(subset=['opinion_id'], keep='last').reset_index(drop=True) #For any remaining duplicates, keep final record in people_positions_df

print('Finding appointing presidents: ', time.time())
def impute_president(row):
    '''Imputes appointing president based on judge start date'''
    # Check if president_id is not null, if yes, return original values
    if pd.notnull(row['president_id']):
        return row['president_id'], row['president_name']

    # If president_id is null, look up in president_df based on date conditions
    president_match = president_df[
        (president_df['date_start'] <= row['date_start']) &
        (row['date_start'] <= president_df['date_termination'])
    ]

    # If there's a match, return the imputed values
    if not president_match.empty:
        return president_match.iloc[0]['president_id'], president_match.iloc[0]['president_name']

    # If no match is found, return null values or any other default values as needed
    return None, None

# Merge with president_df to get populated presidents
visualization_df = visualization_df.merge(president_df[['president_id', 'president_name', 'partisanship']], how='left', left_on='appointer_id', right_on='president_id')

# Apply the imputation function to create the new columns
visualization_df[['imputed_president_id', 'imputed_president_name']] = visualization_df.apply(impute_president, axis=1, result_type='expand')

print('Saving to CSV: ', time.time())
visualization_df.to_csv(f'/vast/amr10211/visualization_results_legal_BERT_chunk_{chunk1}_{chunk2}.csv') #UPDATE

                    
