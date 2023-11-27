from bs4 import BeautifulSoup
import pandas as pd
import csv
from tqdm import tqdm
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
csv.field_size_limit(sys.maxsize)

### TO DO ###
# Make sure to change your net id in all of the file paths
# Search 'TO DO' to make sure the path is correct to the bulk datasets in vast

print("Getting lower court clusters data...")

# Define relevant schemas
opinion_schema = {
    'resource_uri': 'object',
    'id': 'object',
    'absolute_url': 'object',
    'cluster_id': 'object',
    'cluster': 'object',
    'author_id': 'object',
    'author': 'object',
    'joined_by': 'object',
    'date_created': 'object',
    'date_modified': 'object',
    'author_str': 'object',
    'per_curiam': 'str',
    'joined_by_str': 'object',
    'type': 'object',
    'sha1': 'object',
    'page_count': 'str',
    'download_url': 'object',
    'local_path': 'object',
    'plain_text': 'object',
    'html': 'object',
    'html_lawbox': 'object',
    'html_columbia': 'object',
    'html_anon_2020': 'object',
    'xml_harvard': 'object',
    'html_with_citations': 'object',
    'extracted_by_ocr': 'str',
    'opinions_cited': 'object'
}


opinion_clusters_schema = {
    "resource_uri": "string",
    "id": "string",
    "absolute_url": "string",
    "panel": "string",
    "non_participating_judges": "string",
    "docket_id": "string",
    "docket": "string",
    "sub_opinions": "string",
    "citations": "string",
    "date_created": "string",
    "date_modified": "string",
    "judges": "string",
    "date_filed": "string",
    "date_filed_is_approximate": "string",
    "slug": "string",
    "case_name_short": "string",
    "case_name": "string",
    "case_name_full": "string",
    "scdb_id": "string",
    "scdb_decision_direction": "string",
    "scdb_votes_majority": "string",
    "scdb_votes_minority": "string",
    "source": "string",
    "procedural_history": "string",
    "attorneys": "string",
    "nature_of_suit": "string",
    "posture": "string",
    "syllabus": "string",
    "headnotes": "string",
    "summary": "string",
    "disposition": "string",
    "history": "string",
    "other_dates": "string",
    "cross_reference": "string",
    "correction": "string",
    "citation_count": "int",
    "precedential_status": "string",
    "date_blocked": "string",
    "blocked": "string",
    "filepath_json_harvard": "string",
    "arguments": "string",
    "headmatter": "string"
}

#Write cluster data for the lower courts to a CSV file
input_csv_file = '/vast/amh9750/clusters-data.csv' ### TO DO ### #Update File Path: Bulk Clusters Data
output_csv_file = '/vast/amh9750/opinions-cluster-data-lc.csv' #Update File Path: Lower Court Cluster Data

#Create a new CSV file for writing
with open(output_csv_file, 'w', newline='', encoding='latin1') as csvfile:
    
    #Define the fieldnames for your CSV
    fieldnames = list(opinion_clusters_schema.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open(input_csv_file, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f)

        for i, sentence in enumerate(tqdm(reader)):
            if sentence['scdb_id'] == "" and sentence['date_filed']>='1930-01-01':
                #Write the filtered row to the new CSV file
                writer.writerow(sentence)


print("Getting filter set...")

#Write cluster data for filtering for the cluster ids we DON'T want 
#i.e. clusters that are either supreme court or before 1930
input_csv_file = '/vast/amh9750/clusters-data.csv' ### TO DO ### #Update File Path: Bulk Clusters Datset
output_csv_file = '/vast/amh9750/opinions-cluster-data-filtering.csv' #Update File Path: Clusters Datset Filtering

#Create a new CSV file for writing
with open(output_csv_file, 'w', newline='', encoding='latin1') as csvfile:
    
    #Define the fieldnames for your CSV
    fieldnames = list(opinion_clusters_schema.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open(input_csv_file, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f)

        for i, sentence in enumerate(tqdm(reader)):
            #Keep the row if the cluster is supreme court or earlier than 1930
            if sentence['scdb_id'] != "" or sentence['date_filed']<'1930-01-01':
                #Write the filtered row to the new CSV file
                writer.writerow(sentence)


print("Getting lower court opinions data...")

#Read back in the filtering csv
cluster_df_filter = pd.read_csv('/vast/amh9750/opinions-cluster-data-filtering.csv', dtype = opinion_clusters_schema)

#Extract the values of 'id' for filtering
filter_court_case_cluster_ids = cluster_df_filter['id'].tolist()

#Convert the ids to strings for filtering
filter_cluster_ids_map = map(str, filter_court_case_cluster_ids)
filter_court_case_cluster_ids = list(filter_cluster_ids_map)

#Make the id list into a set for more efficient filtering
filter_set = set(filter_court_case_cluster_ids)

#Filter and write opinion data for lower courts from 1930 onwards
input_csv_file = '/vast/amh9750/opinions-data.csv' ### TO DO ### #Update File Path: Bulk Opinions Dataset
output_csv_file = '/vast/amh9750/opinions-data-lc.csv' #Update File Path: Opinions Dataset Lower Courts

#Define the specific types we want to filter for
types = ["010combined", "015unanimous", "020lead", "025plurality", "030concurrence", "035concurrenceinpart", "040dissent"]

#Create a new CSV file for writing
with open(output_csv_file, 'w', newline='', encoding='latin1') as csvfile:
    
    #Define the fieldnames for your CSV
    fieldnames = list(opinion_schema.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open(input_csv_file, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f)

        for i, sentence in enumerate(tqdm(reader)):
            #Take only opinions that have a documented author and are not in the filter set
            if (sentence['cluster_id'] not in filter_set) and (sentence['type'] in types):
                #Write the filtered row to the new CSV file
                writer.writerow(sentence)


print("Getting lower court opinions data with judges...")

judge_text_opinion_schema = {
    'resource_uri': 'object',
    'id': 'object',
    'absolute_url': 'object',
    'cluster_id': 'object',
    'cluster': 'object',
    'author_id': 'object',
    'author': 'object',
    'joined_by': 'object',
    'date_created': 'object',
    'date_modified': 'object',
    'author_str': 'object',
    'per_curiam': 'str',
    'joined_by_str': 'object',
    'type': 'object',
    'sha1': 'object',
    'page_count': 'str',
    'download_url': 'object',
    'local_path': 'object',
    'plain_text': 'object',
    'html': 'object',
    'html_lawbox': 'object',
    'html_columbia': 'object',
    'html_anon_2020': 'object',
    'xml_harvard': 'object',
    'html_with_citations': 'object',
    'extracted_by_ocr': 'str',
    'opinions_cited': 'object',
    'judges': 'str',
    'judge': 'str'
}

# Load in the opinion clusters
cluster_df_lc = pd.read_csv('/vast/amh9750/opinions-cluster-data-lc.csv')

def get_cluster_judges(sentence):
    """
    param: sentence (dictionary)
    return: sentence (dictionary)
    """
    cluster_id = sentence['cluster_id']
    cluster_id = int(cluster_id)
    judges = cluster_df_lc[cluster_df_lc['id']==cluster_id]['judges'].iloc[0]
    sentence['judges'] = judges
    return sentence

def create_judge_column(sentence):
    """
    param: sentence (dictionary)
    return: sentence (dictionary)
    """
    if sentence['author_str']!= '':
        judge = sentence['author_str']
    elif not pd.isna(sentence['judges']):
        judge = sentence['judges']
    elif sentence['author_id']!= '':
        judge = sentence['author_id']
    else:
        judge = None
    sentence['judge'] = judge
    return sentence

# Line by line, add the judges if they do not have an author id or an author string
input_csv_file = '/vast/amh9750/opinions-data-lc.csv' #Update File Path
output_csv_file = '/vast/amh9750/opinions-data-lc-judges-added.csv' #Update File Path

# Create a new CSV file for writing
with open(output_csv_file, 'w', newline='') as csvfile:
    
    # Define the fieldnames for your CSV
    fieldnames = list(judge_text_opinion_schema.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open(input_csv_file, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f)

        for i, sentence in enumerate(tqdm(reader)):
            sentence = get_cluster_judges(sentence)
            sentence = create_judge_column(sentence)
            if sentence['judge'] != None:
                writer.writerow(sentence)


print("Getting lower court opinions plain text...")

text_opinion_schema = {
    'resource_uri': 'object',
    'id': 'object',
    'absolute_url': 'object',
    'cluster_id': 'object',
    'cluster': 'object',
    'author_id': 'object',
    'author': 'object',
    'joined_by': 'object',
    'date_created': 'object',
    'date_modified': 'object',
    'author_str': 'object',
    'per_curiam': 'str',
    'joined_by_str': 'object',
    'type': 'object',
    'sha1': 'object',
    'page_count': 'str',
    'download_url': 'object',
    'local_path': 'object',
    'extracted_by_ocr': 'str',
    'opinions_cited': 'object',
    'judges': 'str',
    'judge': 'str',
    'decision_text': 'str'
}

# The below changes all of the lower court text into plain text and filters out for the types we want 
# Function to extract text from HTML
def extract_text_from_html(html_string):
    """
    param: html_string (str)
    """
    try:
        if pd.isna(html_string) or not isinstance(html_string, str):
            return float('NaN')
        soup = BeautifulSoup(html_string, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return str(e)

# Function to extract text from XML
def extract_text_from_xml(xml_string):
    """
    param: xml_string (str)
    """
    try:
        if pd.isna(xml_string) or not isinstance(xml_string, str):
            return float('NaN')
        soup = BeautifulSoup(xml_string, 'lxml')
        return soup.get_text()
    except Exception as e:
        return str(e)

# Function to apply to each dictionary/row
def get_decision_text(sentence):
    """
    param: sentence (dictionary)
    return: sentence (dictionary)
    """
    if sentence['xml_harvard']!='':
        text = extract_text_from_xml(sentence['xml_harvard'])
    elif sentence['html_with_citations']!='':
        text = extract_text_from_html(sentence['html_with_citations'])
    elif sentence['html_lawbox']!='':
        text = extract_text_from_html(sentence['html_lawbox'])
    elif sentence['html']!='':
        text = extract_text_from_html(sentence['html'])
    elif sentence['plain_text']!='':
        text = sentence['plain_text']
    else:
        text = None
    del sentence['xml_harvard']
    del sentence['html_with_citations']
    del sentence['html_lawbox']
    del sentence['html']
    del sentence['plain_text']
    del sentence['html_columbia']
    del sentence['html_anon_2020']
    sentence['decision_text'] = text
    return sentence

# Filter and write formatted opinion data for lower courts from 1930 onwards for specific types
input_csv_file = '/vast/amh9750/opinions-data-lc-judges-added.csv' #Update File Path: Opinions Text Dataset Lower Courts
output_csv_file = '/vast/amh9750/text-opinions-data-lc.csv' #Update File Path: Converted Opinions Text Dataset Lower Courts

# Create a new CSV file for writing
# Did not specify the encoding here for the output file which prevents the encoding errors
with open(output_csv_file, 'w', newline='') as csvfile:

    # Define the fieldnames for your CSV
    fieldnames = list(text_opinion_schema.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open(input_csv_file, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f)

        for i, sentence in enumerate(tqdm(reader)):
            writer.writerow(get_decision_text(sentence))

print("Done!")


