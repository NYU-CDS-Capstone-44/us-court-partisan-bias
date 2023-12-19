from bs4 import BeautifulSoup
import pandas as pd
import csv
from tqdm import tqdm
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
csv.field_size_limit(sys.maxsize)

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
    'date_filed': 'str',
    'decision_text': 'str'
}

# The below changes all of the lower court text into plain text 
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
    elif sentence['html_columbia']!='':
        text = extract_text_from_html(sentence['html_columbia'])
    elif sentence['html_anon_2020']!='':
        text = extract_text_from_html(sentence['html_anon_2020'])
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

input_csv_file = '/vast/amh9750/opinions-data-lc-date-added.csv' #Update File Path: Opinions Dataset Lower Courts
output_csv_file = '/vast/amh9750/text-opinions-data-lower-court.csv' #Update File Path: Converted Opinions Text Dataset Lower Courts

# Create a new CSV file for writing
# Did not specify the encoding here for the output file which prevents the encoding errors
with open(output_csv_file, 'w', newline='') as csvfile:

    # Define the fieldnames for the CSV
    fieldnames = list(text_opinion_schema.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open(input_csv_file, 'r', encoding='latin1') as f:
        reader = csv.DictReader(f)

        for i, sentence in enumerate(tqdm(reader)):
            writer.writerow(get_decision_text(sentence))

print("Exported lower court opinions with converted plain text!")

