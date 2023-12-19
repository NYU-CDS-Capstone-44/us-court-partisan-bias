import pandas as pd

columns_to_drop = ['plain_text', 'html', 'html_lawbox', 'html_columbia', 'html_anon_2020', 'xml_harvard', 'html_with_citations']

large_file_path = '/vast/amh9750/opinions-data-lc.csv'

chunksize = 5000000

chunks = pd.read_csv(large_file_path, chunksize=chunksize)

for i, chunk in enumerate(chunks):

    columns_to_drop = ['plain_text', 'html', 'html_lawbox', 'html_columbia', 'html_anon_2020', 'xml_harvard', 'html_with_citations']

    chunk.drop(columns=columns_to_drop, inplace=True)

    chunk.to_csv(f'lc_opinions_no_text_chunk_{i}.csv', index=False)
