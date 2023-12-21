# us-court-partisan-bias
<br> \*Note: All files paths will need to be updated in scripts before running

## data_preprocessing
### Data to download from SCDB and CourtListener: 
<br>Courtlistener: opinions, opinions-cluster, dockets, people-db-people, people-db-positions
<br>SCDB: SCDB_2022_01_caseCentered_Citation, SCDB_Legacy_07_caseCentered_Citation

### Unzip data from CourtListener and SCDB: 
<br>1.) Run decompress.py using decompress\_file.s batch script for opinions file
<br>2.) Use gunzip command for all other files, if necessary

### sc_preprocessing: preprocess Supreme Court data for modeling and testing
<br>1.) Filter opinions data for SC data using Jupyter Notebook: extract_sc_data.ipynb
<br>2.) Split into train/val/test data using Jupyter Notebook: sc_data_splitting.ipynb
<br>3.) For topic area prediction, extract the topic area using: extract_topic_area.ipynb

