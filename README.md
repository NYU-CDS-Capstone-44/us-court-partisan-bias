# us-court-partisan-bias
*Note: All files paths will need to be updated in scripts before running

## data_preprocessing
### Data to download from SCDB and CourtListener: 
Courtlistener: opinions, opinions-cluster, dockets, people-db-people, people-db-positions
<br>SCDB: SCDB_2022_01_caseCentered_Citation, SCDB_Legacy_07_caseCentered_Citation

### Unzip data from CourtListener and SCDB: 
1.) Run decompress.py using decompress\_file.s batch script for opinions file
<br>2.) Use gunzip command for all other files, if necessary

### sc_preprocessing: preprocess Supreme Court data for modeling and testing
1.) Filter opinions data for SC data using Jupyter Notebook: extract_sc_data.ipynb
<br>2.) Split into train/val/test data using Jupyter Notebook: sc_data_splitting.ipynb
<br>3.) For topic area prediction, extract the topic area using: extract_topic_area.ipynb

### training_data_summarization: summarize Supreme Court opinion text in training data for model training
1.) [mary add instructions]

### lc_preprocessing: preprocess lower court data for inference and analysis
1.) [annabelle add instructions]

## models

### word2vec_embeddings
1.) Train Word2Vec embeddings (for use in baseline and RNN LSTM models using Jupyter Notebook: w2v_embeddings.ipynb
<br>2.) Completed Word2Vec embeddings: w2v_embedding_model.bin

### baseline_models
partisan_baseline \& topic_baseline: grid search and gradient boosted trees baseline model training 
<br>1.) Scripts to train model/grid search in .py and .s files
<br>2.) Completed model is .pkl file
<br><br>baseline_partisan_test_predictions \& baseline_topic_test_predictions: test accuracy of baseline model:
<br>1.) Run test inference in .py and .s files
<br>2.) Output of test inference results in .csv file

### best_RNN_LSTM_model
partisan_RNN_LSTM \& topic_RNN_LSTM: best RNN LSTM models after hyperparameter tuning
<br>1.) Scripts to train model in .py and .s files
<br>2.) Completed model is .h file
*See all hyperparameter tuning results in [GoogleDrive](https://drive.google.com/drive/folders/1FLyUYnxbc8VfNZUw-J5uK30uddR2MNgP?usp=drive_link)


### best_bert_models


