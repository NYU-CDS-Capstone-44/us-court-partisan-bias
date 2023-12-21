# us-court-partisan-bias
Notes: 
<br>*All files paths will need to be updated in scripts before running
<br>*Additional files are found in our [GoogleDrive](https://drive.google.com/drive/folders/1FLyUYnxbc8VfNZUw-J5uK30uddR2MNgP?usp=drive_link) 

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
*See all hyperparameter tuning results in [GoogleDrive](https://drive.google.com/drive/folders/1FLyUYnxbc8VfNZUw-J5uK30uddR2MNgP?usp=drive_link) as RNN_LSTM_model_comparison
<br><br>RNN_LSTM_partisan_test_predictions \& RNN_LSTM_topic_test_predictions: test accuracy of RNN LSTM model:
<br>1.) Run test inference in .py and .s files
<br>2.) Output of test inference results in .csv file

### best_bert_models
partisan_models \& topic_models: best BERT and LEGAL-BERT models after hyperparameter tuning
<br>1.) Scripts to train models in .py and .SBATCH files
<br>2.) Completed model in HuggingFace [annabelle add link]
<br><br>partisan_test_predictions \& topic_test_predictions: test accuracy of BERT models:
<br>1.) Run test inference in .py and .s files
<br>2.) Output of test inference results in .csv file

##inference

### bert_inference
1.) [annabelle add instructions]

### prep_inference_results_for_analysis: clean up lower court inference data by merging partisan and topic predictions and obtaining author ids and appointing presidents.
1.) Filter docket data for years  >= 1930 since the file is extremely large to work with using Jupyter Notebook: filter_docket.ipynb.  This output will be used when finding author and court ids in the inference_person_id_cleanup script.
<br>2.) Use each of the inference_person_id_cleanup .py and .s files to clean up each chunk of the partisan inference results.  A Jupyter notebook version of this script that may be easier to follow is found in the archive folder in this folder (clean_author_ids.ipynb).
<br>3.) Merge the cleaned partisan inference results with the topic inference results to get one final csv with all cleaned results using Jupyter notebook: merge_results.ipynb
<br>4.) A compressed version of the final csv output can be found in the [GoogleDrive](https://drive.google.com/drive/folders/1FLyUYnxbc8VfNZUw-J5uK30uddR2MNgP?usp=drive_link) as inference_partisan_topic_results3.csv.gz
### hand_label_sample
1.) Scripts to get inference data sample in .py and .s files
<br>2.) Compare hand coded samples to inference sample in Jupyter notebook: check_hand_code.ipynb
### inference_samples
1.) Inference samples from scripts in hand_label_sample found in .csv file

## visualizations
1.) EDA.ipynb: exploratory data analysis on supreme court data
<br>2.) test_results_BERT_legal.ipynb: test result and accuracy comparison for each partisan model 
<br>3.) test_results_topic.ipynb: test result and accuracy comparison for each topic model 
<br>4.) visualizations: lower court inference analyses.  Includes all visuals from poster and paper. 

## archive
Contains older versions of files 

