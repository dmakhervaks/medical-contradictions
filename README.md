# medical-contradictions

## Download and extracting PubMed data
For full replication of our process, then local download and processing of PubMed is advised. Follow the steps below.

1. Download all of PubMed data locally:
- Navigate to the *pubmed/* folder and run *pubmed_access.py*
- Creates the baseline folder
2. Traverse SNOMED ontology to collect phrases of interest:
- Navigate to the *snomed/* folder and run *snomedct_arr.ipynb*
    - Choose **GROUP_SIZE** to be the desired number in the first block
- Populates a file called *snomed_phrases.tsv* which contains all the SNOMED phrase, cuis, and cui_terms
- Populates a file called *snomed_pairs_auto_labeled.csv* which contains all SNOMED term-pairs and there auto label as defined by our methodology
3. Filter PubMed data to only contain relevant phrases:
- Navigate to the *pubmed/* folder and run *process_pubmed.py [--filter_by_unique_words] [--filter_by_exact_phrases]*
    - *filter_by_unique_words* filters all entries by unique words appearing in SNOMED phrases
        - Results stored in folder *unique_words/*
    - *filter_by_exact_phrases* filters the remaining files by exact matches of SNOMED phrases
        - **NOTE:** folder *unique_words/* must be created prior to running with this argument
        - Results stored in folder *exact_match_snomed_phrases/*

## Creating SNOMED Dataset
1. Ensure that all PubMed data is extracted. Notably you will need the following files/folders:
- *snomed/snomed_pairs_auto_labeled.csv*
- *pubmed/exact_match_snomed_phrases/*
2. Create the SNOMED dataset
- Navigate to the *snomed* folder and run *snomed_dataset_creation.py [--thresh <thresh>] [--num_samples <num_samples>] [--use_mesh]*
- Creates the dataset under *pubmed/exact_match_snomed_phrases/SNOMED_dataset.tsv*




## Example Calls
*python training_nli_cross_encoder.py --yaml cross_encoder_deberta_base.yaml --train_data mednli_cardio,Cardio_C35_G25_WN_N10_SN --eval_data mednli_cardio --eval --train_batch_size 16 --eval_batch_size 32 --eval_steps 50 --metric roc_auc --train --sample_train_subset 200*
