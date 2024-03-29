# medical-contradictions

## Important Notes
- Please replace the **RESULTS_DIRECTORY** global variable in *analysis/delong_test.py* with your file path
- Please replace the **RESULTS_DIRECTORY** global variable in *finetuning/CESoftmaxRecallEvaluator.py* with your file path

## Download and extracting PubMed data
For full replication of our process, local download and processing of PubMed is advised. Follow the steps below.

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

## Iterating over SNOMED Ontology
1. The script which iterates over the ontology is located in: *snomed/snomedct.ipynb*
2. In order to run the script, you will need to have the following files in the *snomed* directory:
- The metathesaurus file (*2022AA.zip*) -> can be downloaded form here https://www.nlm.nih.gov/research/umls/archive/archive_home.html
- *all_list_of_dicts.tsv* -> Easily downloaded via Google Drive or created via snomed_mining.py
- *snomed_all_phrases_cui.tsv* -> Easily downloaded via Google Drive or created via snomed_mining.py
- *synonym_dict.json* -> Easily downloaded via Google Drive or filled up using use_synonym_heuristic() method
- *antonym_dict.json* -> Easily downloaded via Google Drive or filled up using use_synonym_heuristic() method
3. Run the notebook and play around with hyperparams such as the group size and the field_terms which you want to focus on during tree processing
4. Creates the *snomed_pairs_auto_labeled.csv* file which is used in the SNOMED Dataset creation

## Creating SNOMED Dataset
1. Ensure that all PubMed data is extracted. Notably you will need the following files/folders:
- *snomed/snomed_pairs_auto_labeled.csv*
- *pubmed/exact_match_snomed_phrases/*
2. Create the SNOMED dataset
- Navigate to the *snomed* folder and run *snomed_dataset_creation.py [--thresh <thresh>] [--num_samples <num_samples>] [--use_mesh]*
- Creates the dataset under *pubmed/exact_match_snomed_phrases/SNOMED_dataset.tsv*

## Baselines
- Romanov and Shivade's baseline can be found under *mednli_baseline*
  - Please update the *mednli_baseline/utils/config.py* if you would to train/evaluate on different datasets
  - More specific instructions can be found under *mednli_baseline/README.md*
- Yazi et al.'s baseline can be run with the Fine-tuning command detailed in the section below

## Fine-tuning
1. Run an individual experiment
- Navigate to *finetuning/* and run *training_nli_cross_encoder.py --yaml <model_yaml_name.yaml> --train_data <dataset1,dataset2,...> --eval_data <dataset1> --eval --train_batch_size <train_batch_size> --eval_batch_size <eval_batch_size> --eval_steps <num_eval_steps> --metric <eval_metric_name> --train --sample_train_subset <-1,or size of random sample>* [--sota]
- **NOTE:** The datasets must be one of the ones defined in *dataset_info.py*
- **NOTE:** The --sota argument is used to reproduce Yazi's model
2. Run a batch of experiments
- Navigate to *finetuning/* and run *automated_experiments.py*

## Result Analysis
1. To generate a table similar to that which is seen in the main results section of the paper:
- Navigate to *analysis/* and run *delong_test.py*
- You can specify here which models and datasets to see results from
- **NOTE:** In addition to wandb, during experimentation all scores are saved in the folder *ARR_Results/*
2. To generate the graphs that were used in the paper:
- Navigate to *analysis/* and run *process_results.py*
- **NOTE:** You need to download the results locally from wandb in order for this to work

## Creation of MedNLI subsets
1. To create the subsets of the MedNLI data, first retrieve the original data
- We do not release any MedNLI data due to MIMIC-III constraints
- You can find the data here: *https://physionet.org/content/mednli/1.0.0/*
2. Filter based on medical specialties
- You will find medical sub-field keywords under *snomed/mednli_subspecialties*
- Navigate to *snomed/mednli_subspecialties/* and run *filter_mednli.py*
    - Make sure to update the path to your downloaded MedNLI data
