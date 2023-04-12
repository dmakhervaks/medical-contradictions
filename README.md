# medical-contradictions
Masters Thesis

## Download and extracting PubMed data
For full replication of our process, then local download and processing of PubMed is advised. Follow the steps below.

1. Download all of PubMed data locally:
-   Navigate to the pubmed folder and run *process_pubmed.py*
2. 



## Example Calls
*python training_nli_cross_encoder.py --yaml cross_encoder_deberta_base.yaml --train_data mednli_cardio,Cardio_C35_G25_WN_N10_SN --eval_data mednli_cardio --eval --train_batch_size 16 --eval_batch_size 32 --eval_steps 50 --metric roc_auc --train --sample_train_subset 200*
