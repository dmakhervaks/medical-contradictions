from enum import Enum

class Datasets(Enum):
    
    cardio = {'data_path':'data/Cardio.tsv.gz','metric':'recall','num_labels':2}

    positive_cardio = {'data_path':'data/Positive_Cardio.tsv.gz','metric':'recall','num_labels':2}

    allnli =  {'data_path':'data/AllNLI.tsv.gz','metric':'accuracy','num_labels':3}

    mednli =  {'data_path':'data/MedNLI.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_justcardio =  {'data_path':'data/MedNLI_JustCardio.tsv.gz','metric':'accuracy','num_labels':3}

    snomed_phrases =  {'data_path':'data/SNOMED_Phrase_Pairs.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_phrases_118_all_test = {'data_path':'data/SNOMED_Phrase_Pairs_No_Nums_118_all_test.tsv.gz','metric':'accuracy','num_labels':2}

    # however_moreover = {'data_path':'data/However_Moreover.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced = {'data_path':'data/However_Moreover_Balanced.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced_cardio = {'data_path':'data/However_Moreover_Balanced_Cardio.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_nli = {'data_path':'data/However_Moreover_NLI.tsv.gz','metric':'accuracy','num_labels':3}


    # ************

    # NOTE: there may be some non-ideal circumastances by which the datasets below were created... not using them for now...

    # snomed_contra_dataset_exact_matches_159 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_159.tsv.gz','metric':'accuracy','num_labels':2}

    # snomed_contra_dataset_exact_matches_763 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_763.tsv.gz','metric':'accuracy','num_labels':2}

    # snomed_contra_dataset_exact_matches_1481 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_1481.tsv.gz','metric':'accuracy','num_labels':2}

    # snomed_contra_dataset_exact_matches_2822 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_2822.tsv.gz','metric':'accuracy','num_labels':2}

    # snomed_contra_dataset_exact_matches_6420 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_6420.tsv.gz','metric':'accuracy','num_labels':2}

    # snomed_contra_dataset_exact_matches_500k =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_500k.tsv.gz','metric':'accuracy','num_labels':2}

    # mednli_train_snomed_contra_dataset_exact_matches_6420 =  {'data_path':'data/MedNLI_train_SNOMED_Contra_Dataset_Exact_Matches_6420.tsv.gz','metric':'accuracy','num_labels':2}


    # ************

    snomed_exact_matches_311 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_311.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_753 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_753.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_1457 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_1457.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_2748 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_2748.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_6102 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_6102.tsv.gz','metric':'accuracy','num_labels':2}

    # ************

    snomed_non_exact_matches_980 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_980.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_non_exact_matches_2406 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_2406.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_non_exact_matches_4701 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_4701.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_non_exact_matches_9041 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_9041.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_non_exact_matches_20825 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_20825.tsv.gz','metric':'accuracy','num_labels':2}