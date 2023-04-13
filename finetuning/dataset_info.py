from enum import Enum

class Datasets(Enum):
    
    cardio = {'data_path':'data/Cardio.tsv.gz','metric':'recall','num_labels':2}

    positive_cardio = {'data_path':'data/Positive_Cardio.tsv.gz','metric':'recall','num_labels':2}

    allnli =  {'data_path':'data/AllNLI.tsv.gz','metric':'accuracy','num_labels':3}

    mednli =  {'data_path':'data/MedNLI.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_cardio =  {'data_path':'data/MedNLI_Cardio.tsv.gz','metric':'accuracy','num_labels':3}


    # *********************

    # special subsets of mednli

    mednli_surgery =  {'data_path':'data/MedNLI_Surgery.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_pulmonology =  {'data_path':'data/MedNLI_Pulmonology.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_endocrinology =  {'data_path':'data/MedNLI_Endocrinology.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_female_reproductive =  {'data_path':'data/MedNLI_Female_Reproductive.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_obstetrics =  {'data_path':'data/MedNLI_Obstetrics.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_urinary =  {'data_path':'data/MedNLI_Urinary.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_nervous =  {'data_path':'data/MedNLI_Nervous.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_digestive =  {'data_path':'data/MedNLI_Digestive.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_immuno =  {'data_path':'data/MedNLI_Immuno.tsv.gz','metric':'accuracy','num_labels':3}


    # *********************

    # 100 train-sample subsets of mednli
    
    mednli_100 =  {'data_path':'data/MedNLI_Cardio_100.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_cardio_100 =  {'data_path':'data/MedNLI_Cardio_100.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_surgery_100 =  {'data_path':'data/MedNLI_Surgery_100.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_endocrinology_100 =  {'data_path':'data/MedNLI_Endocrinology_100.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_female_reproductive_100 =  {'data_path':'data/MedNLI_Female_Reproductive_100.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_obstetrics_100 =  {'data_path':'data/MedNLI_Obstetrics_100.tsv.gz','metric':'accuracy','num_labels':3}

    # inter_cardio

    mednli_surgery_inter_cardio =  {'data_path':'data/MedNLI_Surgery_inter_Cardio.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_endocrinology_inter_cardio =  {'data_path':'data/MedNLI_Endocrinology_inter_Cardio.tsv.gz','metric':'accuracy','num_labels':3}
    
    mednli_nervous_inter_cardio =  {'data_path':'data/MedNLI_Nervous_inter_Cardio.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_female_reproductive_inter_cardio =  {'data_path':'data/MedNLI_Female_Reproductive_inter_Cardio.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_urinary_inter_cardio =  {'data_path':'data/MedNLI_Urinary_inter_Cardio.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_obstetrics_inter_cardio =  {'data_path':'data/MedNLI_Obstetrics_inter_Cardio.tsv.gz','metric':'accuracy','num_labels':3}

    # unique from other fields

    mednli_surgery_unique =  {'data_path':'data/MedNLI_Surgery_Unique.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_endocrinology_unique =  {'data_path':'data/MedNLI_Endocrinology_Unique.tsv.gz','metric':'accuracy','num_labels':3}
    
    mednli_immuno_unique =  {'data_path':'data/MedNLI_Immuno_Unique.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_urinary_unique =  {'data_path':'data/MedNLI_Urinary_Unique.tsv.gz','metric':'accuracy','num_labels':3}

    mednli_female_reproductive_unique =  {'data_path':'data/MedNLI_Female_Reproductive_Unique.tsv.gz','metric':'accuracy','num_labels':3}

    # 

    mednli_pulmonology_unique =  {'data_path':'data/MedNLI_Pulmonology_Unique.tsv.gz','metric':'accuracy','num_labels':3}




    # mednli_justcardio =  {'data_path':'data/MedNLI_JustCardio.tsv.gz','metric':'accuracy','num_labels':3}

    snomed_phrases =  {'data_path':'data/SNOMED_Phrase_Pairs.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_phrases_118_all_test = {'data_path':'data/SNOMED_Phrase_Pairs_No_Nums_118_all_test.tsv.gz','metric':'accuracy','num_labels':2}

    # however_moreover = {'data_path':'data/However_Moreover.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced = {'data_path':'data/However_Moreover_Balanced.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced_cardio = {'data_path':'data/However_Moreover_Balanced_Cardio.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced_cardio_splitted = {'data_path':'data/However_Moreover_Balanced_Cardio_Splitted.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced_cardio_4096 =  {'data_path':'data/However_Moreover_Balanced_Cardio_4096.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced_cardio_4096_splitted =  {'data_path':'data/However_Moreover_Balanced_Cardio_4096_Splitted.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced_cardio_4096_filtered =  {'data_path':'data/However_Moreover_Balanced_Cardio_4096_Filtered.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced_cardio_manual_sent_tok_4096 =  {'data_path':'data/However_Moreover_Balanced_Cardio_Manual_Sent_Tokenizer_4096.tsv.gz','metric':'accuracy','num_labels':2}

    however_moreover_balanced_cardio_manual_sent_tokenizer = {'data_path':'/home/davem/Sentence_Transformers/data/However_Moreover_Balanced_Cardio_Manual_Sent_Tokenizer.tsv.gz','metric':'accuracy','num_labels':2}

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

    snomed_exact_matches_311_shuffled =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_311_Shuffled.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_753 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_753.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_1457 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_1457.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_2748 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_2748.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_6102 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_6102.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_4484 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_4484.tsv.gz','metric':'accuracy','num_labels':2}


    # I think all under 12 phrases with special sampling    
    snomed_exact_matches_25175 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_25175.tsv.gz','metric':'accuracy','num_labels':2}


    # ************

    snomed_non_exact_matches_980 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_980.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_non_exact_matches_2406 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_2406.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_non_exact_matches_4701 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_4701.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_non_exact_matches_9041 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_9041.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_non_exact_matches_20825 =  {'data_path':'data/SNOMED_Contra_Dataset_Non_Exact_Matches_20825.tsv.gz','metric':'accuracy','num_labels':2}

    # ************
    snomed_exact_matches_m_f_4026 = {'data_path':'/home/davem/Sentence_Transformers/data/SNOMED_Contra_Dataset_Exact_Matches_MeSH_Filtered_4026.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_m_f_2013 = {'data_path':'/home/davem/Sentence_Transformers/data/SNOMED_Contra_Dataset_Exact_Matches_MeSH_Filtered_2013.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_m_f_508 = {'data_path':'/home/davem/Sentence_Transformers/data/SNOMED_Contra_Dataset_Exact_Matches_MeSH_Filtered_508.tsv.gz','metric':'accuracy','num_labels':2}

    # ************

    snomed_exact_matches_2660 = {'data_path':'/home/davem/Sentence_Transformers/data/SNOMED_Contra_Dataset_Exact_Matches_2660.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_6599 = {'data_path':'/home/davem/Sentence_Transformers/data/SNOMED_Contra_Dataset_Exact_Matches_6599.tsv.gz','metric':'accuracy','num_labels':2}


    # ************
    # COMPARISON Datasets

    however_for_comparison_1000 = {'data_path':'/home/davem/Sentence_Transformers/data/However_For_Comparison_1000_sampled.tsv.gz','metric':'accuracy','num_labels':2}

    however_for_comparison_548_sampled_evenly = {'data_path':'/home/davem/Sentence_Transformers/data/However_For_Comparison_548_sampled_evenly.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_for_comparison_1000 = {'data_path':'/home/davem/Sentence_Transformers/data/SNOMED_For_Comparison_1000_sampled.tsv.gz','metric':'accuracy','num_labels':2}


    #****************
    hwvr_embed_class_tails_thresh_6599_0_2_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/However_Moreover_Balanced_Cardio_4096_tails_Filtered_On_6599_Thresh_0.2_coder.tsv.gz','metric':'accuracy','num_labels':2}

    hwvr_embed_class_tails_thresh_6599_0_05_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/However_Moreover_Balanced_Cardio_tails_Filtered_On_6599_Thresh_0.05_coder.tsv.gz','metric':'accuracy','num_labels':2}

    however_cardio_class_tails_thresh_6599_0_3_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/However_Moreover_Balanced_Cardio_tails_Filtered_On_6599_Thresh_0.3_coder.tsv.gz','metric':'accuracy','num_labels':2}

    however_cardio_class_tails_thresh_6599_0_01_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/However_Moreover_Balanced_Cardio_tails_Filtered_On_6599_Thresh_0.01_coder.tsv.gz','metric':'accuracy','num_labels':2}

    hwvr_cardio_tails_snomed_labels_thresh_6599_0_3_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/predicted_labels/However_Moreover_Balanced_Cardio_tails_Filtered_On_6599_Thresh_0.3_coder.tsv.gz','metric':'accuracy','num_labels':2}

    however_embedding_classifier_thresh_on_750_0_8_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/However_Moreover_Balanced_Cardio_4096_Filtered_On_750_Thresh_0.8_coder.tsv.gz','metric':'accuracy','num_labels':2}


    snomed_25175_exact_tails_thresh_however_0_1_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/predicted_labels/SNOMED_Contra_Dataset_Exact_Matches_25175_tails_Filtered_On_However_Cardio_Balanced_Thresh_0.1_coder.tsv.gz','metric':'accuracy','num_labels':2}

    snmd_193502_thresh_hwvr_0_75_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/intersecting_labels/SNOMED_Contra_Dataset_Exact_Matches_193502_default_Filtered_On_However_Cardio_Balanced_Thresh_0.75_coder.tsv.gz','metric':'accuracy','num_labels':2}

    snmd_308634_thresh_hwvr_0_56_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/intersecting_labels/SNOMED_Contra_Dataset_Exact_Matches_308634_default_Filtered_On_However_Cardio_Balanced_Thresh_0.56_coder.tsv.gz','metric':'accuracy','num_labels':2}

    #*****************

    snomed_exact_matches_5129 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_5129.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_3339 =  {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_3339.tsv.gz','metric':'accuracy','num_labels':2}

    snomed_exact_matches_m_f_5971 = {'data_path':'data/SNOMED_Contra_Dataset_Exact_Matches_MeSH_Filtered_5971.tsv.gz','metric':'accuracy','num_labels':2}

    #*****************

    transition_00001 = {'data_path':'data/transition_1e-5.tsv.gz','metric':'accuracy','num_labels':2}

    transition_0000001 = {'data_path':'data/transition_1e-7.tsv.gz','metric':'accuracy','num_labels':2}

    transition_001 = {'data_path':'data/transition_001.tsv.gz','metric':'accuracy','num_labels':2}

    transition_x_snomed_4984_0001_88  = {'data_path':'data/transition_x_snomed_4984_1e-4_88.tsv.gz','metric':'accuracy','num_labels':2}

    transition_x_snomed_4984_00005_95 = {'data_path':'data/transition_x_snomed_4984_4e-5_95.tsv.gz','metric':'accuracy','num_labels':2}

    transition_x_snomed_867_00008_995 = {'data_path':'data/transition_x_snomed_867_8e-05_995.tsv.gz','metric':'accuracy','num_labels':2}
    
    transition_x_snomed_1338_0005_999 = {'data_path':'data/transition_x_snomed_1338_0005_999.tsv.gz','metric':'accuracy','num_labels':2}

    transition_x_snomed_1342_0005_999 = {'data_path':'data/transition_x_snomed_1342_0005_999.tsv.gz','metric':'accuracy','num_labels':2}

    hwvr_cardio_default_labels_thresh_6599_0_3_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/intersecting_labels/However_Moreover_Balanced_Cardio_default_Filtered_On_6599_Thresh_0.3_coder.tsv.gz','metric':'accuracy','num_labels':2}

    hwvr_cardio_default_labels_thresh_6599_0_7_coder = {'data_path':'/home/davem/Sentence_Transformers/data/embedding_classifier/intersecting_labels/However_Moreover_Balanced_Cardio_default_Filtered_On_6599_Thresh_0.7_coder.tsv.gz','metric':'accuracy','num_labels':2}



    #********************

    Cardio_M0_G12_WN_N10_SN = {'data_path':'data/Cardio_M0_G12_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G12_WN_N25_SN = {'data_path':'data/Cardio_M0_G12_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G12_WN_N50_SN = {'data_path':'data/Cardio_M0_G12_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G25_WN_N10_SN = {'data_path':'data/Cardio_M0_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G25_WN_N25_SN = {'data_path':'data/Cardio_M0_G25_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G25_WN_N50_SN = {'data_path':'data/Cardio_M0_G25_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G50_WN_N10_SN = {'data_path':'data/Cardio_M0_G50_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G50_WN_N25_SN = {'data_path':'data/Cardio_M0_G50_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G50_WN_N50_SN = {'data_path':'data/Cardio_M0_G50_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G6_WN_N10_SN = {'data_path':'data/Cardio_M0_G6_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G6_WN_N25_SN = {'data_path':'data/Cardio_M0_G6_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G6_WN_N50_SN = {'data_path':'data/Cardio_M0_G6_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_GALL_WN_N10_SN = {'data_path':'data/Cardio_M0_GALL_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_GALL_WN_N25_SN = {'data_path':'data/Cardio_M0_GALL_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_GALL_WN_N50_SN = {'data_path':'data/Cardio_M0_GALL_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}



    #*********************


    Cardio_M2_G12_WN_N10_SN = {'data_path':'data/Cardio_M2_G12_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G12_WN_N25_SN = {'data_path':'data/Cardio_M2_G12_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G12_WN_N50_SN = {'data_path':'data/Cardio_M2_G12_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G25_WN_N10_SN = {'data_path':'data/Cardio_M2_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G25_WN_N25_SN = {'data_path':'data/Cardio_M2_G25_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G25_WN_N50_SN = {'data_path':'data/Cardio_M2_G25_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G50_WN_N10_SN = {'data_path':'data/Cardio_M2_G50_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G50_WN_N25_SN = {'data_path':'data/Cardio_M2_G50_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G50_WN_N50_SN = {'data_path':'data/Cardio_M2_G50_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G6_WN_N10_SN = {'data_path':'data/Cardio_M2_G6_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G6_WN_N25_SN = {'data_path':'data/Cardio_M2_G6_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G6_WN_N50_SN = {'data_path':'data/Cardio_M2_G6_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_GALL_WN_N25_SN = {'data_path':'data/Cardio_M2_GALL_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}


    #*********************


    Cardio_M0_G12_WY_N25_SN = {'data_path':'data/Cardio_M0_G12_WY_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G25_WY_N10_SN = {'data_path':'data/Cardio_M0_G25_WY_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M0_G25_WY_N25_SN = {'data_path':'data/Cardio_M0_G25_WY_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G12_WY_N25_SN = {'data_path':'data/Cardio_M2_G12_WY_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G25_WY_N10_SN = {'data_path':'data/Cardio_M2_G25_WY_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M2_G25_WY_N25_SN = {'data_path':'data/Cardio_M2_G25_WY_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}


    #*********************

    Cardio_C2_G12_WN_N10_SN = {'data_path':'data/Cardio_C2_G12_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C2_G12_WN_N25_SN = {'data_path':'data/Cardio_C2_G12_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C2_G25_WN_N10_SN = {'data_path':'data/Cardio_C2_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C2_G25_WN_N25_SN = {'data_path':'data/Cardio_C2_G25_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G12_WN_N10_SN = {'data_path':'data/Cardio_C35_G12_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G12_WN_N25_SN = {'data_path':'data/Cardio_C35_G12_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G25_WN_N10_SN = {'data_path':'data/Cardio_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G25_WN_N25_SN = {'data_path':'data/Cardio_C35_G25_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M35_G12_WN_N10_SN = {'data_path':'data/Cardio_M35_G12_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M35_G12_WN_N25_SN = {'data_path':'data/Cardio_M35_G12_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M35_G25_WN_N12_SN = {'data_path':'data/Cardio_M35_G25_WN_N12_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_M35_G25_WN_N25_SN = {'data_path':'data/Cardio_M35_G25_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}

    Cardio_C35_G25_WN_N50_SN = {'data_path':'data/Cardio_C35_G25_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G12_WN_N50_SN = {'data_path':'data/Cardio_C35_G12_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G6_WN_N50_SN = {'data_path':'data/Cardio_C35_G6_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G50_WN_N50_SN = {'data_path':'data/Cardio_C35_G50_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G6_WN_N10_SN = {'data_path':'data/Cardio_C35_G6_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G6_WN_N25_SN = {'data_path':'data/Cardio_C35_G6_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G50_WN_N10_SN = {'data_path':'data/Cardio_C35_G50_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Cardio_C35_G50_WN_N25_SN = {'data_path':'data/Cardio_C35_G50_WN_N25_SN.tsv.gz','metric':'accuracy','num_labels':2}

    Cardio_M35_G25_WN_N10_SN = {'data_path':'data/Cardio_M35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}

    # for regular mednli
    #**********************
    All_M2_G12_WN_N10_SN = {'data_path':'data/All_M2_G12_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    All_M2_G25_WN_N10_SN = {'data_path':'data/All_M2_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    All_M35_G12_WN_N10_SN = {'data_path':'data/All_M35_G12_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    All_M35_G25_WN_N10_SN = {'data_path':'data/All_M35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    All_C35_G12_WN_N10_SN = {'data_path':'data/All_C35_G12_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}

    All_C35_G25_WN_N10_SN = {'data_path':'data/All_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}


    #**********************
    Surgery_C35_G25_WN_N10_SN = {'data_path':'data/Surgery_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Surgery_C35_G25_WN_N50_SN = {'data_path':'data/Surgery_C35_G25_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}

    #**********************
    Pulmonology_C35_G25_WN_N10_SN = {'data_path':'data/Pulmonology_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Pulmonology_C35_G25_WN_N50_SN = {'data_path':'data/Pulmonology_C35_G25_WN_N50_SN.tsv.gz','metric':'accuracy','num_labels':2}


    #**********************
    Endocrinology_C35_G25_WN_N10_SN = {'data_path':'data/Endocrinology_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Immuno_C35_G25_WN_N10_SN = {'data_path':'data/Immuno_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Urinary_C35_G25_WN_N10_SN = {'data_path':'data/Urinary_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Female_Reproductive_C35_G25_WN_N10_SN = {'data_path':'data/Female_Reproductive_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}
    Obstetrics_C35_G25_WN_N10_SN = {'data_path':'data/Obstetrics_C35_G25_WN_N10_SN.tsv.gz','metric':'accuracy','num_labels':2}



    #**********************
    SNOMED_Phrase_Pairs_L3_R8 = {'data_path':'data/SNOMED_Phrase_Pairs_L3_R8.tsv.gz','metric':'accuracy','num_labels':2}
