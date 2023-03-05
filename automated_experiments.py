from dataset_info import Datasets
from training_nli_cross_encoder import load_model_info
import subprocess as s
import time
"""
Automated way to run tons of experiments :P
"""
def train_on_cardio_also():
    num_trials = 3
    # models = ['cross_encoder_distilroberta.yaml', 'cross_encoder_bioelectra.yaml']
    models = ['cross_encoder_bioelectra.yaml']
    eval_datasets = ['cardio','positive_cardio']
    # eval_datasets = ['positive_cardio']
    train_or_not = [True, False]
    for model in models:
        for train in train_or_not:
            for eval_dataset in eval_datasets:
                if train:
                    for dataset in Datasets:
                        dataset_name = dataset.name
                        if 'snomed_exact' in dataset_name or 'snomed_non_exact' in dataset_name:
                            dataset_attrs =  dataset.value
                            dataset_path = dataset_attrs['data_path']
                            num_labels = dataset_attrs['num_labels']
                            metric = dataset_attrs['metric']

                            print(f"model: {model} | train: {train} | eval_dataset: {eval_dataset} | train_dataset: {dataset_name}")
                            if train:
                                for _ in range(num_trials):
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",f"{eval_dataset},{dataset_name}","--eval_data",eval_dataset,"--eval", "--train",
                                        "--train_batch_size", "8", "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc"])
                else:
                    for _ in range(num_trials):
                        s.call(["python","training_nli_cross_encoder.py","--yaml",model, "--train_data",f"{eval_dataset}", "--eval_data",eval_dataset,"--eval", "--train",
                                "--train_batch_size", "8", "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc"])

def train_without_cardio():
    num_trials = 3
    # models = ['cross_encoder_distilroberta.yaml', 'cross_encoder_bioelectra.yaml']
    # eval_datasets = ['cardio','positive_cardio']
    models = ['cross_encoder_bioelectra.yaml']
    eval_datasets = ['positive_cardio']
    train_or_not = [True, False]
    for model in models:
        for train in train_or_not:
            for eval_dataset in eval_datasets:
                if train:
                    for dataset in Datasets:
                        dataset_name = dataset.name
                        if 'snomed_exact' in dataset_name or 'snomed_non_exact' in dataset_name:
                            dataset_attrs =  dataset.value
                            dataset_path = dataset_attrs['data_path']
                            num_labels = dataset_attrs['num_labels']
                            metric = dataset_attrs['metric']

                            print(f"model: {model} | train: {train} | eval_dataset: {eval_dataset} | train_dataset: {dataset_name}")
                            for _ in range(num_trials):
                                s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",dataset_name,"--eval_data",eval_dataset,"--eval",
                                    "--train_batch_size", "8", "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train"])
                else:
                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",dataset_name,"--eval_data",eval_dataset,"--eval",
                            "--train_batch_size", "8", "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc"])


def train_all_models_zipped_train_test(zipped_train_test, train_samples_sizes = [-1], sota=False):
    num_trials = 3
    # models = ['cross_encoder_distilroberta.yaml', 'cross_encoder_bioelectra.yaml']
    # eval_datasets = ['cardio','positive_cardio']
    models = ['cross_encoder_electra_small.yaml',
            'cross_encoder_bert_small.yaml',
            'cross_encoder_deberta_small.yaml',
            'cross_encoder_bioelectra.yaml',
            'cross_encoder_albert_base.yaml',
            'cross_encoder_deberta_base.yaml',
            'cross_encoder_electra_base.yaml',
            'cross_encoder_bert_base.yaml'
            ]

    train_on_addtl = [True]
    # train_on_addtl = [True]
    for model in models:
        model_info = load_model_info(model)
        for train_addtl in train_on_addtl:
            for eval_dataset,rel_dataset in zipped_train_test:
                for sample_size in train_samples_sizes:
                    if train_addtl:
                        for dataset in Datasets:
                            dataset_name = dataset.name
                            if dataset_name == rel_dataset:
                                time.sleep(30)
                                dataset_attrs =  dataset.value
                                dataset_path = dataset_attrs['data_path']
                                num_labels = dataset_attrs['num_labels']
                                metric = dataset_attrs['metric']
                                

                                print(f"model: {model} | train addtl: {train_addtl} | eval_dataset: {eval_dataset} | train_dataset: {dataset_name}")
                                for _ in range(num_trials):
                                    if sota:
                                        if sample_size==-1:
                                            s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset+","+dataset_name,"--eval_data",eval_dataset,"--eval",
                                                "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train","--sota"])
                                        else:
                                            s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset+","+dataset_name,"--eval_data",eval_dataset,"--eval",
                                                "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train","--sota","--sample_train_subset",str(sample_size)])
                                    else:
                                        if sample_size==-1:
                                            s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset+","+dataset_name,"--eval_data",eval_dataset,"--eval",
                                                "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train"])
                                        else:
                                            s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset+","+dataset_name,"--eval_data",eval_dataset,"--eval",
                                                "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train","--sample_train_subset",str(sample_size)])
                    else:
                        for _ in range(num_trials):
                            if sota:
                                if sample_size==-1:
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset,"--eval_data",eval_dataset,"--eval",
                                            "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc", "--train","--sota"])
                                else:
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset,"--eval_data",eval_dataset,"--eval",
                                            "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc", "--train","--sota","--sample_train_subset",str(sample_size)])
                            else:
                                if sample_size==-1:
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset,"--eval_data",eval_dataset,"--eval",
                                            "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc", "--train"])
                                else:
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset,"--eval_data",eval_dataset,"--eval",
                                        "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc", "--train","--sample_train_subset",str(sample_size)])




def train_all_models(eval_datasets = ["cardio"], rel_datasets = [], train_samples_sizes = [-1], sota=False):
    num_trials = 3
    # models = ['cross_encoder_distilroberta.yaml', 'cross_encoder_bioelectra.yaml']
    # eval_datasets = ['cardio','positive_cardio']
    models = ['cross_encoder_electra_small.yaml',
            'cross_encoder_bert_small.yaml',
            'cross_encoder_deberta_small.yaml',
            'cross_encoder_bioelectra.yaml',
            'cross_encoder_albert_base.yaml',
            'cross_encoder_deberta_base.yaml',
            'cross_encoder_electra_base.yaml',
            'cross_encoder_bert_base.yaml'
            ]

    # models = ['cross_encoder_bert_small.yaml'
    #         ]

    # models = ['cross_encoder_electra_base.yaml',
    #         'cross_encoder_bert_base.yaml',
    #         ]

    train_on_addtl = [False,True]
    # train_on_addtl = [True]
    for model in models:
        model_info = load_model_info(model)
        for train_addtl in train_on_addtl:
            for eval_dataset in eval_datasets:
                for sample_size in train_samples_sizes:
                    if train_addtl:
                        for dataset in Datasets:
                            dataset_name = dataset.name
                            if dataset_name in rel_datasets:
                                
                                dataset_attrs =  dataset.value
                                dataset_path = dataset_attrs['data_path']
                                num_labels = dataset_attrs['num_labels']
                                metric = dataset_attrs['metric']
                                

                                print(f"model: {model} | train addtl: {train_addtl} | eval_dataset: {eval_dataset} | train_dataset: {dataset_name}")
                                for _ in range(num_trials):
                                    if sota:
                                        if sample_size==-1:
                                            s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset+","+dataset_name,"--eval_data",eval_dataset,"--eval",
                                                "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train","--sota"])
                                        else:
                                            s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset+","+dataset_name,"--eval_data",eval_dataset,"--eval",
                                                "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train","--sota","--sample_train_subset",str(sample_size)])
                                    else:
                                        if sample_size==-1:
                                            s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset+","+dataset_name,"--eval_data",eval_dataset,"--eval",
                                                "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train"])
                                        else:
                                            s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset+","+dataset_name,"--eval_data",eval_dataset,"--eval",
                                                "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc","--train","--sample_train_subset",str(sample_size)])
                    else:
                        for _ in range(num_trials):
                            if sota:
                                if sample_size==-1:
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset,"--eval_data",eval_dataset,"--eval",
                                            "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc", "--train","--sota"])
                                else:
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset,"--eval_data",eval_dataset,"--eval",
                                            "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc", "--train","--sota","--sample_train_subset",str(sample_size)])
                            else:
                                if sample_size==-1:
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset,"--eval_data",eval_dataset,"--eval",
                                            "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc", "--train"])
                                else:
                                    s.call(["python","training_nli_cross_encoder.py","--yaml",model,"--train_data",eval_dataset,"--eval_data",eval_dataset,"--eval",
                                        "--train_batch_size", str(model_info['batch_size']), "--eval_batch_size", "32", "--eval_steps", "50", "--metric", "roc_auc", "--train","--sample_train_subset",str(sample_size)])




if __name__ == "__main__":
    # train_on_cardio_also()
    # train_without_cardio()
    # eval_datasets = ["cardio"]
    # eval_datasets = ["mednli_cardio"]
    # eval_datasets = ["mednli_surgery","mednli_surgery_unique"]
    # eval_datasets = ["mednli_pulmonology","mednli_pulmonology_unique"]
    # eval_datasets = ["mednli_endocrinology","mednli_endocrinology_unique"]
    # eval_datasets = ["mednli_female_reproductive","mednli_female_reproductive_unique"]

    # eval_datasets = ["mednli_surgery_u_cardio"]

    # eval_datasets = ["mednli_pulmonology_u_cardio"]


    # # intersecting 
    # eval_datasets  = ["mednli_surgery_inter_cardio","mednli_endocrinology_inter_cardio","mednli_nervous_inter_cardio",
    #                 "mednli_urinary_inter_cardio","mednli_female_reproductive_inter_cardio","mednli_obstetrics_inter_cardio"]

    # rel_datasets = ["Surgery_C35_G25_WN_N10_SN","Endocrinology_C35_G25_WN_N10_SN","Nervous_C35_G25_WN_N10_SN",
    #                 "Urinary_C35_G25_WN_N10_SN","Female_Reproductive_C35_G25_WN_N10_SN","Obstetrics_C35_G25_WN_N10_SN"]

    # "Obstetrics_C35_G25_WN_N10_SN"]
    # unique
    # eval_datasets = ["mednli_surgery_unique","mednli_endocrinology_unique","mednli_immuno_unique",
    #                 "mednli_urinary_unique","mednli_female_reproductive_unique"]

    # rel_datasets = ["Surgery_C35_G25_WN_N10_SN","Endocrinology_C35_G25_WN_N10_SN","Immuno_C35_G25_WN_N10_SN","Urinary_C35_G25_WN_N10_SN",
    #                 "Female_Reproductive_C35_G25_WN_N10_SN"]

    # test = [Endocrinology_C35_G25_WN_N10_SN,Immuno_C35_G25_WN_N10_SN,Urinary_C35_G25_WN_N10_SN,Female_Reproductive_C35_G25_WN_N10_SN,Obstetrics_C35_G25_WN_N10_SN]


    # eval_datasets = ["mednli_female_reproductive"]
    # rel_datasets = ["All_C35_G25_WN_N10_SN","Female_Reproductive_C35_G25_WN_N10_SN","Cardio_C35_G25_WN_N10_SN"]

    # eval_datasets = ["mednli_surgery"]
    # rel_datasets = ["All_C35_G25_WN_N10_SN","Surgery_C35_G25_WN_N10_SN","Cardio_C35_G25_WN_N10_SN"]

    # eval_datasets = ["mednli_endocrinology"]
    # rel_datasets = ["All_C35_G25_WN_N10_SN","Endocrinology_C35_G25_WN_N10_SN","Cardio_C35_G25_WN_N10_SN"]

    eval_datasets = ["mednli_obstetrics"]
    rel_datasets = ["All_C35_G25_WN_N10_SN","Obstetrics_C35_G25_WN_N10_SN","Cardio_C35_G25_WN_N10_SN"]

    # eval_datasets = ["positive_cardio"]
    # eval_datasets = ["mednli"]
    # rel_datasets = ["snmd_193502_thresh_hwvr_0_75_coder"]
    # rel_datasets = ["snmd_308634_thresh_hwvr_0_56_coder"]
    # rel_datasets = ["snomed_exact_matches_6599"]

    # rel_datasets = ["Cardio_M2_G12_WN_N10_SN",
    #                 "Cardio_M2_G12_WN_N25_SN",
    #                 "Cardio_M2_G12_WN_N50_SN",
    #                 "Cardio_M2_G25_WN_N10_SN",
    #                 "Cardio_M2_G25_WN_N25_SN",
    #                 "Cardio_M2_G25_WN_N50_SN"]

    # rel_datasets = ["Cardio_M2_G50_WN_N10_SN",
    #                 "Cardio_M2_G50_WN_N25_SN",
    #                 "Cardio_M2_G50_WN_N50_SN",
    #                 "Cardio_M2_G6_WN_N10_SN",
    #                 "Cardio_M2_G6_WN_N25_SN",
    #                 "Cardio_M2_G6_WN_N50_SN"]

    # rel_datasets = ["Cardio_M0_G12_WY_N25_SN",
    #                 "Cardio_M0_G25_WY_N10_SN",
    #                 "Cardio_M0_G25_WY_N25_SN"]

        
    # rel_datasets =  ["Cardio_M2_G12_WY_N25_SN",
    #                 "Cardio_M2_G25_WY_N10_SN",
    #                 "Cardio_M2_G25_WY_N25_SN"]


    # rel_datasets = ["Cardio_C2_G12_WN_N10_SN",
    #                 "Cardio_C2_G12_WN_N25_SN",
    #                 "Cardio_C2_G25_WN_N10_SN",
    #                 "Cardio_C2_G25_WN_N25_SN",
    #                 "Cardio_C35_G12_WN_N10_SN",
    #                 "Cardio_C35_G12_WN_N25_SN",
    #                 "Cardio_C35_G25_WN_N10_SN",
    #                 "Cardio_C35_G25_WN_N25_SN",
    #                 "Cardio_M35_G12_WN_N10_SN",
    #                 "Cardio_M35_G12_WN_N25_SN",
    #                 "Cardio_M35_G25_WN_N12_SN",
    #                 "Cardio_M35_G25_WN_N25_SN"]


    # rel_datasets = ["Cardio_C2_G12_WN_N10_SN",
    #                 "Cardio_C2_G12_WN_N25_SN",
    #                 "Cardio_C2_G25_WN_N10_SN",
    #                 "Cardio_C2_G25_WN_N25_SN"]

    
    # rel_datasets = ["Cardio_C35_G12_WN_N10_SN",
    #                 "Cardio_C35_G12_WN_N25_SN",
    #                 "Cardio_C35_G25_WN_N10_SN",
    #                 "Cardio_C35_G25_WN_N25_SN",]


    # only tested these on the small models
    # rel_datasets = ["Cardio_M35_G12_WN_N10_SN",
    #                 "Cardio_M35_G12_WN_N25_SN",
    #                 "Cardio_M35_G25_WN_N12_SN",
    #                 "Cardio_M35_G25_WN_N25_SN",
    #                 "Cardio_C35_G25_WN_N50_SN"]

    # rel_datasets = ["Cardio_M0_G12_WN_N10_SN"]

    # rel_datasets = ["Cardio_C35_G6_WN_N10_SN",
    #                 "Cardio_C35_G6_WN_N25_SN",
    #                 "Cardio_C35_G6_WN_N50_SN",
    #                 "Cardio_C35_G12_WN_N50_SN",
    #                 "Cardio_C35_G50_WN_N10_SN",
    #                 "Cardio_C35_G50_WN_N25_SN",
    #                 "Cardio_C35_G50_WN_N50_SN"]

    # rel_datasets = ["Cardio_C35_G12_WN_N50_SN"]

    # rel_datasets = ["Cardio_C35_G25_WN_N50_SN","Cardio_C35_G25_WN_N10_SN"]

    # rel_datasets = ["All_C35_G25_WN_N10_SN"]
    # rel_datasets = ["Cardio_C35_G25_WN_N10_SN"]
    # rel_datasets = ["Cardio_C35_G25_WN_N50_SN"]
    # rel_datasets = ["Pulmonology_C35_G25_WN_N10_SN"]
    # rel_datasets = ["Pulmonology_C35_G25_WN_N50_SN"]
    # rel_datasets = ["Surgery_C35_G25_WN_N10_SN"]
    # rel_datasets = ["Surgery_C35_G25_WN_N50_SN"]

    # Sets evaluated on positive cardio
    # rel_datasets = ["Cardio_C35_G25_WN_N50_SN"
    #                 "Cardio_C35_G12_WN_N10_SN",
    #                 "Cardio_C35_G25_WN_N10_SN",
    #                 "Cardio_M0_G12_WN_N25_SN",
    #                 "Cardio_M2_G12_WN_N25_SN",
    #                 "Cardio_M2_G25_WN_N10_SN",
    #                 "Cardio_C2_G25_WN_N10_SN",
    #                 "Cardio_C2_G25_WN_N25_SN",
    #                 "Cardio_C35_G25_WN_N25_SN",
    #                 "Cardio_M0_G25_WN_N25_SN",
    #                 "Cardio_M2_G25_WN_N25_SN"]

    # rel_datasets = ["All_M2_G12_WN_N10_SN",
    #                 "All_M2_G25_WN_N10_SN",
    #                 "All_M35_G12_WN_N10_SN",
    #                 "All_M35_G25_WN_N10_SN"]

    # rel_datasets = ["All_C35_G12_WN_N10_SN"]

    # rel_datasets = ["Cardio_M0_G12_WN_N10_SN",
    #                 "Cardio_M0_G12_WN_N25_SN",
    #                 "Cardio_M0_G12_WN_N50_SN",
    #                 "Cardio_M0_G25_WN_N10_SN",
    #                 "Cardio_M0_G25_WN_N25_SN",
    #                 "Cardio_M0_G25_WN_N50_SN",
    #                 "Cardio_M0_G50_WN_N10_SN",
    #                 "Cardio_M0_G50_WN_N25_SN",
    #                 "Cardio_M0_G50_WN_N50_SN",
    #                 "Cardio_M0_G6_WN_N10_SN",
    #                 "Cardio_M0_G6_WN_N25_SN",
    #                 "Cardio_M0_G6_WN_N50_SN",
    #                 "Cardio_M0_GALL_WN_N10_SN",
    #                 "Cardio_M0_GALL_WN_N25_SN",
    #                 "Cardio_M0_GALL_WN_N50_SN"]

    # train_all_models_zipped_train_test(zip(eval_datasets, rel_datasets), train_samples_sizes=[-1])
    train_all_models(eval_datasets, rel_datasets, train_samples_sizes=[100,-1])
        # python training_nli_cross_encoder.py --yaml cross_encoder_distilroberta.yaml --train_data snomed_contra_dataset_exact_matches_311 --eval_data cardio  --eval --train --train_batch_size 16 --eval_batch_size 32 --eval_steps 50 --metric roc_auc
        # python training_nli_cross_encoder.py --yaml cross_encoder_bioelectra.yaml --train_data however_moreover --eval_data mednli  --eval --train --train_batch_size 8 --eval_batch_size 32 --eval_steps 10000 --metric roc_auc --save
