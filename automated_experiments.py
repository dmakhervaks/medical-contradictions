from dataset_info import Datasets
import subprocess as s
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


if __name__ == "__main__":
    # train_on_cardio_also()
    train_without_cardio()
        # python training_nli_cross_encoder.py --yaml cross_encoder_distilroberta.yaml --train_data snomed_contra_dataset_exact_matches_311 --eval_data cardio  --eval --train --train_batch_size 16 --eval_batch_size 32 --eval_steps 50 --metric roc_auc
        # python training_nli_cross_encoder.py --yaml cross_encoder_bioelectra.yaml --train_data however_moreover --eval_data mednli  --eval --train --train_batch_size 8 --eval_batch_size 32 --eval_steps 10000 --metric roc_auc --save


        