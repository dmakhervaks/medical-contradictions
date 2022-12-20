import gzip, csv
from sentence_transformers.readers import InputExample
import wandb
from dataset_info import Datasets


def parse_training_meta_data(args,eval_data_info):

    # if we pass in more than one training set, it will comma separated
    # in this case we juse shuffle/combine the sets
    train_data_list = args.train_data.split(",")
    train_data_info_list = []
    for data_name in train_data_list:
        curr_train_data_info = Datasets[data_name].value
        train_data_info_list.append(curr_train_data_info)

        if curr_train_data_info['metric'] == eval_data_info['metric']:
            print("WARNING: metrics do not match!")

    return train_data_info_list


def process_dataset(eval_data_info, num_labels, args):

    train_data_info_list = parse_training_meta_data(args, eval_data_info)


    # label2int_original = {"contradiction": 0, "entailment": 1, "neutral": 2}
    # label2int_cardio = {"contradiction": 0, "non-contradiction":1}
    label2int_original = {"contradiction": 1, "entailment": 0, "neutral": 2} # TAKE INTO ACCOUNT POSITIVE CLASS
    label2int_cardio = {"contradiction": 1, "non-contradiction":0}
    train_class_counts = [0]*num_labels
    dev_class_counts = [0]*num_labels
    test_class_counts = [0]*num_labels
    train_samples, dev_samples, test_samples = [], [], []

    for train_data_info in train_data_info_list:
        with gzip.open(train_data_info['data_path'], 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if train_data_info['num_labels'] == 3:
                    label_id = label2int_original[row['label']]
                elif train_data_info['num_labels'] == 2:
                    label_id = label2int_cardio[row['label']]
                else:
                    assert False
                # TODO: remove this if done experimenting with 2 class system
                if num_labels == 2 and label_id != 2:
                    if row['split'] == 'train':
                        train_class_counts[label_id]+=1
                        train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
                elif num_labels == 3: 
                    if row['split'] == 'train':
                        train_class_counts[label_id]+=1
                        train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

    # have a different test/dev path potentially depending on the experiment
    with gzip.open(eval_data_info['data_path'], 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if eval_data_info['num_labels'] == 3:
                label_id = label2int_original[row['label']]
            elif eval_data_info['num_labels'] == 2:
                label_id = label2int_cardio[row['label']]
            else:
                assert False
            # TODO: remove this if done experimenting with 2 class system
            if num_labels == 2 and label_id != 2:
                if row['split'] == 'dev':
                    dev_class_counts[label_id]+=1
                    dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
                elif row['split'] == 'test':
                    test_class_counts[label_id]+=1
                    test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
            elif num_labels == 3: 
                if row['split'] == 'dev':
                    dev_class_counts[label_id]+=1
                    dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))
                elif row['split'] == 'test':
                    test_class_counts[label_id]+=1
                    test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

    if args.train:
        wandb.config.update({"distribution_train": train_class_counts})
    
    wandb.config.update({"distribution_dev": dev_class_counts})
    wandb.config.update({"distribution_test": test_class_counts})

    assert len(dev_samples) == sum(dev_class_counts)
    assert len(test_samples) == sum(test_class_counts)

    return train_samples, dev_samples, test_samples