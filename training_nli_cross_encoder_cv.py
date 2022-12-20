"""
This examples trains a CrossEncoder for the NLI task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it learns to predict the labels: "contradiction": 0, "entailment": 1, "neutral": 2.
It does NOT produce a sentence embedding and does NOT work for individual sentences.
Usage:
python training_nli.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample
from CESoftmaxRecallEvaluator import CESoftmaxGeneralEvaluator
from CESoftmaxAccuracyEvaluatorAdjusted import CESoftmaxAccuracyEvaluatorAdjusted
import logging
import numpy as np
from datetime import datetime
import os
import gzip
import csv
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


#As dataset, we use SNLI + MultiNLI
#Check if dataset exsist. If not, download and extract  it
# nli_dataset_path = 'data/AllNLI.tsv.gz'
# DATASET_TYPE = "AllNLI"

# nli_dataset_path = 'data/MedNLI.tsv.gz'
# DATASET_TYPE = "MedNLI"

nli_dataset_path = 'data/Cardio.tsv.gz'
DATASET_TYPE = "Cardio"

nli_dataset_path = 'data/Positive_Cardio.tsv.gz'
DATASET_TYPE = "Positive_Cardio"

nli_dataset_path = 'data/SNOMED_Phrase_Pairs.tsv.gz'
DATASET_TYPE = "SNOMED_Phrases"

nli_dataset_path = 'data/SNOMED_Phrase_Pairs_No_Nums_118.tsv.gz'
DATASET_TYPE = "SNOMED_Phrase_Pairs_No_Nums_118.tsv.gz"

nli_dataset_path = 'data/SNOMED_Phrase_Pairs_No_Nums_144.tsv.gz'
DATASET_TYPE = "SNOMED_Phrase_Pairs_No_Nums_144.tsv.gz"

if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)


# Read the AllNLI.tsv.gz file and create the training dataset
logger.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
label2int = {"contradiction": 0, "non-contradiction":1}
train_samples = []
dev_samples = []
test_samples = []
dev_test_samples = []
all_samples = []
count0 = 0
count1 = 0
test_count0 = 0
test_count1 = 0
num_folds = 5

with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        print(row)
        label_id = label2int[row['label']]
        # TODO: remove this if done experimenting with 2 class system
        if label_id == 0:
            count0+=1
        if label_id == 1:
            count1+=1
        if label_id != 2:            
            all_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

random.shuffle(all_samples)
folds = list(np.array_split(all_samples, num_folds))


train_batch_size = 16
num_epochs = 5

# TODO: try other models and see how they perform on phrases
# model = CrossEncoder('/home/davem/Sentence_Transformers/output_cross_encoder_nli_mednli_2_class/training_mednli-2022-11-25_16-05-13',num_labels=2)
# MODEL_NAME = 'cross_encoder_bioelectra_mnli_trained'

# model = CrossEncoder('distilroberta-base',num_labels=2)
# MODEL_NAME = 'distilroberta-base'

# model = CrossEncoder('/home/davem/Sentence_Transformers/output_cross_encoder_nli_trained_mednli_2_class/training_MedNLI-2022-11-25_20-32-20',num_labels=2)
# MODEL_NAME = 'cross_encoder_trained_on_allnli'

# this one should be correct...
# model = CrossEncoder('/home/davem/Sentence_Transformers/output_cross_encoder_nli_mednli_2_class/training_nli-2022-11-25_16-16-18/', num_labels=2)

# MODEL_NAME = 'distilroberta-base'
# MODEL_NAME = 'cross_encoder_distilroberta_finetuned_on_allnli'
# MODEL_NAME = 'cross_encoder_distilroberta_finetuned_on_allnli_mednli'
# MODEL_NAME = 'cross_encoder_bioelectra'
MODEL_NAME = 'cross_encoder_bioelectra_mednli'

model_save_base_path = f'output_{MODEL_NAME}_2_class_cross_validation/finetuned_{DATASET_TYPE}'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'

average_acc = 0
fold_accuracies = []
total_wrong_samples = []

# perform cross-validation num_folds number of times, each one trained for num_epochs epochs
for iter_count in range(num_folds):
    curr_train_samples = []
    curr_test_samples = []

    # assign for the current split
    for i,fold in enumerate(folds):

        if i == iter_count:
            curr_test_samples.extend(list(fold))
        else:
            curr_train_samples.extend(list(fold))

    model_save_path = f'{model_save_base_path}split_{iter_count}'
    print(len(curr_test_samples))
    print(len(curr_train_samples))
    train_dataloader = DataLoader(curr_train_samples, shuffle=True, batch_size=train_batch_size)

    # each time, train from scratch
    # model = CrossEncoder('distilroberta-base',num_labels=2)
    # model = CrossEncoder('/home/davem/Sentence_Transformers/output_cross_encoder_nli_mednli_2_class/training_nli-2022-11-25_16-16-18/', num_labels=2)
    # model = CrossEncoder('/home/davem/Sentence_Transformers/output_cross_encoder_nli_trained_mednli_2_class/training_MedNLI-2022-11-25_20-32-20', num_labels=2) 
    # model = CrossEncoder('kamalkraj/bioelectra-base-discriminator-pubmed', num_labels=2)
    # model = CrossEncoder('/home/davem/Sentence_Transformers/output_cross_encoder_nli_mednli_2_class/training_mednli-2022-11-25_16-05-13', num_labels=2)
    model = CrossEncoder('/home/davem/Sentence_Transformers/important_models/cross_encoder_bioelectra_finetuned_howevermoreover', num_labels=2)
    # model = CrossEncoder('kamalkraj/bioelectra-base-discriminator-pubmed', num_labels=2)
    # model = CrossEncoder('distilroberta-base', num_labels=2)

    #During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
    evaluator = CESoftmaxAccuracyEvaluatorAdjusted.from_input_examples(curr_test_samples, name=f'{DATASET_TYPE}-test')
    # evaluator = CESoftmaxRecallEvaluator.from_input_examples(dev_samples, name=f'{DATASET_TYPE}-dev')


    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))


    # # Train the model
    model.fit(train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=10000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)

    # Train the model
    # model.fit(train_dataloader=train_dataloader,
    #         epochs=num_epochs,
    #         evaluation_steps=10000,
    #         warmup_steps=warmup_steps,
    #         output_path=model_save_path)

    # Loading the best model we have
    model = CrossEncoder(model_save_path)

    # just the test accuracy
    test_evaluator = CESoftmaxAccuracyEvaluatorAdjusted.from_input_examples(curr_test_samples, name=f'{DATASET_TYPE}-test-samples-split{iter_count}')
    curr_acc = test_evaluator(model,output_path=model_save_path)

    fold_accuracies.append(curr_acc)
    average_acc += curr_acc/num_folds
    

epoch_average_acc = np.array([0.0]*num_epochs)
for iter_count in range(num_folds):
    model_save_path = f'{model_save_base_path}split_{iter_count}'
    epoch_results_path = f'{model_save_path}/CESoftmaxAccuracyEvaluator_{DATASET_TYPE}-test_results.csv'
    with open(epoch_results_path, "r") as f:
        lines = f.readlines()[1:]
        assert len(lines) == num_epochs
        for idx, line in enumerate(lines):
            _,_,acc = [x.strip() for x in line.split(",")]
            acc = float(acc)
            epoch_average_acc[idx] += acc

epoch_average_acc /= num_folds
best_epoch = np.argmax(epoch_average_acc) + 1
best_avg_acc = np.max(epoch_average_acc)

results_file = model_save_base_path+"/results.tsv"
with open(results_file,"w") as f:
    f.write("\t".join(("Fold Number","Accuracy")) + "\n")
    for idx, fold_acc in enumerate(fold_accuracies):
        f.write("\t".join((str(idx),str(fold_acc))) + "\n")
    f.write("\n")
    f.write("\t".join(("Epoch Number","Average Accuracy")) + "\n")
    for idx, epoch_acc in enumerate(epoch_average_acc):
        f.write("\t".join((str(idx),str(epoch_acc))) + "\n")
    f.write("\t".join(("Epoch",str(best_epoch),"Average",str(best_avg_acc))) + "\n")

for iter_count in range(num_folds):
    wrong_samples_path = f'{model_save_base_path}split_{iter_count}/wrong_samples.tsv'
    with open(wrong_samples_path,"r") as f:
        if iter_count == 0:
            total_wrong_samples.extend(f.readlines())
        else: 
            total_wrong_samples.extend(f.readlines()[1:])

total_wrong_samples = set(total_wrong_samples)
with open(model_save_base_path+"/total_wrong_samples.tsv","w") as f:
    for w in total_wrong_samples:
        f.write(w)

print(f'/home/davem/Sentence_Transformers/{model_save_base_path}')
print("Epoch",str(best_epoch),"Average",str(best_avg_acc))