from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer


# dataset = load_dataset("csv",data_files={'train':"cardiac_train.tsv",'val':"cardiac_val.tsv",'dev':"cardiac_test.tsv"},delimiter="\t")
dataset = load_dataset("csv",data_files={'train':"/home/davem/Sentence_Transformers/data/MedNLI_train.tsv",'dev':"/home/davem/Sentence_Transformers/data/MedNLI_dev.tsv",'test':"/home/davem/Sentence_Transformers/data/MedNLI_test.tsv"},delimiter="\t")

tokenizer = AutoTokenizer.from_pretrained("kamalkraj/bioelectra-base-discriminator-pubmed")
max_length = 512
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"],padding='max_length', truncation=True, max_length=max_length)

def map_labels(example):
    mapping = {"contradiction": 0, "entailment": 1, "neutral": 2}
    example['label'] = mapping[example['label']]
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True).map(map_labels)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["dev"].shuffle(seed=42)
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42)

model = AutoModelForSequenceClassification.from_pretrained("kamalkraj/bioelectra-base-discriminator-pubmed", num_labels=3)
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()