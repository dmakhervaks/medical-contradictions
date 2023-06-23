import csv
import sys
from pathlib import Path

import numpy as np
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from utils.dataset import NLIDataset
from utils.helpers import create_model
from utils.mednli import read_mednli, read_sentences
from utils.pickle import load_pickle
from utils.torch import load_weights, create_data_loader, to_device

RESULTS_DIRECTORY =  "" # TODO: PLEASE ENTER A VALID RESULTS DIRECTORY

def save_predictions(predictions, filename, cfg, score):
    score = round(score,3)
    filename = str(filename).split(".csv")[0] + str(score) + '.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f)

        # write the header
        if predictions.shape[1] == 3:
            labels = sorted(NLIDataset.THREE_LABEL_TO_ID.keys(), key=lambda l: NLIDataset.THREE_LABEL_TO_ID[l])
        else:
            labels = sorted(NLIDataset.TWO_LABEL_TO_ID.keys(), key=lambda l: NLIDataset.TWO_LABEL_TO_ID[l])

        header = ['label_pred', ] + labels
        writer.writerow(header)

        labels_pred = [labels[i] for i in predictions.argmax(axis=1)]
        base_dir = f'{RESULTS_DIRECTORY}{cfg.dataset.value}/{cfg.model.name.lower()}'
        if not os.path.exists(base_dir):
            Path(base_dir).mkdir(parents=True, exist_ok=False)
        pred_score_file = f'{base_dir}/pred_scores_{score}'
        np.save(pred_score_file,predictions)
        for label, probs in zip(labels_pred, predictions):
            writer.writerow([label, ] + [f'{p:.5f}' for p in probs])

    print(f'Saved: {filename}')


def get_input_data(filename):
    input_data = None
    if filename.suffix == '.jsonl':
        input_data = read_mednli(filename)

    if filename.suffix == '.txt':
        input_data = read_sentences(filename)

    if input_data is None:
        raise ValueError(f'Cannot determine input file format: {filename}')

    return input_data


def main(model_spec_filename, input_filename, output_filename):
    model_spec = load_pickle(model_spec_filename)
    model_name = model_spec['model_name']
    model_params = model_spec['model_params']
    vocab = model_spec['vocab']
    cfg = model_spec['cfg']

    print(f'Model name: {model_name}')
    print(f'Model params: {model_params}')

    model = create_model(cfg, model_params)
    model.eval()
    load_weights(model, cfg.models_dir.joinpath(f'{model_name}.pt'))

    input_data = get_input_data(input_filename)
    num_labels = 3 if 'mednli' in cfg.dataset.value else 2

    print(f'Input data: {len(input_data)}')

    dataset = NLIDataset(input_data, vocab=vocab, lowercase=cfg.lowercase, max_len=cfg.max_len, three_labels=num_labels==3)
    data_loader = create_data_loader(dataset, cfg.batch_size, shuffle=False)

    predictions = []
    labels_true = []
    for batch in data_loader:
        (premise, hypothesis), label = to_device(batch)
        logits = model(premise, hypothesis)
        probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()
        predictions.append(probabilities)

        labels_true.extend(label.cpu().numpy())

    predictions = np.concatenate(predictions)
    print(f'Predictions: {predictions.shape}')

    # calc the accuracy
    labels_pred = predictions.argmax(axis=1)
    # accuracy = accuracy_score(labels_true, labels_pred)
    # print(f'Accuracy: {accuracy:.3f}')

    # 3 class case
    # labels_true = label_binarize(labels_true, classes=[0,1,2])
    # labels_pred = label_binarize(labels_pred, classes=[0,1,2])
    # # roc_auc = roc_auc_score(labels_true, labels_pred, multi_class='ovo')
    # roc_auc = roc_auc_score(labels_true, predictions, multi_class='ovo')

    # 2 class case
    if predictions.shape[1] == 3:
        labels_true = np.array(labels_true)
        labels_true[labels_true==2] = 1
        labels_pred[labels_pred==2] = 1
        predictions_2_class = predictions[:,:2]
        predictions_2_class[:,1] += predictions[:,2]
    else:
        labels_true = np.array(labels_true)
        predictions_2_class = predictions

    # flip so contradiction is the positive label
    labels_true = np.where((labels_true == 0) | (labels_true == 1), labels_true ^ 1, labels_true)
    labels_pred = np.where((labels_pred == 0) | (labels_pred == 1), labels_pred ^ 1, labels_pred)
    predictions_2_class = np.column_stack((predictions_2_class[:,1],predictions_2_class[:,0]))

    roc_auc = roc_auc_score(labels_true, predictions_2_class[:,1])

    print("python", model_spec_filename, input_filename, output_filename)
    print(f'ROC AUC: {roc_auc:.3f}')

    save_predictions(predictions, output_filename, cfg, roc_auc)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'Usage: {__file__} <model_spec_filename> <input_file> <output_file>')
        sys.exit(1)

    model_spec_filename = Path(sys.argv[1])
    input_filename = Path(sys.argv[2])
    output_filename = Path(sys.argv[3])
    main(model_spec_filename, input_filename, output_filename)
