from pathlib import Path

import numpy as np
import glob
import torch
import subprocess as s
import shutil
from ignite.engine import Engine, Events
from ignite.metrics import Loss #, CategoricalAccuracy
# from ignite.contrib.metrics import ROC_AUC
from roc_auc_metric import ROC_AUC
from config import Datasets

from config import Config
from utils.helpers import get_model_params, create_word_embeddings, create_model, get_dataset, randomize_name, \
    create_dirs
from utils.pickle import save_pickle
from utils.torch import create_data_loader, get_trainable_parameters, to_device, save_weights


def get_model_name(cfg):
    model_name = f'{cfg.dataset.value}.{cfg.model.name.lower()}.{cfg.word_embeddings.name.lower()}.{cfg.hidden_size}'

    return model_name


def main(cfg):
    model_name = get_model_name(cfg)
    model_name = randomize_name(model_name)
    num_labels = 3 if 'mednli' in cfg.dataset.value else 2
    num_labels = 2
    print(f'Model name: {model_name}')

    print("About to get dataset")
    dataset_train, dataset_dev=get_dataset(cfg)
    print("dataset_train")
    print(dataset_train)
    print(dir(dataset_train))

    W_emb = create_word_embeddings(cfg, dataset_train.vocab)
    model_params = get_model_params(cfg, W_emb, num_labels)
    model = create_model(cfg, model_params, W_emb=W_emb)

    data_loader_train = create_data_loader(dataset_train, cfg.batch_size, shuffle=True)
    data_loader_dev = create_data_loader(dataset_dev, cfg.batch_size, shuffle=False)

    model_parameters = get_trainable_parameters(model.parameters())
    optimizer = torch.optim.Adam(model_parameters, cfg.learning_rate, weight_decay=cfg.weight_decay, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()

    def update_function(engine, batch):
        model.train()
        optimizer.zero_grad()

        (premise, hypothesis), label = to_device(batch)

        logits = model(premise, hypothesis)
        loss = criterion(logits, label)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_parameters, cfg.max_grad_norm)
        optimizer.step()

        return loss.item()

    def inference_function(engine, batch):
        model.eval()
        with torch.no_grad():
            (premise, hypothesis), label = to_device(batch)

            logits = model(premise, hypothesis)

            return logits, label

    trainer = Engine(update_function)
    evaluator = Engine(inference_function)

    # metrics = [
    #     ('loss', Loss(criterion)),
    #     ('accuracy', CategoricalAccuracy())
    # ]
    metrics = [
        ('loss', Loss(criterion)),
        ('roc_auc', ROC_AUC())
    ]
    for name, metric in metrics:
        metric.attach(evaluator, name)

    best_dev_acc = -np.inf

    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_model(engine):
        nonlocal best_dev_acc

        def format_metric_str(metrics_values):
            metrics_str = ', '.join([
                f'{metric_name} {metrics_values[metric_name]:.3f}' for metric_name, _ in metrics
            ])
            return metrics_str

        evaluator.run(data_loader_train)
        metrics_train = evaluator.state.metrics.copy()

        evaluator.run(data_loader_dev)
        metrics_dev = evaluator.state.metrics.copy()

        print(f'Epoch {engine.state.epoch}', end=' | ')
        print('Train:', format_metric_str(metrics_train), end=' | ')
        print('Dev:', format_metric_str(metrics_dev), end=' ')
        print()

        if metrics_dev['roc_auc'] > best_dev_acc:
            best_dev_acc = metrics_dev['roc_auc']
            save_weights(model,cfg.models_dir.joinpath(f'{model_name}.pt'))

    # save models specifications
    create_dirs(cfg)
    model_spec = dict(model_name=model_name, model_params=model_params, vocab=dataset_train.vocab, cfg=cfg)
    save_pickle(model_spec, cfg.models_dir.joinpath(f'{model_name}.pkl'))

    trainer.run(data_loader_train, max_epochs=cfg.nb_epochs)

    print(f'Best dev accuracy: {best_dev_acc:.3f}')

    print(f'Running Predict...')
    test_data_path = f'{cfg.mednli_dir}/{cfg.dataset.value}_test_stanford_parse.jsonl'
    print(f'{cfg.mednli_dir}/*.pkl')
    pkl_files = glob.glob(f'{cfg.models_dir}/*.pkl')
    saved_output_path = f'{cfg.data_dir}/predictions/predictions_{cfg.dataset.value}_{best_dev_acc}_test.csv'
    pkl_file = cfg.models_dir.joinpath(f'{model_name}.pkl')
    # model_path = pkl_files[0]
    model_path = pkl_file

    s.call(["python", "predict.py",  model_path, test_data_path, saved_output_path])


if __name__ == '__main__':
    data_dir = Path(__file__).parent.joinpath('data/')
    for _ in range(3):
        for d in Datasets:
            print(d)
            cfg = Config(data_dir=data_dir,dataset=d)
            print(f'Training for dataset: {d}')
            main(cfg)
