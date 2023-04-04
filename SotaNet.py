import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator
import wandb
import numpy as np
from plotting import plot_roc

logger = logging.getLogger(__name__)
class SotaNet(nn.Module):
    def __init__(self, model_name ,config, automodel_args:Dict = {}):
      super(SotaNet, self).__init__()
    #   device = "cuda" if torch.cuda.is_available() else "cpu"
    #   logger.info("Use pytorch device: {}".format(device))

    #   self._target_device = torch.device(device)
      self.base_model = AutoModel.from_pretrained(model_name,config=config, **automodel_args)
    #   for param in self.base_model.parameters():
    #         param.requires_grad = False
      hidden_size = self.base_model.config.hidden_size
    #   self.base_model.to(self._target_device)
      self.config = config
      # with an input probability
      self.dropout1 = nn.Dropout(0.3)
      self.dropout2 = nn.Dropout(0.3)
      self.dropout3 = nn.Dropout(0.3)

      self.fc0 = nn.Linear(hidden_size*2, 2048)
      # First fully connected layer
      self.fc1 = nn.Linear(2048, 1024)
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(1024, 512)

      self.fc3 = nn.Linear(512, 2)


    # x represents our data
    def forward(self, a,b):
      a = self.base_model(**a)
      b = self.base_model(**b)

      x = torch.cat((a.pooler_output,b.pooler_output), 1)

      x = self.fc0(x)
      x = self.dropout1(x)
      x = F.relu(x)

      x = self.fc1(x)
      x = self.dropout2(x)
      x = F.relu(x)

      x = self.fc2(x)
      x = self.dropout3(x)
      x = F.relu(x)

      x = self.fc3(x)

      # Apply softmax to x
    #   output = F.log_softmax(x, dim=1)
      return x


class ModifiedCrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                  automodel_args:Dict = {}, default_activation_function = None, saved_path=None):

        print(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config, **automodel_args)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.model = SotaNet(model_name, config=self.config,**automodel_args)
        if saved_path is not None:
            self.model.load_state_dict(torch.load(saved_path+"/pytorch_model.pt"))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length
        self.best_threshold_stats = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        max_0 = 0
        max_1 = 0
        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())
                if len(text.split(" ")) > max_0 and idx == 0:
                    max_0 = len(text.split(" "))
                elif len(text.split(" ")) > max_1 and idx == 1:
                    max_1 = len(text.split(" "))
            labels.append(example.label)

        assert len(texts) == 2
        # tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        tokenized_0 = self.tokenizer(texts[0], padding=True, truncation=True, return_tensors="pt", max_length=512)
        tokenized_1 = self.tokenizer(texts[1], padding=True, truncation=True, return_tensors="pt", max_length=512)

        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized_0:
            tokenized_0[name] = tokenized_0[name].to(self._target_device)

        for name in tokenized_1:
            tokenized_1[name] = tokenized_1[name].to(self._target_device)

        return tokenized_0, tokenized_1, labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized_0 = self.tokenizer(texts[0], padding=True, truncation=True, return_tensors="pt", max_length=512)
        tokenized_1 = self.tokenizer(texts[1], padding=True, truncation=True, return_tensors="pt", max_length=512)

        for name in tokenized_0:
            tokenized_0[name] = tokenized_0[name].to(self._target_device)

        for name in tokenized_1:
            tokenized_1[name] = tokenized_1[name].to(self._target_device)

        return tokenized_0, tokenized_1

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()


        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            running_loss = []
            self.model.zero_grad()
            self.model.train()

            for features_0, features_1, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(features_0, features_1)
                        logits = activation_fct(model_predictions)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(features_0,features_1)
                    logits = activation_fct(model_predictions)
                    
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    running_loss.append(loss_value.item())
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback, np.mean(running_loss))
                    self.model.zero_grad()
                    self.model.train()
                    running_loss = []

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback, np.mean(running_loss))
                running_loss = []

        # TODO: choose only one of these thresholds
        evaluator.best_threshold_stats = evaluator.youden_j_threshold(evaluator.best_fpr, evaluator.best_tpr, evaluator.best_thresholds)
        # evaluator.best_threshold_stats = evaluator.distance_from_perf_threshold(evaluator.best_fpr, evaluator.best_tpr, evaluator.best_thresholds)
        
        plot_roc("Eval ROC", evaluator.best_fpr, evaluator.best_tpr, evaluator.best_threshold_stats, evaluator.best_roc_auc)
        self.best_threshold_stats = evaluator.best_threshold_stats

    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.
        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features_0, features_1 in iterator:
                model_predictions = self.model(features_0,features_1)
                logits = activation_fct(model_predictions)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback, train_loss):
        """Runs evaluation during the training"""
        if evaluator is not None:
            metrics = evaluator(self, output_path=output_path, epoch=epoch, steps=steps, train_loss=train_loss)
            score = metrics[evaluator.metric]
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                wandb.run.summary["eval/optimized_f1"] = metrics["f1"]
                wandb.run.summary["eval/optimized_f2"] = metrics["f2"]
                wandb.run.summary["eval/optimized_loss"] = -metrics["loss"]
                wandb.run.summary["eval/optimized_recall"] = metrics["recall"]
                wandb.run.summary["eval/optimized_precision"] = metrics["precision"]
                wandb.run.summary["eval/optimized_roc_auc"] = metrics["roc_auc"]
                wandb.run.summary["eval/optimized_accuracy"] = metrics["accuracy"]
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.tokenizer.save_pretrained(path)
        torch.save(self.model.state_dict(), path+"/pytorch_model.pt")

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)