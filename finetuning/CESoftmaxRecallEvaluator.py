import logging
import os
import csv
from typing import List
import numpy as np
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import BinaryClassificationEvaluator
import numpy as np
from sklearn.metrics import RocCurveDisplay, accuracy_score, recall_score, precision_score, confusion_matrix, f1_score, fbeta_score, average_precision_score, roc_auc_score, auc, roc_curve
from torch import nn
import torch
import wandb
import math
import matplotlib.pyplot as plt
from plotting import plot_roc
from dython.model_utils import metric_graph
from pathlib import Path

logger = logging.getLogger(__name__)
RESULTS_DIRECTORY = # TODO: PLEASE ENTER A VALID RESULTS DIRECTORY

class CESoftmaxGeneralEvaluator:
    """
    This evaluator can be used with the CrossEncoder class.
    It is designed for CrossEncoders with 2 or more outputs. It measure the
    recall of the predict class vs. the gold labels.
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], name: str='', write_csv: bool = True, pos_class: int = 1, main_eval_loop: bool = False, steps_in_epoch: int = -1, test_loop: bool = False,
                batch_size: int = 32, metric: str = 'recall', best_threshold_stats: List[float] = None, fine_tuned_datasets: str="",base_model_name: str=""):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.name = name
        self.pos_class = pos_class
        self.csv_file = "CESoftmaxRecallEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "recall", "precision", "f1", "f2", "loss_value","roc_auc","accuracy"]
        self.write_csv = write_csv
        self.main_eval_loop = main_eval_loop
        self.steps_in_epoch = steps_in_epoch
        self.test_loop = test_loop
        self.batch_size = batch_size
        self.metric = metric
        self.best_roc_auc = -1
        self.best_roc_preds = []
        self.best_fpr = []
        self.best_tpr = []
        self.best_thresholds = []
        self.base_model_name=base_model_name.replace("cross_encoder_","")
        # print(fine_tuned_datasets)
        assert len(fine_tuned_datasets) == 1
        self.fine_tuned_datasets=fine_tuned_datasets[0]

        self.best_threshold_stats = best_threshold_stats
        if main_eval_loop:
            assert self.steps_in_epoch > 0

    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    
    def youden_j_threshold(self, fpr, tpr, thresholds):
        # Calculate how good each threshold is from our TPR and FPR. 
        # Our criteria is that the TPR is as high as possible and 
        # the FPR is as low as possible. We consider them equally important
        scores = np.array(tpr) - np.array(fpr)

        # Find the entry with the lowest score according to our criteria
        index_of_best_score = np.argmax(scores)
        best_threshold = thresholds[index_of_best_score]

        wandb.run.summary["eval/youden_threshold"] = best_threshold
        wandb.run.summary["eval/youden_fpr"] = fpr[index_of_best_score]
        wandb.run.summary["eval/youden_tpr"] = tpr[index_of_best_score]
        return [fpr[index_of_best_score], tpr[index_of_best_score],best_threshold]


    def distance_from_perf_threshold(self, fpr, tpr, thresholds):
        # Calculate how good each threshold is from our TPR and FPR. 
        # Our criteria is that the TPR is as high as possible and 
        # the FPR is as low as possible. We consider them equally important
        distances = np.sqrt(np.square(1-np.array(tpr)) + np.square(np.array(fpr)))

        # Find the entry with the lowest score according to our criteria
        index_of_best_score = np.argmin(distances)
        best_threshold = thresholds[index_of_best_score]

        wandb.run.summary["eval/distance_threshold"] = best_threshold
        wandb.run.summary["eval/distance_fpr"] = fpr[index_of_best_score]
        wandb.run.summary["eval/distance_tpr"] = tpr[index_of_best_score]
        return [fpr[index_of_best_score], tpr[index_of_best_score],best_threshold]


    def best_accuracy_threshold(self, pred_prob):
        pred_prob=np.array(pred_prob)
        for threshold in np.arange(0.3, 0.7, 0.02):
            y_predict_class = [1 if prob > threshold else 0 for prob in pred_prob[:,1]]
            print(f"{round(threshold,3)} Accuracy:", round(accuracy_score(self.labels, y_predict_class), 3))
            # print(f"{threshold} ROC AUC Score:", round(roc_auc_score(however_y, y_predict_class), 3))
            total_matches = np.sum(self.labels==y_predict_class)
            total_contras = np.sum(self.labels[self.labels==y_predict_class])
            print(f"percent: {total_contras/total_matches}")
            # print(f"totals -> matches: {total_matches} | contras: {total_contras} | percent: {total_contras/total_matches}")
        

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, train_loss:int = 0) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CESoftmaxRecallEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False)
        # pred_prob_flipped = []
        # pred_prob = []
        # for x,y in pred_scores:
        #     pred_prob_flipped.append([self.sigmoid(y),self.sigmoid(x)])
        #     pred_prob.append([self.sigmoid(x),self.sigmoid(y)])
        self.labels = np.array(self.labels)

        # TODO: I think it is the positive class...
        # roc_auc = roc_auc_score(self.labels^(self.labels&1==self.labels), np.array(pred_scores)[:,self.pos_class])
        # roc_auc_sanity = roc_auc_score(self.labels^(self.labels&1==self.labels), np.array(pred_prob_flipped)[:,1])
 
        average = 'macro'
        roc_auc = 0
        if pred_scores.shape[1] == 2:
            roc_auc = roc_auc_score(self.labels, np.array(pred_scores)[:,self.pos_class])
            fpr, tpr, thresholds = roc_curve(self.labels, np.array(pred_scores)[:,self.pos_class], pos_label=self.pos_class)
            average = 'binary'
            
            # print(pred_scores)
            # print(self.fine_tuned_datasets)            
            # print(self.name)
            # assert False
            pred_prob = []
            for x,y in pred_scores:
                pred_prob.append([self.sigmoid(x),self.sigmoid(y)])

            # fpr_pred, tpr_pred, thresholds_pred = roc_curve(self.labels, np.array(pred_prob)[:,self.pos_class], pos_label=self.pos_class)
            # roc_auc_sanity_sanity = auc(fpr,tpr)
            # roc_auc_sanity_pred = auc(fpr_pred,tpr_pred)
            
            # assert (fpr == fpr_pred).all()
            # assert (tpr == tpr_pred).all()
            # assert roc_auc == roc_auc_sanity_sanity and roc_auc == roc_auc_sanity_pred

            if roc_auc > self.best_roc_auc:
                self.best_roc_auc = roc_auc
                self.best_roc_preds = np.array(pred_scores)[:,self.pos_class]
                self.best_fpr = fpr
                self.best_tpr = tpr
                self.best_thresholds = thresholds

        # TODO: maybe want to use a threshold?
        pred_labels = np.argmax(pred_scores, axis=1)
        pred_labels_probs = np.argmax(pred_prob, axis=1)
        assert (pred_labels==pred_labels_probs).all()
        if self.test_loop and self.best_threshold_stats is not None:
            best_threshold = self.best_threshold_stats[2]
            pred_labels = [1 if x > best_threshold else 0 for x in pred_scores[:,self.pos_class]]

        loss_fct = nn.CrossEntropyLoss()

        assert len(pred_labels) == len(self.labels)

        #TODO: could be wrong...
        pred_labels = np.array(pred_labels)
        total_matches = np.sum(self.labels==pred_labels)
        total_contras = np.sum(self.labels[self.labels==pred_labels])
        loss_value = loss_fct(torch.from_numpy(pred_scores), torch.from_numpy(self.labels))
        recall = recall_score(self.labels, pred_labels,pos_label=self.pos_class, average=average)
        f1 = f1_score(self.labels, pred_labels,pos_label=self.pos_class, average=average)
        f2 = fbeta_score(self.labels, pred_labels, pos_label=self.pos_class, beta=2.0, average=average) # puts more weight on recall because of our motivation...
        precision = precision_score(self.labels, pred_labels,pos_label=self.pos_class, average=average)
        accuracy = accuracy_score(self.labels, pred_labels)
        print(f"THE ACTUAL ACCURACY/PERCENTAGE: {round(accuracy,3)}|{round(total_contras/total_matches,3)}")

        self.best_accuracy_threshold(pred_prob) # TODO: remove
        metrics = {"loss":-loss_value, "recall":recall, "precision":precision, "f1":f1, "f2":f2, "roc_auc":roc_auc, "accuracy":accuracy} # neg loss because it is opposite of other metrics

        logger.info("Recall: {:.2f}".format(recall*100))
        logger.info("Precision: {:.2f}".format(precision*100))
        logger.info("F1: {:.2f}".format(f1*100))
        logger.info("F2: {:.2f}".format(f2*100))
        logger.info("ROC_AUC: {:.2f}".format(roc_auc*100))
        logger.info("Accuracy: {:.2f}".format(accuracy*100))

        if self.main_eval_loop:
            if steps == -1 and epoch != -1:
                current_step = (epoch+1)*self.steps_in_epoch
            elif steps != -1 and epoch != -1:
                current_step = (epoch*self.steps_in_epoch)+steps # don't add one in this case
            else:
                assert False

            wandb.log({"eval/loss":loss_value,"eval/recall":recall, "eval/precision":precision, "eval/f1":f1, "eval/f2":f2, "eval/roc_auc":roc_auc, "eval/accuracy":accuracy,"train/loss":train_loss}, step=current_step)

        if self.test_loop:
            rounded_roc = round(roc_auc,3)
            base_dir = RESULTS_DIRECTORY+self.fine_tuned_datasets+"/"+self.base_model_name
            pred_score_file = base_dir+"/pred_scores_"+str(rounded_roc)+".npy"
            label_file = base_dir+"/labels.npy"

            if not os.path.exists(base_dir):
                Path(base_dir).mkdir(parents=True, exist_ok=False)
            np.save(pred_score_file,pred_scores)
            np.save(label_file,self.labels)
            wandb.log({"test/recall":recall,"test/precision":precision, "test/f1":f1, "test/f2":f2, "test/roc_auc":roc_auc, "test/accuracy":accuracy})
            self.best_threshold_stats = self.youden_j_threshold(self.best_fpr, self.best_tpr, self.best_thresholds)
            plot_roc("Test ROC", self.best_fpr, self.best_tpr, self.best_threshold_stats, roc_auc)
            np.save('predictions/pred_probs.npy', pred_prob)
            np.save("predictions/self_probs.npy", self.labels)
            
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, recall, precision, f1, f2, loss_value, roc_auc, accuracy])

        return metrics