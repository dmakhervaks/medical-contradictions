import logging
import os
import csv
from typing import List
from sentence_transformers.readers import InputExample
import numpy as np


logger = logging.getLogger(__name__)

class CESoftmaxAccuracyEvaluatorAdjusted:
    """
    This evaluator can be used with the CrossEncoder class.
    It is designed for CrossEncoders with 2 or more outputs. It measure the
    accuracy of the predict class vs. the gold labels.
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], name: str='', write_csv: bool = True):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.name = name

        self.csv_file = "CESoftmaxAccuracyEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Accuracy"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)


    # TODO: this doesn't work...
    def find_wrong_samples(self, pred_labels, output_path):
        fp = []
        fn = []
        wrong_samples = []
        mapping = {0:"contradiction",1:"non-contradiction"}

        for i, (pred_label, label) in enumerate(zip(pred_labels, self.labels)):
            # predicted contradiction
            if pred_label == 0 and label == 1:
                wrong_samples.append([self.sentence_pairs[i][0], self.sentence_pairs[i][1], mapping[pred_label], mapping[label]])
            elif pred_label == 1 and label == 0:
                wrong_samples.append([self.sentence_pairs[i][0], self.sentence_pairs[i][1], mapping[pred_label], mapping[label]])


        tsv_path = os.path.join(output_path, "wrong_samples.tsv")
        output_file_exists = os.path.isfile(tsv_path)
        with open(tsv_path,mode="a" if output_file_exists else 'w') as f:
            f.write("\t".join(("Sentence1","Sentence2","Predicted","True Label")) + "\n")
            for sample in wrong_samples:
                f.write("\t".join(sample) + "\n")
            
        return wrong_samples


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CESoftmaxAccuracyEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
        pred_labels = np.argmax(pred_scores, axis=1)

        assert len(pred_labels) == len(self.labels)

        acc = np.sum(pred_labels == self.labels) / len(self.labels)

        wrong_samples = self.find_wrong_samples(pred_labels,output_path)

        logger.info("Accuracy: {:.2f}".format(acc*100))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc])

        return acc