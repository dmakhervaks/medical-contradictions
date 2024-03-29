import torch
import torch.utils.data
import numpy as np

from utils.vocab import Vocab


class NLIDataset(torch.utils.data.Dataset):
    THREE_LABEL_TO_ID = {"contradiction": 1, "entailment": 0, "neutral": 0, "non-contradiction":0} # TAKE INTO ACCOUNT POSITIVE CLASS
    TWO_LABEL_TO_ID = {"contradiction": 1, "non-contradiction":0}

    def __init__(self, mednli_data, lowercase=True, vocab=None, max_len=50, three_labels=True):
        premise, hypothesis, label, = zip(*mednli_data)

        if lowercase:
            premise = self._lowercase(premise)
            hypothesis = self._lowercase(hypothesis)

        premise = self._restrict_max_len(premise, max_len)
        hypothesis = self._restrict_max_len(hypothesis, max_len)

        if vocab is None:
            self.vocab = Vocab()

            self.vocab.add_documents(premise)
            self.vocab.add_documents(hypothesis)
        else:
            self.vocab = vocab

        self.premise = self._convert_to_numpy(premise)
        self.hypothesis = self._convert_to_numpy(hypothesis)

        if three_labels:
            self.label = [NLIDataset.THREE_LABEL_TO_ID[l] if l is not None else -1 for l in label]
        else:
            self.label = [NLIDataset.TWO_LABEL_TO_ID[l] if l is not None else -1 for l in label]

    def _pad(self, sent, max_len):
        sent = sent[:max_len]

        nb_pad = max_len - len(sent)
        if nb_pad > 0:
            sent = sent + [Vocab.PAD_TOKEN, ] * nb_pad

        return sent

    def _restrict_max_len(self, data, max_len):
        data = [self._pad(sent, max_len) for sent in data]
        return data

    def _lowercase(self, data):
        data = [[t.lower() for t in sent] for sent in data]
        return data

    def _convert_to_numpy(self, data):
        data = [
            np.array([
                self.vocab[token] if token in self.vocab else self.vocab[Vocab.UNK_TOKEN] for token in sent
            ], dtype=np.long)
            for sent in data
        ]
        return data

    def __getitem__(self, index):
        premise = self.premise[index]
        hypothesis = self.hypothesis[index]
        label = self.label[index]

        return (premise, hypothesis), label

    def __len__(self):
        return len(self.label)
