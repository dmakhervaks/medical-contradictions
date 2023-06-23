from pathlib import Path
import enum

import dataclasses


@enum.unique
class WordEmbeddings(enum.Enum):
    GloVe = 'glove'
    MIMIC = 'mimic'
    BioAsq = 'bioasq'
    WikiEn = 'wikien'
    WikiEnMIMIC = 'wikien_mimic'
    GloVeBioAsq = 'glove_bioasq'
    GloVeBioAsqMIMIC = 'glove_bioasq_mimic'


@enum.unique
class Models(enum.Enum):
    Simple = 'simple' # CBOW model
    InferSent = 'infersent'
    ESIM = 'esim'

@enum.unique
class Datasets(enum.Enum):
    All_C35_G25_WN_N10_SN_mednli = 'All_C35_G25_WN_N10_SN+mednli'
    All_C35_G25_WN_N10_SN_mednli_100 = 'All_C35_G25_WN_N10_SN+mednli_100'
    All_C35_G25_WN_N10_SN_mednli_cardio_100 = 'All_C35_G25_WN_N10_SN+mednli_cardio_100'
    All_C35_G25_WN_N10_SN_mednli_endocrinology_100 = 'All_C35_G25_WN_N10_SN+mednli_endocrinology_100'
    All_C35_G25_WN_N10_SN_mednli_female_reproductive_100 = 'All_C35_G25_WN_N10_SN+mednli_female_reproductive_100'
    All_C35_G25_WN_N10_SN_mednli_obstetrics_100 = 'All_C35_G25_WN_N10_SN+mednli_obstetrics_100'
    All_C35_G25_WN_N10_SN_mednli_surgery_100 = 'All_C35_G25_WN_N10_SN+mednli_surgery_100'
    Cardio_C35_G25_WN_N10_SN_cardio = 'Cardio_C35_G25_WN_N10_SN+cardio'
    Cardio_C35_G25_WN_N10_SN_positive_cardio = 'Cardio_C35_G25_WN_N10_SN+positive_cardio'
    mednli = 'mednli'
    mednli_100 = 'mednli_100'
    mednli_cardio_100 = 'mednli_cardio_100'
    mednli_surgery_100 = 'mednli_surgery_100'
    mednli_endocrinology_100 = 'mednli_endocrinology_100'
    mednli_female_reproductive_100 = 'mednli_female_reproductive_100'
    mednli_obstetrics_100 = 'mednli_obstetrics_100'
    cardio = 'cardio'
    positive_cardio = 'positive_cardio'


@dataclasses.dataclass
class Config:
    data_dir: Path

    dataset: Datasets

    model: Models = Models.InferSent
    word_embeddings: WordEmbeddings = WordEmbeddings.GloVe

    lowercase: bool = True
    max_len: int = 50
    hidden_size: int = 128
    dropout: float = 0.4
    trainable_embeddings: bool = False

    weight_decay: float = 0.0001
    learning_rate: float = 1e-3
    max_grad_norm: float = 5.0
    batch_size: int = 64
    nb_epochs: int = 30

    @property
    def mednli_dir(self) -> Path:
        # return self.data_dir.joinpath('mednli/')
        return self.data_dir.joinpath(f'{self.dataset.value}/')

    @property
    def cache_dir(self) -> Path:
        return self.data_dir.joinpath('cache/')

    @property
    def word_embeddings_dir(self) -> Path:
        return self.data_dir.joinpath('word_embeddings/')

    @property
    def models_dir(self) -> Path:
        return self.data_dir.joinpath(f'models/{self.dataset.value}')
