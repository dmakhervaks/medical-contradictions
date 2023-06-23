import json

from nltk import word_tokenize


def get_tokens(sentence_binary_parse):
    sentence = sentence_binary_parse \
        .replace('(', ' ').replace(')', ' ') \
        .replace('-LRB-', '(').replace('-RRB-', ')') \
        .replace('-LSB-', '[').replace('-RSB-', ']')

    tokens = sentence.split()

    return tokens


def read_mednli(filename):
    data = []
    print(f'Inside read_mednli for: {filename}')
    with open(filename, 'r') as f:
        for line in f:
            print(line)
            example = json.loads(line)
            premise = example['sentence1_binary_parse']
            hypothesis = example['sentence2_binary_parse']
            if isinstance(premise,str):
                premise = get_tokens(premise)
            if isinstance(hypothesis,str):
                hypothesis = get_tokens(hypothesis)

            assert type(premise) is list
            assert type(hypothesis) is list
            label = example.get('gold_label', None)
            data.append((premise, hypothesis, label))

    print(f'Name of file loaded: {filename}, {len(data)} examples')
    return data


def read_sentences(filename):
    with open(filename, 'r') as f:
        lines = [l.split('\t') for l in f.readlines()]

    input_data = [(word_tokenize(l[0]), word_tokenize(l[1]), None) for l in lines if len(l) == 2]
    return input_data


def load_mednli(cfg):
    # filenames = [
    #     'mli_train_v1.jsonl',
    #     'mli_dev_v1.jsonl',
    #     'mli_test_v1.jsonl',
    # ]
    dataset_name = cfg.dataset.value
    print("DATASET NAME")
    print(dataset_name)
    filenames = [
        f'{dataset_name}_train_stanford_parse.jsonl',
        f'{dataset_name}_dev_stanford_parse.jsonl',
        f'{dataset_name}_test_stanford_parse.jsonl',
    ]

    print(f'Dataset Name: {dataset_name}')
    filenames = [cfg.mednli_dir.joinpath(f) for f in filenames]

    mednli_train, mednli_dev, mednli_test = [read_mednli(f) for f in filenames]
    return mednli_train, mednli_dev, mednli_test
