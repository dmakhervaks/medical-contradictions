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
# from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from CESoftmaxRecallEvaluator import CESoftmaxGeneralEvaluator
from CrossEncoder import CrossEncoder
from SotaNet import ModifiedCrossEncoder
from datetime import datetime
from dataset_info import Datasets
from process_dataset import process_dataset
from plotting import plot_graph
import yaml,logging
import argparse
import wandb
import shutil

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout
"""
model_info:
  model_api_path: 'kamalkraj/bioelectra-base-discriminator-pubmed'
  model_name: 'cross_encoder_bioelectra'
  base: bioelectra
  num_labels: 2
  fine_tuned_on:
"""
def load_model_info(yaml_path):
    yaml_path = f"yaml_model_configs/{yaml_path}"
    with open(yaml_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    return config['model_info']

"""
(self, sentence_pairs: List[List[str]], labels: List[int], name: str='', write_csv: bool = True, pos_class: int = 1, main_eval_loop: bool = False, steps_in_epoch: int = -1, test_loop: bool = False,
                batch_size: int = 32, metric: str = 'recall'):
"""
# TODO: Figure out pos class!!!!
# def get_evaluator(metric, samples, data_name, split, steps_in_epoch=0, pos_class=0, main_eval_loop=False, test_loop=False, batch_size=32):
def get_evaluator(metric, samples, data_name, split, steps_in_epoch=0, pos_class=1, main_eval_loop=False, test_loop=False, batch_size=32, best_threshold_stats=None,fine_tuned_datasets=None,base_model_name=None):
    
    if metric in {'recall','loss','f1', 'f2', 'precision','roc_auc','accuracy'}:
        samples_evaluator = CESoftmaxGeneralEvaluator.from_input_examples(samples, name=f'{data_name}-{split}-{metric}', steps_in_epoch=steps_in_epoch, pos_class=pos_class, 
                                                                main_eval_loop=main_eval_loop, test_loop=test_loop, metric=metric, batch_size=batch_size,
                                                                best_threshold_stats=best_threshold_stats,fine_tuned_datasets=fine_tuned_datasets,base_model_name=base_model_name)
    else:
        assert False

    return samples_evaluator

def create_tags(base_model_name, eval_data_info, fine_tuned_datasets,args):
    tags=[base_model_name, f"optimized for {eval_data_info['metric']}", "sentence transformer"]
    tags.extend([f"tuned on {x}" for x in fine_tuned_datasets])

    for x in fine_tuned_datasets: 
        if 'snomed_contra_dataset_exact_matches' in x:
            tags.append("snomed_contra_dataset_exact_matches")
            break

    if args.freeze:
        tags.append("frozen")

    if args.sample_train_subset:
        tags.append(f"{args.sample_train_subset} TRAIN SAMPLES")

    return tags

def main(model, model_save_path, args, num_labels, eval_data_info, fine_tuned_datasets, model_info):
    logger.info("Read train dataset")
    POS_CLASS = 1

    best_threshold_stats=None

    train_samples, dev_samples, test_samples = process_dataset(eval_data_info, num_labels, args)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)

    eval_data_name = args.eval_data
    evaluator = get_evaluator(eval_data_info['metric'],dev_samples,eval_data_name,"dev", steps_in_epoch=len(train_dataloader), pos_class=POS_CLASS, main_eval_loop=True, 
                                batch_size=args.eval_batch_size,fine_tuned_datasets=fine_tuned_datasets,base_model_name=base_model_name) 

    if args.train:
        print(f"TRAINING MODEL! SIZE OF TRAINING SET: {len(train_samples)}")
        warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1) #10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))
        # Train the model
        
        """
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
        
        """
        if args.freeze:
            for param in model.model.base_model.parameters():
                param.requires_grad = False
        
        wandb.config.update({"warmup_steps": warmup_steps, "optimizer_class": "torch.optim.AdamW", "scheduler": 'WarmupLinear', "learning_rate":2e-5, "weight_decay":0.01, "steps_per_epoch":len(train_dataloader)})
        model.fit(train_dataloader=train_dataloader,
                evaluator=evaluator,
                epochs=args.num_epochs,
                evaluation_steps=args.eval_steps,
                warmup_steps=warmup_steps,
                output_path=model_save_path)

        best_threshold_stats = model.best_threshold_stats

        # TODO: get rid of local graph plotting if not necessary...
        # plot_graph(evaluator.csv_file, model_save_path, eval_data_info, args, train_dataloader)

    if args.eval:
        print(f"EVALUATING MODEL! SIZE OF DEV, TEST SET: {len(dev_samples), len(test_samples)}")
    
        if not args.sota:
            model = CrossEncoder(model_save_path, num_labels=num_labels)
        else:
            model = ModifiedCrossEncoder(model_info['model_api_path'], num_labels=num_labels,saved_path=model_save_path)
        dev_result = None
        test_result = None

        # # just the dev accuracy
        # dev_samples_evaluator =  get_evaluator(eval_data_info['metric'],dev_samples,eval_data_name,"dev")
        # dev_result = dev_samples_evaluator(model,output_path=output_path)  

        test_samples_evaluator = get_evaluator(eval_data_info['metric'],test_samples,eval_data_name,"test", pos_class=POS_CLASS, test_loop=True,
                                                batch_size=args.eval_batch_size, best_threshold_stats=best_threshold_stats,fine_tuned_datasets=fine_tuned_datasets,base_model_name=base_model_name) 
        test_result = test_samples_evaluator(model)

        print(f"{model_save_path}")
        print(f"{eval_data_info['metric']}, Dev Result: {dev_result}")
        print(f"{eval_data_info['metric']}, Test Result: {test_result}")

        if args.save == False and "important" not in model_save_path:
            shutil.rmtree(model_save_path, ignore_errors=True)

if __name__ == "__main__":
    # python training_nli_cross_encoder.py --yaml cross_encoder_bioelectra_3_class.yaml --train_data mednli --eval_data mednli --train --eval --save
    # python training_nli_cross_encoder.py --yaml cross_encoder_bioelectra.yaml --train_data cardio --eval_data cardio --train --eval --save
    # python training_nli_cross_encoder.py --yaml cross_encoder_distilroberta.yaml --train_data allnli --eval_data allnli --train --eval --save --train_batch_size 16 --eval_batch_size 32 --eval_steps 10000
    # python training_nli_cross_encoder.py --yaml cross_encoder_bioelectra.yaml --train_data cardio --eval_data cardio --train --eval --save --eval_steps 100 --num_epochs 5
    # python training_nli_cross_encoder.py --yaml cross_encoder_distilroberta_finetuned_allnli.yaml --train_data cardio --eval_data cardio --train --eval --save --eval_steps 100 --num_epochs 5
    # python training_nli_cross_encoder.py --yaml cross_encoder_bioelectra_finetuned_howevermoreover_c.yaml --train_data cardio --eval_data cardio --train --eval --save --eval_steps 100 --num_epochs 5 --train_batch_size 8 --eval_batch_size 32
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', dest='yaml', type=str, help='The yaml file')
    parser.add_argument('--train_data', dest='train_data', type=str, required=True, help='The train data file')
    parser.add_argument('--eval_data', dest='eval_data', type=str, required=False, help='The eval data file')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_steps', dest='eval_steps', type=int, required=False, default=500, help='The number of eval steps')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--train_batch_size', dest='train_batch_size', type=int, required=False, default=32, help='The batch size')
    parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, required=False, default=16, help='The batch size')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int, required=False, default=5, help='The batch size')
    parser.add_argument('--metric', dest='metric', type=str, required=False, help='The metric for which was are optimizing for')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--sample_train_subset', dest='sample_train_subset', type=int, required=False)
    parser.add_argument('--sota', action='store_true')

    args = parser.parse_args()

    # assume that we will always eval, sometime train
    assert args.eval == True

    model_info = load_model_info(args.yaml)

    # NOTE: num_labels is the number of classes the model has
    num_labels = model_info['num_labels']

    if not args.sota:
        model = CrossEncoder(model_info['model_api_path'], num_labels=num_labels)
    else:
        model = ModifiedCrossEncoder(model_info['model_api_path'], num_labels=num_labels)

    base_model_name = "sota_"+ model_info['base_model_name']if args.sota else model_info['base_model_name']
    base = model_info['base']

    fine_tuned_datasets = [] if model_info['fine_tuned_on'] is None else model_info['fine_tuned_on']

    eval_data_name = args.train_data if args.eval_data is None else args.eval_data

    train_data_names = args.train_data.split(",") # in case there are multiple datasets train on... meaning we mix them

    num_datasets_sampled = 0
    if args.sample_train_subset:
        for i,train_data_name in enumerate(train_data_names):
            if train_data_name==eval_data_name:
                train_data_names[i]+="_"+str(args.sample_train_subset)
                num_datasets_sampled+=1
        assert num_datasets_sampled==1

    train_data_names.sort() # to have consistent tagging
    train_data_name = '+'.join(train_data_names) # designates mixed dataset

    if args.train and args.sample_train_subset:
        fine_tuned_datasets.append(train_data_name) # this will also be finetuned on this dataset
    elif args.train and not args.sample_train_subset:
        fine_tuned_datasets.append(train_data_name)
    model_info['fine_tuned_on'] = fine_tuned_datasets

    
    model_save_path = f'{base}/{base_model_name}_{num_labels}_class/finetuned_{"-".join(fine_tuned_datasets)}/evaluated_{eval_data_name}/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    eval_data_info = Datasets[args.eval_data].value

    eval_data_info['metric'] = eval_data_info['metric'] if args.metric is None else args.metric

    tags = create_tags(base_model_name, eval_data_info, fine_tuned_datasets, args)

    group_name = f"Tuned on {eval_data_name}" if args.train else f"NOT tuned on {eval_data_name}"

    config={"epochs": args.num_epochs, 
            "train_batch_size":args.train_batch_size, 
            "eval_batch_size":args.eval_batch_size, 
            "eval_steps":args.eval_steps,
            "pretrained_on":model_info['pretrained_on'],
            "tuned_on":fine_tuned_datasets,
            "eval_metric":eval_data_info['metric']}

    wandb_project_name = f'EVALUATED_{eval_data_name}'
    
    if args.sample_train_subset:
        wandb_run_name = f'SENTENCETRANSFORMER_{base_model_name}_{num_labels}_FINETUNED_{"-".join(fine_tuned_datasets)}'
    else:
        wandb_run_name = f'SENTENCETRANSFORMER_{base_model_name}_{num_labels}_FINETUNED_{"-".join(fine_tuned_datasets)}'

    wandb.init(project=wandb_project_name, name=wandb_run_name, notes=f"/home/davem/Sentence_Transformers/{model_save_path}",tags=tags,group=group_name,config=config)
    wandb.config.update(model.config)
    if args.eval and not args.train: # just evaluating
        model_save_path = model_info['model_api_path']
    main(model, model_save_path, args, num_labels, eval_data_info, fine_tuned_datasets, model_info)
