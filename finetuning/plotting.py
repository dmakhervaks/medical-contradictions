import matplotlib.pyplot as plt
from plotnine import *
import plotnine
import pandas as pd
import wandb

def plot_graph(csv_file, model_save_path, data_info, args, train_dataloader):
    csv_file = f"{model_save_path}/{csv_file}"
    if data_info['metric'] == "accuracy":
            plot_accuracy_over_time(csv_file, args.eval_steps, len(train_dataloader),model_save_path, data_info['metric'])
    elif data_info['metric'] == "recall":
        plot_metrics_over_time(csv_file, args.eval_steps, len(train_dataloader),model_save_path, data_info['metric'])


def plot_roc(name, fpr, tpr, opt_stats, roc_auc):
    plt.plot(fpr,tpr,)
    x, y, thresh, = opt_stats

    plt.plot(x, y,linewidth=2, marker ='.')
    plt.text(0.95,0,f"AUC: {str(roc_auc)[:6]}", fontsize='large')
    thresh = str(thresh)[:6]

    plt.annotate(f'Opt Thresh: {thresh}' , xy=(x, y), textcoords='data', xytext=(x-0.05, y+0.05))

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    wandb.log({name: plt})

    return plt

def plot_metrics_over_time(plot_file, eval_steps, steps_in_epoch, model_save_path, eval_metric):
    total_steps = 0
    epoch_num = 0
    with open(plot_file,"r") as f:
        x_axis = []
        y_axis_r, y_axis_p, y_axis_f, y_axis_l = [],[],[],[]
        lines = f.readlines()[1:]
        for line in lines:
            e,steps,recall, precision, f1, loss = [x.strip() for x in line.split(',')]
            steps = int(steps)
            recall,precision,f1 = float(recall),float(precision),float(f1)
            if int(steps) != -1:
                assert steps%eval_steps==0
                total_steps += eval_steps
            else:
                epoch_num+=1
                total_steps = steps_in_epoch*epoch_num
            
            x_axis.append(total_steps)
            
            y_axis_r.append(recall)
            y_axis_p.append(precision)
            y_axis_f.append(f1)


    plt.plot(x_axis, y_axis_r, label = "recall")
    plt.plot(x_axis, y_axis_p, label = "precision")
    plt.plot(x_axis, y_axis_f, label = "f1")
    # plt.plot(x_axis, y_axis_l, label = "loss")
    plt.legend()
    plt.title(model_save_path.replace('/',' '))
    plt.xlabel('steps')
    plt.ylabel(eval_metric)
    plt.savefig(f"{model_save_path}/eval_{eval_metric}")

def plot_accuracy_over_time(plot_file, eval_steps, steps_in_epoch, model_save_path, eval_metric):
    total_steps = 0
    epoch_num = 0
    with open(plot_file,"r") as f:
        x_axis = []
        y_axis = []
        lines = f.readlines()[1:]
        for line in lines:
            e,steps,acc = [x.strip() for x in line.split(',')]
            steps = int(steps)
            acc = float(acc)
            if int(steps) != -1:
                assert steps%eval_steps==0
                total_steps += eval_steps
            else:
                epoch_num+=1
                total_steps = steps_in_epoch*epoch_num
            x_axis.append(total_steps)
            y_axis.append(acc)

    plt.plot(x_axis, y_axis)
    plt.title(model_save_path.replace('/',' '))
    plt.xlabel('steps')
    plt.ylabel(eval_metric)
    plt.savefig(f"{model_save_path}/eval_{eval_metric}")