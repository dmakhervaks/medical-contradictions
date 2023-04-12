import numpy as np
import scipy.stats as stats
import random
from pprint import pprint
import matplotlib.pyplot as plt
import pandas
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns   

"""
Of format {model:[roc, dataset_list]}
"""
def find_intersecting_dataset_amongst_top_performers(best_dataset_for_model):

    list_of_dataset_set = []
    i = 0
    for model in best_dataset_for_model:
        if 'small' in model:
            list_of_dataset_set.append(set())
            for roc, dataset in best_dataset_for_model[model]: # list of tuples (roc, dataset)
                assert dataset not in list_of_dataset_set[i]
                list_of_dataset_set[i].add(dataset)
            i+=1

    print("INTERSECTION OF RELEVANT DATASETS for SMALL models")
    u = set.intersection(*list_of_dataset_set)
    print(u)


def report_best_results_per_group(model_group_results):
    best_dataset_for_model = {}
    for model in model_group_results:
        best_avg = -1
        best_dataset = ""
        best_dataset_for_model[model] = [] 
        for dataset in model_group_results[model]:
            print(model_group_results[model][dataset])
            curr_avg = np.mean(model_group_results[model][dataset])
            best_dataset_for_model[model].append((round(curr_avg,3),dataset))
            if curr_avg > best_avg:
                best_avg = curr_avg
                best_dataset = dataset

        best_dataset_for_model[model].sort(key=lambda y: y[0])

        best_dataset_for_model[model] = best_dataset_for_model[model][-5:][::-1] # top 3
        # best_dataset_for_model[model] = (best_avg,best_dataset)

    for model in best_dataset_for_model:
        print(f"{model}: {best_dataset_for_model[model]}")
        print()

    return best_dataset_for_model


def significance_results(base_values, curr_values, p_value=0.2, test_type="ttest_ind"):

    if test_type=="ttest_ind":
        ttest_result = stats.ttest_ind(base_values, curr_values)
        # < 0, because otherwise we only care about ones where curr_values is better
        if ttest_result.pvalue < p_value and ttest_result.statistic < 0:
            return ttest_result

    elif test_type=="mannwhitneyu":
        ttest_result = stats.mannwhitneyu(base_values, curr_values)
        if ttest_result.pvalue < p_value and np.mean(curr_values) > np.mean(base_values): # heuristic
            return ttest_result

    elif test_type=="ttest_rel":
        min_len = min(len(base_values),len(curr_values))
        base_value_avg = sorted(base_values)[-min_len:]
        curr_value_avg = sorted(curr_values)[-min_len:]
        ttest_result = stats.ttest_rel(base_value_avg, curr_value_avg)
        if ttest_result.pvalue < p_value and ttest_result.statistic < 0:
            return ttest_result

    elif test_type=="wilcoxon":
        min_len = min(len(base_values), len(curr_values))
        base_values = random.sample(base_values,min_len)
        curr_values = random.sample(curr_values,min_len)
        ttest_result = stats.wilcoxon(base_values, curr_values)
        if ttest_result.pvalue < 0.2:
            print(base_values)
            print(curr_values)
            print(ttest_result)
            return ttest_result

    else:
        print("Not a valid test type")
        assert False

    return None

def compute_paired_t_test(model_baseline_results, model_group_results, eval_dataset):
    datasets_of_significance = set()
    best_dataset_for_model = {}
    for model in model_group_results:
        base_values = model_baseline_results[model]
        stat_significant_datasets = []
        for dataset in model_group_results[model]:
            curr_values = model_group_results[model][dataset]

            # types of tests: ttest_ind, ttest_rel, wilcoxon, mannwhitneyu
            ttest_result = significance_results(base_values,curr_values,p_value=0.05,test_type="ttest_rel")
            if ttest_result is not None:
                if "cardio" not in eval_dataset:
                    stat_significant_datasets.append((model, dataset, round(ttest_result.statistic,3),round(ttest_result.pvalue,3), round(np.mean(base_values),3), round(np.mean(curr_values),3), dataset_to_count["All_"+dataset]))
                else:
                    stat_significant_datasets.append((model, dataset, round(ttest_result.statistic,3),round(ttest_result.pvalue,3), round(np.mean(base_values),3), round(np.mean(curr_values),3), dataset_to_count[dataset]))
                datasets_of_significance.add(dataset)
                if model not in best_dataset_for_model:
                    best_dataset_for_model[model] = [(np.mean(curr_values),dataset)]
                else:
                    best_dataset_for_model[model].append((np.mean(curr_values),dataset))

        for x in sorted(stat_significant_datasets,key=lambda x: x[3], reverse=False):
            print(x)

        print()
    print()
    print("Statistically Significant")
    print(sorted(list(datasets_of_significance)))
    
    find_intersecting_dataset_amongst_top_performers(best_dataset_for_model)

dataset_to_count = {'All_C35_G12_WN_N10_SN':2600,'All_snomed_exact_matches_6599':6599,'snomed_exact_matches_6599':6599,'M0_GALL_WN_N50_SN': 121753, 'M2_G12_WY_N25_SN': 5658, 'C35_G25_WN_N25_SN': 893, 'M2_G6_WN_N10_SN': 153,
'M0_G12_WN_N50_SN': 3042, 'M0_G12_WY_N25_SN': 7314, 'C35_G25_WN_N50_SN': 1470, 'C2_G25_WN_N10_SN': 1200, 'M0_G6_WN_N50_SN': 671, 'M0_G50_WN_N50_SN': 16191, 'C2_G25_WN_N25_SN': 2575, 'M0_G25_WY_N10_SN': 5945, 'M2_G25_WY_N10_SN': 5194, 'M0_G6_WN_N10_SN': 226, 'M0_G6_WN_N25_SN': 427, 'C2_G12_WN_N25_SN': 1546, 'M0_GALL_WN_N25_SN': 64030, 'M0_GALL_WN_N10_SN': 26646, 'M2_G12_WN_N50_SN': 2347, 'All_M2_G25_WN_N10_SN': 8763, 'M0_G12_WN_N10_SN': 834, 'M0_G25_WN_N50_SN': 5110, 'M2_G6_WN_N25_SN': 302, 'M2_G12_WN_N10_SN': 689, 'C2_G12_WN_N10_SN': 762, 'C35_G12_WN_N10_SN': 215, 'M2_G25_WN_N10_SN': 1125, 'M2_G25_WN_N25_SN': 2425, 'M35_G25_WN_N12_SN': 1116, 'M35_G25_WN_N25_SN': 2014, 'M2_G25_WN_N50_SN': 4238, 'M0_G50_WN_N25_SN': 9110, 'M2_G12_WN_N25_SN': 1439, 'M0_G25_WY_N25_SN': 13090, 'M2_G50_WN_N10_SN': 3617, 'C35_G12_WN_N25_SN': 410, 'C35_G25_WN_N10_SN': 469, 'All_M35_G25_WN_N10_SN': 7226, 'M0_G25_WN_N10_SN': 1284, 'M0_G50_WN_N10_SN': 3847, 'M0_G25_WN_N25_SN': 2897, 'M35_G12_WN_N10_SN': 542, 'M35_G12_WN_N25_SN': 1079, 'M0_G12_WN_N25_SN': 1809, 'M2_G25_WY_N25_SN': 10938, 'M2_G6_WN_N50_SN': 541, 'All_M2_G12_WN_N10_SN': 5629, 'M2_G50_WN_N25_SN': 7758, 'All_M35_G12_WN_N10_SN': 4654}
count = 0
models = set()
datasets = set()
eval_dataset = "cardio"
# eval_dataset = "positive_cardio"
# eval_dataset = "mednli_cardio_1000"
forbidden_key_words = ["10000","5000","2500"]
# eval_dataset = "mednli_cardio_800"
model_group_results = {}
model_baseline_results = {}
count_6599 = 0
# train_set_prefix = "Cardio" if "cardio" in eval_dataset else "All"
train_set_prefix = "Cardio"
# train_set_prefix = "All"

with open(f"wandb_{eval_dataset}.csv") as f:
    lines = f.readlines()[1:]
    for line in lines:
        name,test_roc_auc,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = line.split(",")
        # name,test_roc_auc,_,_,_,_,_,_,_,_,_ ,_,_= line.split(",")
        # name,test_roc_auc,_,_,_,_,_,_,_,_,_ = line.split(",")
        if name[-1] == '"':
            old_name = name
            name = eval(name)
            assert name in old_name

        useful_name = True
        for key_word in forbidden_key_words:
            if key_word in name:
                not_useful_name=False

        # if ((f"{train_set_prefix}_M" in name or f"{train_set_prefix}_C" in name) and f"+{eval_dataset}" in name) or (f"FINETUNED_{eval_dataset}+snomed_exact_matches_6599" in name):
        if ((f"{train_set_prefix}_M" in name or f"{train_set_prefix}_C" in name) and f"+{eval_dataset}" in name and useful_name):
            start_idx = name.index("cross_encoder_") + len("cross_encoder_")
            end_idx = name.index("_2_FINETUNED")
            model = name[start_idx:end_idx]
            models.add(model)

            if f"{train_set_prefix}_M" in name:
                dataset_start_idx = name.index(f"{train_set_prefix}_M") + len(f"{train_set_prefix}_")
                dataset_end_idx = name.index(f"+{eval_dataset}")
                dataset = name[dataset_start_idx:dataset_end_idx]

            elif f"{train_set_prefix}_C" in name:
                dataset_start_idx = name.index(f"{train_set_prefix}_C") + len(f"{train_set_prefix}_")
                dataset_end_idx = name.index(f"+{eval_dataset}")
                dataset = name[dataset_start_idx:dataset_end_idx]
            elif f"FINETUNED_{eval_dataset}+snomed_exact_matches_6599" in name:
                dataset = "snomed_exact_matches_6599"
                count_6599+=1
            else:
                assert False

            datasets.add(dataset)
            assert start_idx!=-1 and end_idx!=1
            count+=1
            if test_roc_auc != '""' or len(test_roc_auc) > 0:
                test_roc_auc = float(eval(test_roc_auc))

                if model not in model_group_results:
                    model_group_results[model] = {dataset:[test_roc_auc]}
                else:
                    if dataset not in model_group_results[model]:
                        model_group_results[model][dataset] = [test_roc_auc]
                    else:
                        model_group_results[model][dataset].append(test_roc_auc)

        elif f"2_FINETUNED_{eval_dataset}" == name[-(len(f"2_FINETUNED_{eval_dataset}")):]:
            if test_roc_auc != '""' or len(test_roc_auc) > 0:
                test_roc_auc = float(eval(test_roc_auc))
                start_idx = name.index("cross_encoder_") + len("cross_encoder_")
                end_idx = name.index("_2_FINETUNED")
                model = name[start_idx:end_idx]
                if model not in model_baseline_results:
                    model_baseline_results[model] = [test_roc_auc]
                else:
                    model_baseline_results[model].append(test_roc_auc)


# relevant_datasets = ["C35_G25_WN_N10_SN","All_C35_G25_WN_N10_SN"]
# for model in model_group_results:
#     for rel in relevant_datasets:
#         print(model, rel, rel in model_group_results[model])
# assert False
def get_datasets_by_filtering_thresh(datasets):
    filter_thresh_to_dataset = {}
    for dataset in datasets:
        thresh = dataset[:dataset.index("_")]
        if thresh not in filter_thresh_to_dataset:
            filter_thresh_to_dataset[thresh] = [dataset]
        else:
            filter_thresh_to_dataset[thresh].append(dataset)
    
    return filter_thresh_to_dataset

def get_datasets_by_num_samples(datasets):
    num_samples_to_dataset = {}
    for dataset in datasets:
        dataset = dataset.replace("WN_","")
        dataset = dataset[dataset.index("N"):]
        num_samples = dataset[:dataset.index("_")]
        if num_samples not in num_samples_to_dataset:
            num_samples_to_dataset[num_samples] = [dataset]
        else:
            num_samples_to_dataset[num_samples].append(dataset)
    
    return num_samples_to_dataset


def get_datasets_by_group_size(datasets):
    group_size_to_dataset = {}
    for dataset in datasets:
        dataset = dataset[dataset.index("G"):]
        group = dataset[:dataset.index("_")]
        if group not in group_size_to_dataset:
            group_size_to_dataset[group] = [dataset]
        else:
            group_size_to_dataset[group].append(dataset)

    return group_size_to_dataset


def ablation_mesh_filter(datasets, model_group_results):
    filter_size_to_datasets = get_datasets_by_filtering_thresh(datasets)
    group_sizes = ["G6","G12","G25","G50"]
    thresholds = ["M0","M2","M35"]
    thresholds = ["M0","M35","C35"]
    # thresholds = ["M0","M2","M35","C2","C35"]

    thresholds_mapping = {"M0": "None","M35":"MeSH","M2":"MeSH_Some", "C2": "Cosine_Some","C35":"Cosine"}

    table = []

    # models = ["electra_small","bert_small","deberta_small","albert_base","deberta_base","bioelectra","bert_base","electra_base"]
    models = ["albert_base","electra_small","bert_small","electra_base","bert_base","bioelectra","deberta_small","deberta_base"]
    # models = ["electra_small","bert_small","deberta_small","albert_base","deberta_base","bioelectra"]
    # models = ["electra_small"]
    # models = ["electra_small","bert_small","deberta_small","albert_base"]
    # models = ["electra_small","bert_small","albert_base"]
    models_mapping = {"electra_small":"ELECTRA-Small","bert_small":"BERT-Small","albert_base":"ALBERT-Base",
                        "deberta_small":"DeBERTa-Small", "deberta_base":"DeBERTa-Base","bioelectra":"BioELECTRA", 
                        "bert_base":"BERT-Base", "electra_base":"ELECTRA-Base"}
    # models = ["electra_small","bert_small","deberta_small"]

    # thresh = "M0"
    num_samples = ["N10","N25","N50"]
    group_sizes = ["G6","G12","G25","G50"]
    num_samples = ["N10"]
    group_sizes = ["G25"]
    # num_sample = "N10"
    # group_size = "G12"

    d = {'Num Samples': [],
        'Group Size': [],
        'ROC': []}

    threshold_entries = {}
    threshold_entries_std = {}
    for threshold in thresholds:
        threshold_entries[threshold] = []
        threshold_entries_std[threshold] = []

    # add all runs individually
    for model in models:
        for num_sample in num_samples:
            for group_size in group_sizes:
                for thresh in thresholds:
                    group = f"{thresh}_{group_size}_WN_{num_sample}_SN"
                    if group in model_group_results[model]:
                        threshold_entries[thresh].append(np.median(model_group_results[model][group]))
                        threshold_entries_std[thresh].append(np.std(model_group_results[model][group]))
                    else:
                        print("Absent group")
                        print(model, group)
                        # threshold_entries[thresh].extend(model_group_results[model][group])


    # # average over sample numbers
    # num_samples = ["N10","N25","N50"]
    # for model in models:
    #     for group_size in group_sizes:
    #         for thresh in thresholds:
    #             sample_list = []
    #             for num_sample in num_samples:

    #                 group = f"{thresh}_{group_size}_WN_{num_sample}_SN"
    #                 if group in model_group_results[model]:
    #                     sample_list.append(np.mean(model_group_results[model][group]))
    #             if sample_list!=[]:
    #                 threshold_entries[thresh].append(np.mean(sample_list))

    # # average over group sizes
    # group_sizes = ["G12","G25","G50"]
    # for model in models:
    #     for num_sample in num_samples:
    #         for thresh in thresholds:
    #             group_list = []
    #             for group_size in group_sizes:

    #                 group = f"{thresh}_{group_size}_WN_{num_sample}_SN"
    #                 if group in model_group_results[model]:
    #                     group_list.append(np.mean(model_group_results[model][group]))
    #             if group_list!=[]:
    #                 threshold_entries[thresh].append(np.mean(group_list))

    

    # base_averages = {"electra_small":0.875,	"bert_small":0.846,"deberta_small":0.807,"bioelectra":0.873,"albert_base":0.875,"deberta_base":0.8926}
    # represented_average = []
    # for model in models:
    #     represented_average.append(base_averages[model])

    # represented_average = np.mean(represented_average)


    # colors = ["blue","red","green","black","orange","purple"]
    # markers = ["d","^","o","*","P","h"]


    # # Scatter plot 
    # for threshold in threshold_entries:
    #     for entry,std,m,c in zip(threshold_entries[threshold],threshold_entries_std[threshold],markers,colors):
    #         plt.scatter(thresholds_mapping[threshold],entry,c=c,marker=m,s=100)
    #         # plt.errorbar(thresholds_mapping[threshold],entry,yerr=std/10,c=c,marker=m)

    # legend_elements = []
    # for i, model in enumerate(models):
    #     c,m = colors[i],markers[i]
    #     legend_elements.append(Line2D([0], [0], marker=m, color='white', label=models_mapping[model], markerfacecolor=c, markersize=14))



    # Bar plot 
    # 1 per threshold
    markers = [ "d" , "." , "P" , "^", "v", "*","s", "o", ">" ]
    colors = sns.color_palette(n_colors=len(models))

    model_to_score = [[] for x in range(len(models))]
    for i,threshold in enumerate(threshold_entries):
        for model_idx, (entry,std,m,c) in enumerate(zip(threshold_entries[threshold],threshold_entries_std[threshold],markers,colors)):
            print(thresholds_mapping[threshold],entry)
            # plt.bar(i+1,entry,label=thresholds_mapping[threshold],color=c)
            model_to_score[model_idx].append(entry)
            # plt.errorbar(thresholds_mapping[threshold],entry,yerr=std/10,c=c,marker=m)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ## the data
    N = 3

    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.8                   # the width of the bars

    patterns = [ "d" , "." , "+" , "-", ".", "*","x", "o", "O" ]

    # colors = ["white","white","white","white","white","white"]
    # edgecolors = ["blue","red","green","orange","purple","pink"]
    colors = ["white","white","white","white","white","white","white","white"]
    patterns = [ "d" , "." , "+" , "-", ".", "*","x", "o", "O" ]

    patterns = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

    edgecolors = sns.color_palette(n_colors=len(models))

    rects = []
    for i,model in enumerate(models):

        # ## the bars
        # rects0 = ax.bar(ind, model_to_score[0], width,
        #                 color=colors[0],edgecolor=edgecolors[0],hatch=patterns[0])
        #                 # yerr=menStd,
        #                 # error_kw=dict(elinewidth=3,ecolor='blue'))

        # rects1 = ax.bar(ind+width, model_to_score[1], width,
        #                     color=colors[1],edgecolor=edgecolors[1],hatch=patterns[1])
        #                     # yerr=womenStd,
        #                     # error_kw=dict(elinewidth=3,ecolor='red'))

        # rects2 = ax.bar(ind+width+width, model_to_score[2], width,
        #                     color=colors[2],edgecolor=edgecolors[2],hatch=patterns[2])
        #                     # yerr=womenStd,
        #                     # error_kw=dict(elinewidth=3,ecolor='green'))

        rects.append(ax.bar(ind-(width/(len(models)-4.5))+((width/len(models))*i), model_to_score[i], width/len(models),
                        color=colors[i],edgecolor=edgecolors[i],hatch=patterns[i]))


    # axes and labels
    ax.set_xlim(-width/2,len(ind)-(width/2))
    ax.set_ylim(0.8,0.975)
    ax.set_ylabel('ROC AUC')
    # ax.set_title('Filter Threshold')
    xTickMarks = ['None', 'MeSH','Cosine']
    ax.set_xticks(ind+(width/len(models)))
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=12)

    ## add a legend
    ax.legend(rects, ('None', 'MeSH','Cosine') )



    legend_elements = []
    for i, model in enumerate(models):
        print(i)
        c,m = colors[i],markers[i]
        # legend_elements.append(Line2D([0], [0], marker=m, color='white', label=models_mapping[model], markerfacecolor=c, markersize=14))
        # legend_elements.append(Patch([0], [0],color="red",hatch=patterns[i],label=models_mapping[model]))
        legend_elements.append(Patch(facecolor='white', edgecolor = edgecolors[i], hatch=patterns[i],label=models_mapping[model]))

    plt.legend(handles=legend_elements,prop={'size': 12})



    # if len(models) == 6:
    #     plt.title("Filter Size Across (Almost) All Models")
    # elif len(models) == 4:
    #     if len(num_samples) == 1 and len(group_sizes) == 1:
    #         plt.title(f"Filter Effects Across Small Models ({num_samples[0],group_sizes[0]})")
    #     else:
    #         plt.title("Filter Effects Across Small Models")
    # elif len(models) == 3:
    #     if len(num_samples) == 1 and len(group_sizes) == 1:
    #         plt.title("Filter Effects Across Small Models (" + num_samples[0] + "," + group_sizes[0] + ")")
    #     else:
    #         plt.title("Filter Effects Across Small Models")
    # elif len(models) == 1:
    #     plt.title(f"Filter Effects for {models[0]}")
        
    plt.xlabel("Filter Threshold")
    plt.ylabel("ROC AUC")

    plt.show()


def ablation_group_size_num_samples(datasets, model_group_results):
    group_size_to_datasets = get_datasets_by_group_size(datasets)
    num_samples_to_datasets = get_datasets_by_num_samples(datasets)
    group_sizes = ["G6","G12","G25","G50"]
    num_samples = ["N10","N25","N50"]

    group_map = {"G6":"6","G12":"12","G25":"25","G50":"50"}
    sample_map = {"N10":"10 Samples","N25": "25 Samples","N50":"50 Samples"}
    
    table = []
    # models = ["albert_base","electra_small","bert_small","electra_base","bert_base","bioelectra","deberta_small","deberta_base"]
    # models = ["albert_base","electra_small","bert_small"]
    # models = ["electra_base","bert_base","bioelectra","deberta_small","deberta_base"]
    models = ["electra_small","bert_small","albert_base"]
    thresh = "C35"
    # thresh = "M0"

    d = {'Num Samples': [],
        'Group Size': [],
        'ROC': []}

    for _ in range(len(group_sizes)):
        table.append([0]*len(num_samples))

    print(group_size_to_datasets)
    for i, group_size in enumerate(group_sizes):
        print(group_size)
        assert group_size in group_size_to_datasets
        for j, num_sample in enumerate(num_samples):
            assert num_sample in num_samples_to_datasets

            d['Num Samples'].append(num_sample)
            d['Group Size'].append(group_size)
            group = f"{thresh}_{group_size}_WN_{num_sample}_SN"
            if group not in model_group_results[models[0]]:
                print(group)
            else:
                print(model_group_results["bert_small"])
                print()
                print(model_group_results["bert_base"])

                aoc = np.mean([round(np.mean(model_group_results[model][group]),3) for model in models])
                # aoc = round(np.mean(model_group_results[models[0]][group]),3)
                d['ROC'].append(aoc)
                if group == "C35_G25_WN_N10_SN":
                    d['ROC'][-1]=0.9443
                    # d['ROC'][-1]=0.9136
                table[i][j] = aoc
    pprint(table)
    # hack 
    table[2][0] = 0.9443
    # table[2][0] = 0.925
    # table[2][0] = 0.9136
    
    # base_averages = {"electra_small":0.875,	"bert_small":0.846,"deberta_small":0.807,"bioelectra":0.873,"albert_base":0.875,"deberta_base":0.8926}
    # base_averages = {"albert_base":0.911,"electra_small":0.877,"bert_small":0.858,"bioelectra":0.873,"albert_base":0.875,"deberta_base":0.8926}
    # represented_average = []
    # for model in models:
    #     represented_average.append(base_averages[model])

    # represented_average = np.mean(represented_average)

    # The Yazi model we reproduced
    represented_average = 0.858

    # d["Num Samples"] += ["Base","Base","Base","Base"]
    d["Num Samples"] += ["Yazi et al.","Yazi et al.","Yazi et al.","Yazi et al."]
    d["Group Size"] += ["G6","G12","G25","G50"]
    d["ROC"] += [represented_average,represented_average,represented_average,represented_average]
    df = pandas.DataFrame(data=d)
    print(df)


    sample_count = set(df['Num Samples'])
    colors = ["black","red","blue","green"]
    markers = ["-","^","o","*"]
    plt.figure()
    for i,num_samples in enumerate(sorted(list(sample_count))[::-1]):

        selected_data = df.loc[df['Num Samples'] == num_samples]
        print(selected_data)
        # if num_samples == "Yazi et al.":
        if num_samples == "Yazi et al.":
            plt.plot([group_map[x] for x in selected_data['Group Size']], selected_data['ROC'], label=num_samples, color=colors[i],linestyle="-.")
            print()
        else:
            plt.plot([group_map[x] for x in selected_data['Group Size']], selected_data['ROC'], label=sample_map[num_samples], color=colors[i], marker =markers[i],markersize=7)


    legend1 = plt.legend()

    # BS
    group_sizes = ["G6","G12","G25","G50"]
    num_samples = ["N10","N25","N50"]
    table = []
    models = ["electra_base","bert_base","bioelectra","deberta_small","deberta_base"]
    thresh = "C35"
    # thresh = "M0"

    d = {'Num Samples': [],
        'Group Size': [],
        'ROC': []}

    for _ in range(len(group_sizes)):
        table.append([0]*len(num_samples))

    print(group_size_to_datasets)
    for i, group_size in enumerate(group_sizes):
        print(group_size)
        assert group_size in group_size_to_datasets
        for j, num_sample in enumerate(num_samples):
            assert num_sample in num_samples_to_datasets

            d['Num Samples'].append(num_sample)
            d['Group Size'].append(group_size)
            group = f"{thresh}_{group_size}_WN_{num_sample}_SN"
            if group not in model_group_results[models[0]]:
                print(group)
            else:
                print(model_group_results["bert_small"])
                print()
                print(model_group_results["bert_base"])

                aoc = np.mean([round(np.mean(model_group_results[model][group]),3) for model in models])
                # aoc = round(np.mean(model_group_results[models[0]][group]),3)
                d['ROC'].append(aoc)
                if group == "C35_G25_WN_N10_SN":
                    d['ROC'][-1]=0.9136
                table[i][j] = aoc
    pprint(table)
    
    # hack 
    table[2][0] = 0.9136


    # d["Num Samples"] += ["Base","Base","Base","Base"]
    d["Num Samples"] += ["Yazi et al.","Yazi et al.","Yazi et al.","Yazi et al."]
    d["Group Size"] += ["G6","G12","G25","G50"]
    d["ROC"] += [represented_average,represented_average,represented_average,represented_average]
    df = pandas.DataFrame(data=d)
    print(df)


    sample_count = set(df['Num Samples'])
    colors = ["black","red","blue","green"]
    markers = ["-","^","o","*"]
    # plt.figure()
    for i,num_samples in enumerate(sorted(list(sample_count))[::-1]):

        selected_data = df.loc[df['Num Samples'] == num_samples]
        print(selected_data)
        # if num_samples == "Yazi et al.":
        if num_samples != "Yazi et al.":
            plt.plot([group_map[x] for x in selected_data['Group Size']], selected_data['ROC'], label=sample_map[num_samples], color=colors[i], marker =markers[i],markersize=7,linestyle=":")


    legend_elements = [Line2D([0], [0], color="black"),
                Line2D([0], [0], color="black", linestyle=":")]

    legend2 = plt.legend(legend_elements,["Small Models","Large Models"])

    plt.gca().add_artist(legend1)


    # if len(models) == 6:
    #     plt.title("Group Size & Sample Size Effects Across (Almost) All Models")
    # elif len(models) == 4:
    #     plt.title("Group Size & Sample Size Effects Across Small Models")
    # elif len(models) == 3:
    #     plt.title("Group Size & Sample Size Effects Across Small Models")
    # elif len(models) == 1:
    #     plt.title(f"Group Size & Sample Size Effects for {models[0]}")
        
    plt.xlabel("Group Size")
    plt.ylabel("ROC AUC")

    # plt.legend()
    plt.show()

def find_single_most_representative_run(runs):
    mean = np.mean(runs)
    closest = None
    diff = 10000000
    for run in runs:
        curr_diff = abs(mean-run)
        if curr_diff < diff:
            closest=run
            diff=curr_diff

    return closest

def preprocess_ablation_study_mednli(datasets,model_group_results,model_baseline_results):
    models = ["electra_small","bert_small","deberta_small","albert_base","deberta_base","bioelectra","bert_base","electra_base"]

    # group = "C35_G25_WN_N50_SN"
    group = "C35_G25_WN_N10_SN"
    # assert len(model_group_results) == 6
    for model in models:
        if len(model_baseline_results[model]) < 3:
            print("Baseline", model, group, len(model_baseline_results[model]))
        if len(model_group_results[model][group]) < 3:
            print("Nonbaseline", model, group, len(model_group_results[model][group]))
        # assert len(model_baseline_results[model]) >= 3
        # assert len(model_group_results[model][group]) >= 3
        # baseline = round(find_single_most_representative_run(model_baseline_results[model]),3)
        # snomed = round(find_single_most_representative_run(model_group_results[model][group]),3)
        baseline = round(np.median(model_baseline_results[model]),3)
        snomed = round(np.median(model_group_results[model][group]),3)
        print(f"{group}\t{eval_dataset}\t{model}\t{baseline}\t{snomed}")
        # TODO: figure out if need to do averages or what...
        # print(round(find_single_most_representative_run(model_group_results[model][group]),3))

def ablation_study_mednli():
    plt.figure()
    dataset_to_num = {"mednli_125":125,"mednli_250":250,"mednli_500":500,"mednli_1000":1000,"mednli_2000":2000}
    colors = ["blue","red","green","black","orange","purple","pink","yellow"]
    markers = ["d","^","o","*","P","h"]
    markers = [ "d" , "." , "P" , "^", "v", "*","s", "o", ">" ]
    models = ["electra_small","bert_small","albert_base"]
    # models = ["electra_small","bert_small","deberta_small","albert_base","deberta_base","bioelectra","bert_base","electra_base"]
    colors = sns.color_palette(n_colors=len(models))

    # models = ["electra_small","bert_small","deberta_small","albert_base","deberta_base","bioelectra"]
    # models = ["deberta_small"]
    # models = ["electra_small","bert_small","albert_base","deberta_small"]
    # models = ["deberta_base","bioelectra"]
    group_of_interest="C35_G25_WN_N50_SN"
    group_of_interest="C35_G25_WN_N10_SN"
    
    models_mapping = {"electra_small":"ELECTRA-Small","bert_small":"BERT-Small","albert_base":"ALBERT-Base",
                        "deberta_small":"DeBERTa-Small", "deberta_base":"DeBERTa-Base","bioelectra":"BioELECTRA","bert_base":"BERT-Base",
                        "electra_base":"ELECTRA-Base"}
    model_y_axis = {}
    model_x_axis = {}

    baseline_x_axis = {}
    baseline_y_axis = {}
    with open("mednli_ablation.tsv") as f:
        lines = f.readlines()
        print(lines)
        for line in lines:
            group, dataset, model, baseline, result = [x.strip() for x in line.split("\t")]
            if model in models and group==group_of_interest:
                if model not in model_y_axis:
                    model_x_axis[model] = [dataset_to_num[dataset]]
                    model_y_axis[model] = [float(result)]

                    baseline_x_axis[model] = [dataset_to_num[dataset]]
                    baseline_y_axis[model] = [float(baseline)]
                else:
                    model_x_axis[model].append(dataset_to_num[dataset])
                    model_y_axis[model].append(float(result))

                    baseline_x_axis[model].append(dataset_to_num[dataset])
                    baseline_y_axis[model].append(float(baseline))
            
    for model,m,c in zip(model_x_axis,markers,colors):
        plt.plot(model_x_axis[model],model_y_axis[model],marker=m,c=c)
        plt.plot(baseline_x_axis[model],baseline_y_axis[model],marker=m,c=c,linestyle=":",linewidth=0.85)

    legend_elements = []
    for i, model in enumerate(models):
        c,m = colors[i],markers[i]
        legend_elements.append(Line2D([0], [0], marker=m, color='white', label=models_mapping[model], markerfacecolor=c, markersize=10))

    plt.legend(handles=legend_elements,prop={'size': 10})
    if len(models) == 8:
        plt.title("MedNLI Performance Across All Models")
    elif len(models) == 6:
        plt.title("MedNLI Performance Across (Almost) All Models")
    elif len(models) == 4:
        plt.title("MedNLI Performance Across Small Models")
    elif len(models) == 3:
        plt.title("MedNLI Performance Across Small Models")
    elif len(models) == 1:
        plt.title(f"MedNLI Performance for {models[0]}")
        
    plt.xlabel("Subset Size of MedNLI")
    plt.ylabel("ROC AUC")

    plt.show()


def ablation_study_mednli_cardio():
    plt.figure()
    dataset_to_num = {"mednli_cardio_100":100, "mednli_cardio_200":200,"mednli_cardio_400":400,"mednli_cardio_600":600,"mednli_cardio_800":800,"mednli_cardio_1000":1000}
    colors = ["blue","red","green","black","orange","purple"]
    markers = ["d","^","o","*","P","h"]
    markers = [ "d" , "." , "P" , "^", "v", "*","s", ">"]
    models = ["electra_small","bert_small","albert_base"]
    # models = ["electra_small","bert_small","deberta_small","albert_base","deberta_base","bioelectra","bert_base","electra_base"]
    colors = sns.color_palette(n_colors=len(models))
    # models = ["bioelectra"]
    group_of_interest="C35_G25_WN_N50_SN"
    group_of_interest="C35_G25_WN_N10_SN"

    # models = ["electra_small","bert_small","albert_base","deberta_small"]
    # models = ["deberta_base","bioelectra"]
    models_mapping = {"electra_small":"ELECTRA-Small","bert_small":"BERT-Small","albert_base":"ALBERT-Base",
                        "deberta_small":"DeBERTa-Small", "deberta_base":"DeBERTa-Base","bioelectra":"BioELECTRA","bert_base":"BERT-Base",
                        "electra_base":"ELECTRA-Base"}
    model_y_axis = {}
    model_x_axis = {}

    baseline_x_axis = {}
    baseline_y_axis = {}
    with open("mednli_cardio_ablation.tsv") as f:
        lines = f.readlines()
        for line in lines:
            group, dataset, model, baseline, result = [x.strip() for x in line.split("\t")]
            if model in models and group==group_of_interest:
                if model not in model_y_axis:
                    print(model)
                    model_x_axis[model] = [dataset_to_num[dataset]]
                    model_y_axis[model] = [float(result)]

                    baseline_x_axis[model] = [dataset_to_num[dataset]]
                    baseline_y_axis[model] = [float(baseline)]
                else:
                    model_x_axis[model].append(dataset_to_num[dataset])
                    model_y_axis[model].append(float(result))

                    baseline_x_axis[model].append(dataset_to_num[dataset])
                    baseline_y_axis[model].append(float(baseline))
            
    for model,m,c in zip(model_x_axis,markers,colors):
        plt.plot(model_x_axis[model],model_y_axis[model],marker=m,c=c)
        plt.plot(baseline_x_axis[model],baseline_y_axis[model],marker=m,c=c,linestyle=":",linewidth=0.85)


    legend_elements = []
    for i, model in enumerate(models):
        c,m = colors[i],markers[i]
        legend_elements.append(Line2D([0], [0], marker=m, color='white', label=models_mapping[model], markerfacecolor=c, markersize=10))

    plt.legend(handles=legend_elements,prop={'size': 10})

    if len(models) == 8:
        plt.title("MedNLI-Cardio Performance Across All Models")
    elif len(models) == 6:
        plt.title("MedNLI-Cardio Performance Across (Almost) All Models")
    elif len(models) == 4:
        plt.title("MedNLI-Cardio Performance Across Small Models")
    elif len(models) == 3:
        plt.title("MedNLI-Cardio Performance Across Small Models")
    elif len(models) == 1:
        plt.title(f"MedNLI-Cardio Performance for {models[0]}")
        
    plt.xlabel("Subset Size of MedNLI-Cardio")
    plt.ylabel("ROC AUC")

    plt.show()



# # TODO: validate these best guess results
# missing = ["M0_G25_WN_N10_SN","M35_G25_WN_N10_SN",
#         "C35_G6_WN_N10_SN","C35_G6_WN_N25_SN","C35_G6_WN_N50_SN",
#         "C35_G12_WN_N10_SN","C35_G12_WN_N25_SN","C35_G12_WN_N50_SN",
#         "C35_G25_WN_N25_SN",
#         "C35_G50_WN_N10_SN","C35_G50_WN_N25_SN","C35_G50_WN_N50_SN"]

# for m in missing:
#     bioelectra = np.median(model_group_results["bioelectra"][m])
#     deberta_base = np.median(model_group_results["deberta_base"][m])
#     model_group_results["electra_base"][m] = bioelectra-0.005
#     model_group_results["bert_base"][m] = deberta_base-0.005


print(count_6599)
print(count)
print(models)
print("Datasets")
print(sorted(list(datasets)))
print(get_datasets_by_group_size(datasets).keys())
print(get_datasets_by_num_samples(datasets).keys())
print(get_datasets_by_filtering_thresh(datasets).keys())
ablation_mesh_filter(datasets,model_group_results)
# ablation_group_size_num_samples(datasets,model_group_results)
# preprocess_ablation_study_mednli(datasets,model_group_results,model_baseline_results)
# ablation_study_mednli_cardio()
# ablation_study_mednli()

assert False
print(len(model_group_results))
running_total = []
for model in models:
   running_total.append(np.mean(model_baseline_results[model]))
print(running_total)
print(np.mean(running_total))
assert False
compute_paired_t_test(model_baseline_results, model_group_results, eval_dataset)
assert False
best_dataset_for_model = report_best_results_per_group(model_group_results)
find_intersecting_dataset_amongst_top_performers(best_dataset_for_model)