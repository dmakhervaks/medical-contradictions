import pandas as pd
import numpy as np
import scipy.stats
import random
import os 
from tabulate import tabulate


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov), aucs


def get_medians(one_files, two_files):    
    # slicing extracting middle elements
    one_files = one_files[len(one_files)//2]
    two_files = two_files[len(two_files)//2]

    return [one_files], [two_files]

def get_closest_to_average(one_files, two_files):    
    # slicing extracting middle elements
    one_avg = np.mean([float(x.split("_")[-1][:-4])for x in one_files])
    two_avg = np.mean([float(x.split("_")[-1][:-4]) for x in two_files])

    closest_one = None
    closest_one_diff = 1
    for one in one_files:
        score = float(one.split("_")[-1][:-4])
        if abs(score-one_avg) < closest_one_diff:
            closest_one = one
            closest_one_diff = abs(score-one_avg)

    closest_two = None
    closest_two_diff = 1
    for two in two_files:
        score = float(two.split("_")[-1][:-4])
        if abs(score-two_avg) < closest_two_diff:
            closest_two = two
            closest_two_diff = abs(score-two_avg)


    return [closest_one], [closest_two]

def remove_outliers(one_files, two_files):
    min_size = min(len(one_files),len(two_files))

    one_strt_idx = (len(one_files) // 2) - (min_size // 2)
    one_end_idx = (len(one_files) // 2) + (min_size // 2)

    two_strt_idx = (len(two_files) // 2) - (min_size // 2)
    two_end_idx = (len(two_files) // 2) + (min_size // 2)
    
    # slicing extracting middle elements
    one_files = one_files[one_strt_idx: one_end_idx + 1]
    two_files = two_files[two_strt_idx: two_end_idx + 1]

    # getting rid of edge cases
    min_size = min(len(one_files),len(two_files))
    one_files = one_files[:min_size]
    two_files = two_files[:min_size]


    return one_files, two_files


import re

models = ["albert_base","electra_small","bert_small","electra_base","bert_base","bioelectra","deberta_small","deberta_base","biogpt"]

target_group = "Cardio_C35_G25_WN_N10_SN"
base_folder_pairs = []
base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/cardio/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+cardio/"
base_folder_pairs.append((base_folder_one,base_folder_two))

base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/positive_cardio/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+positive_cardio/"
base_folder_pairs.append((base_folder_one,base_folder_two))

target_group = "All_C35_G25_WN_N10_SN"

base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/mednli_100/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+mednli_100/"
base_folder_pairs.append((base_folder_one,base_folder_two))

base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/mednli_cardio_100/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+mednli_cardio_100/"
base_folder_pairs.append((base_folder_one,base_folder_two))

base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/mednli_female_reproductive_100/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+mednli_female_reproductive_100/"
base_folder_pairs.append((base_folder_one,base_folder_two))

base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/mednli_endocrinology_100/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+mednli_endocrinology_100/"
base_folder_pairs.append((base_folder_one,base_folder_two))

base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/mednli_obstetrics_100/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+mednli_obstetrics_100/"
base_folder_pairs.append((base_folder_one,base_folder_two))

base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/mednli_surgery_100/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+mednli_surgery_100/"
base_folder_pairs.append((base_folder_one,base_folder_two))

base_folder_one = "/home/davem/Sentence_Transformers/ARR_Results/mednli/"
base_folder_two = f"/home/davem/Sentence_Transformers/ARR_Results/{target_group}+mednli/"
base_folder_pairs.append((base_folder_one,base_folder_two))

all_pred_labels = [None]*len(models)

# test_instances = []
# with open("cardio.tsv") as f:
#     lines = f.readlines()[1:]
#     for line in lines:
#         split,_,_,s1,s2,label = [x.strip() for x in line.split("\t")]
#         if split == "test":
#             test_instances.append((s1,s2,label))


for base_folder_one, base_folder_two in base_folder_pairs:
    theirs = [base_folder_one.split("/")[-2] + "."*(30-len(base_folder_one.split("/")[-2]))] 
    ours = [base_folder_two.split("/")[-1]]
    theirs_sanity = []
    ours_sanity = []
    all_p_values=[]
    max_aoc = 0
    for i,model in enumerate(models):
        # print(base_folder_one.split("/")[-1])
        output_log_file_one = base_folder_one+model+"/pred_scores.npy"
        output_log_file_two = base_folder_two+model+"/pred_scores.npy"


        one_files = [x for x in sorted(os.listdir(base_folder_one+model)) if "pred_scores_" in x]
        two_files = [x for x in sorted(os.listdir(base_folder_two+model)) if "pred_scores_" in x]

        for file in one_files:
            if "pred_scores_" in file:
                score = file.split("_")[-1][:-4]
                a = np.load(base_folder_one+model+"/"+file)

        for file in two_files:
            if "pred_scores_" in file:
                score = file.split("_")[-1][:-4]
                a = np.load(base_folder_two+model+"/"+file)

        # one_files,two_files = remove_outliers(one_files,two_files)
        one_files,two_files = get_medians(one_files,two_files)
        # one_files,two_files=get_closest_to_average(one_files,two_files)

        one_avg = np.array([])
        two_avg = np.array([])
        one_total = np.array([])
        two_total = np.array([])

        one_instances = 0
        one_scores = 0
        one_score_names = []
        for file in one_files:
            if "pred_scores_" in file:
                score = file.split("_")[-1][:-4]
                one_score_names.append(score)
                one_scores+=eval(score)
                one_instances+=1
                a = np.load(base_folder_one+model+"/"+file)
                one_avg = one_avg+a if one_avg.size else a
                one_total = np.vstack((one_total,a)) if one_total.size else a
        one_avg = one_avg/one_instances
        one_scores = one_scores/one_instances

        two_instances = 0
        two_scores = 0
        two_score_names = []
        for file in two_files:
            if "pred_scores_" in file:
                score = file.split("_")[-1][:-4]
                two_score_names.append(score)
                two_scores+=eval(score)
                two_instances+=1
                a = np.load(base_folder_two+model+"/"+file)
                two_avg = two_avg+a if two_avg.size else a
                two_total = np.vstack((two_total,a)) if two_total.size else a
        two_avg = two_avg/two_instances
        two_scores = two_scores/two_instances
        labels = np.load(base_folder_two+model+"/labels.npy")

        pred_labels_one = np.argmax(one_avg, axis=1)
        pred_labels_two = np.argmax(two_avg, axis=1)

        all_pred_labels[i] = pred_labels_two

        p_value, aucs = delong_roc_test(labels,one_total[:,1],two_total[:,1])
        p_value = (10**p_value)
        all_p_values.append(round(p_value[0][0],3))
        if round(aucs[0],3) > max_aoc:
            max_aoc = round(aucs[0],3)
        if round(aucs[1],3) > max_aoc:
            max_aoc = round(aucs[1],3)

        curr_theirs = str(round(aucs[0],3))+"*" if p_value <= 0.05 and aucs[0] > aucs[1] else str(round(aucs[0],3))
        curr_ours = str(round(aucs[1],3))+"*" if p_value <= 0.05 and aucs[1] > aucs[0] else str(round(aucs[1],3))
        
        if curr_theirs > curr_ours:
            curr_theirs+="..."
        else:
            curr_ours+="..."
        theirs.append(curr_theirs)
        ours.append(curr_ours)

        theirs_sanity.append(round(one_scores,3))
        ours_sanity.append(round(two_scores,3))


    for i in range(len(theirs)):
        if str(max_aoc) in theirs[i]:
            theirs[i]+= "<---"
        if str(max_aoc) in ours[i]:
            ours[i]+= "<---"

    print(tabulate([theirs, ours], headers=models))
    print()
