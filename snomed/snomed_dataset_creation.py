import argparse
import os
import nltk
from nltk.corpus import stopwords
nltk.download('words')
from nltk.corpus import words
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import random
import copy
import numpy as np

def get_represented_phrases(base_path, version, map=None,only_single_phrases=False):
    # the two files below should be the same size
    set_of_represented_phrases = set()
    curr_file_name = base_path+"/all_files.tsv"
    curr_file_name_prime = base_path+"/all_files_with_mapping.tsv"
    total=0
    num_with_mult = 0
    with open(curr_file_name_prime, "w") as fw:
        with open(curr_file_name, "r") as f:
            lines = f.readlines()
            total = len(lines)
            for line in lines:
                sent, phrases = [x.strip() for x in line.split("\t")]
                phrases = phrases.split(",")
                # exclude sentences with extra sentences
                if only_single_phrases and len(phrases) > 1:
                    num_with_mult+=1
                    continue
                
                phrases_prime = []
                for phrase in phrases:
                    if version == 2 and map is not None:
                        phrase = map[phrase]
                        phrases_prime.append(phrase)
                    set_of_represented_phrases.add(phrase)
                arr_str = ",".join(phrases_prime)
                fw.write(f"{line.strip()}\t{arr_str}\n")

    print(num_with_mult/total)
    print(num_with_mult)

    return set_of_represented_phrases


"""
Since we check for all phrases, we will get rid of redundant ones...
Maybe we should just keep the longest phrase in general...
"""
def check_if_phrases_are_redundant(phrases):
    phrases_list = list(phrases)
    new_phrases = []
    for i in range(len(phrases_list)):
        is_substring = False
        p1 = phrases_list[i]
        for j in range(len(phrases_list)):
            if i != j:
                p2 = phrases_list[j]
                if p1 in p2: # currently checking if p1 is substring of anything... if it isn't then we can add it 
                    is_substring=True
                    break
        if not is_substring:
            new_phrases.append(p1)

    return new_phrases


"""
Returns statistics about how often certain phrases occur
"""
def get_statistics_about_data(base_path):
    phrase_to_frequency_dict = {}
    curr_file_name = base_path+"/all_files.tsv"

    with open(curr_file_name, "r") as f:
        lines = f.readlines()
        dict_of_mults = {}
        for line in lines:
            sent,pmid,title,mesh_headings,keywords,phrases = [x.strip() for x in line.split("\t")]
            phrases = set([x.strip() for x in phrases.split(",")])

            # exclude sentences with extra phrases, or could include them if the same label, but then must choose a "representative phrase"
            if len(phrases) > 1:
                phrase_list = list(phrases)
                phrase_list.sort()
                phrase_tuple = tuple(phrase_list)
                if phrase_tuple in dict_of_mults:
                    dict_of_mults[phrase_tuple].append(sent)
                else:
                    dict_of_mults[phrase_tuple] = [sent]

            phrases_list = check_if_phrases_are_redundant(phrases)

            for phrase in phrases_list:
                if phrase in phrase_to_frequency_dict:
                    phrase_to_frequency_dict[phrase]+=1
                else:
                    phrase_to_frequency_dict[phrase]=1

    print(f"Len of mults: {len(dict_of_mults)}")

    return phrase_to_frequency_dict, dict_of_mults


def get_phrase_mapping(map_file_name):
    phrase_mapping = {}
    with open(map_file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            phrase1, phrase2 = [x.strip() for x in line.split(",")]
            phrase_mapping[phrase1]=phrase2

    return phrase_mapping


def get_plia_labeled_phrases():
    set_of_snomed_phrases = set()
    set_of_snomed_phrases_no_nums = set()
    list_of_phrase_pairs = []
    list_of_phrase_pairs_no_nums = []
    labels = []
    labels_without_nums = []

    with open ("plia_labeled.tsv","r") as f:
        lines = f.readlines()[1:] #skip the header
        for line in lines:
            phrase1,phrase2,_,_,label,_ = line.split("\t")
            phrase1 = phrase1.strip().lower()
            phrase2 = phrase2.strip().lower()
            set_of_snomed_phrases.add(phrase1)
            set_of_snomed_phrases.add(phrase2)
            list_of_phrase_pairs.append((phrase1,phrase2))

            word_tokens_1 = phrase1.split(" ")
            word_tokens_2 = phrase2.split(" ")

            if int(label) > 5:
                labels.append("contradiction")
            else:
                labels.append("non-contradiction")

            contains_full_num_1,contains_full_num_2 = False,False
            for word in word_tokens_1:
                if '+' in word:
                        contains_full_num = True
                        break
                try:
                    attempt = int(word)
                    contains_full_num_1 = True
                    break
                except:
                    pass

            for word in word_tokens_2:
                if '+' in word:
                        contains_full_num = True
                        break
                try:
                    attempt = int(word)
                    contains_full_num_2 = True
                    break
                except:
                    pass

            if not contains_full_num_1:
                set_of_snomed_phrases_no_nums.add(phrase1)
            if not contains_full_num_2:
                set_of_snomed_phrases_no_nums.add(phrase2)
            if not contains_full_num_1 and not contains_full_num_2:
                list_of_phrase_pairs_no_nums.append((phrase1,phrase2))
                labels_without_nums.append(labels[-1])

    return set_of_snomed_phrases, list_of_phrase_pairs, set_of_snomed_phrases_no_nums, list_of_phrase_pairs_no_nums, labels, labels_without_nums

"""
Returns the list of phrase-pairs and their corresponding labels
"""
def populated_no_num_snomed_phrase_pairs(keep_numbers=False):
    list_of_phrase_pairs = []
    phrase_pairs_to_label = {}
    labels = []
    mapping = {"1":"contradiction", "0":"non-contradiction"}
    pairs_path = f"snomed_pairs_auto_labeled.csv"

    problem_count=0
    with open (pairs_path,"r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            _, phrase1,_, phrase2,label = [x.strip().lower() for x in line.split(",")]
                                
            word_tokens_1 = phrase1.split(" ")
            word_tokens_2 = phrase2.split(" ")
            contains_full_num = False

            for word in word_tokens_1:
                if '+' in word:
                    contains_full_num = True
                    break
                try:
                    attempt = int(word)
                    contains_full_num = True
                    break
                except:
                    pass

            for word in word_tokens_2:
                if '+' in word:
                    contains_full_num = True
                    break
                try:
                    attempt = int(word)
                    contains_full_num = True
                    break
                except:
                    pass

            if keep_numbers or not contains_full_num :
                if (phrase1,phrase2) in phrase_pairs_to_label or (phrase2,phrase1) in phrase_pairs_to_label:
                    problem_count+=1
                else:
                    phrase_pairs_to_label[(phrase1,phrase2)] = mapping[label]
                    list_of_phrase_pairs.append((phrase1,phrase2))
                    labels.append(mapping[label])
            else:
                print(phrase1,phrase2)

    return list_of_phrase_pairs, labels, phrase_pairs_to_label

def populate_snomed_phrases(path):
    set_of_snomed_phrases = set()
    with open (path,"r") as f:
        phrases = f.readlines()
        for phrase in phrases:
            set_of_snomed_phrases.add(phrase.strip())
    return set_of_snomed_phrases


"""
MISC functions
"""
def create_large_file(base_path):
    with open(base_path+"/all_files.tsv","w") as fw:
        fw.write(f"split\tdataset\tfilename\tsentence1\tsentence2\tlabel\n")
        for i in range(1,1115):
            file_name = base_path+"/{:04}.tsv".format(i)
            if os.path.isfile(file_name):
                with open(file_name, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        fw.write(line)

"""
Write to a file the number of counts of each pair in our dataset
Helps understand the distribution we're dealing with...
"""
def create_phrase_pair_counts_file(list_of_phrase_pairs, phrase_to_frequency_dict, file_name):
    list_of_represented_phrase_pairs = []
    for idx, (phrase1,phrase2) in enumerate(list_of_phrase_pairs):
        if phrase1 in phrase_to_frequency_dict and phrase2 in phrase_to_frequency_dict:
            curr_combo_count = phrase_to_frequency_dict[phrase1]*phrase_to_frequency_dict[phrase2]
            list_of_represented_phrase_pairs.append((phrase1,phrase2,curr_combo_count))

    # sort by increasing count
    list_of_represented_phrase_pairs = sorted(list_of_represented_phrase_pairs, key=lambda x: x[2])

    with open(file_name,"w") as fw:
        for phrase1,phrase2,curr_combo_count in list_of_represented_phrase_pairs:
            fw.write(f"{phrase1}\t{phrase2}\t{curr_combo_count}\n") 


"""
Gets the counts for the pairs of phrases
list_of_phrase_pairs is a list of all possible pairs
phrase_to_frequency_dict is already calculated on the particular dataset level
NOTE: remember that there will be duplicate counts here... want to take least frequent
pairs into account when creating dataset
RETURNS: sorted ascending by counts list in format of [(phrase1, phrase2, count)]
"""
def get_represented_phrase_pair_counts(list_of_phrase_pairs, heuristic_labels_without_nums, phrase_to_frequency_dict):
    list_of_represented_phrase_pairs = []
    for idx, (phrase1,phrase2) in enumerate(list_of_phrase_pairs):
        if phrase1 in phrase_to_frequency_dict and phrase2 in phrase_to_frequency_dict:
            curr_combo_count = phrase_to_frequency_dict[phrase1]*phrase_to_frequency_dict[phrase2]
            label = heuristic_labels_without_nums[idx]
            list_of_represented_phrase_pairs.append((phrase1,phrase2,curr_combo_count,label))

    # sort by increasing count
    list_of_represented_phrase_pairs = sorted(list_of_represented_phrase_pairs, key=lambda x: x[2])

    return list_of_represented_phrase_pairs


def create_sampled_dataset(dict_of_contradictions, dict_of_non_contradictions, represented_phrase_pair_counts, sample_factor):
    list_of_contradictions, list_of_non_contradictions = [], []
    for p1,p2,count,label in represented_phrase_pair_counts:
        if (p1,p2) in dict_of_non_contradictions:
            sentence_pairs = random.sample(dict_of_non_contradictions[(p1,p2)],min(len(dict_of_non_contradictions[(p1,p2)]),sample_factor))
            sentence_pairs = [("non-contradiction",x) for x in sentence_pairs]
            list_of_non_contradictions.extend(sentence_pairs)
        elif (p1,p2) in dict_of_contradictions:
            sentence_pairs = random.sample(dict_of_contradictions[(p1,p2)],min(len(dict_of_contradictions[(p1,p2)]),sample_factor))
            sentence_pairs = [("contradiction",x) for x in sentence_pairs]
            list_of_contradictions.extend(sentence_pairs)
        elif p1 in p2 or p2 in p1:
            continue
        else:
            assert False

    return list_of_contradictions, list_of_non_contradictions


def get_dict_of_phrases_to_sentences(base_path):
    curr_file_name = base_path+"/all_files.tsv" 

    dict_of_phrases_to_sentences = {}
    set_of_various_mesh_headings = set()
    with open(curr_file_name, "r") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            sent,pmid,title,mesh_headings,keywords,phrases = [x.strip() for x in line.split("\t")]
            # sent, phrases = [x.strip() for x in line.split("\t")]
            phrases = phrases.split(",")

            mesh_headings_set = set([x.lower().strip() for x in mesh_headings.split(";")])
            mesh_headings_list = list(mesh_headings_set)
            mesh_headings_list.sort()
            mesh_headings_tuple = tuple(mesh_headings_list)

            phrases_list = check_if_phrases_are_redundant(phrases)

            for phrase in phrases_list:
                if phrase in dict_of_phrases_to_sentences:
                    dict_of_phrases_to_sentences[phrase].append((sent,pmid,mesh_headings_tuple))
                else:
                    dict_of_phrases_to_sentences[phrase] = [(sent,pmid,mesh_headings_tuple)]

            set_of_various_mesh_headings.add(mesh_headings_tuple)

    return dict_of_phrases_to_sentences 


def determine_if_sentences_have_valid_relationship(phrase1, sentence1, pmid1, mesh_headings1, phrase2, sentence2, pmid2, mesh_headings2, thresh):

    cosine_sim = basic_cosine_similarity(sentence1,sentence2)
    if cosine_sim > thresh:
        return True
    else:
        return False


def basic_cosine_similarity(sent1,sent2):
    # tokenization
    X_list = word_tokenize(sent1.lower())
    Y_list = word_tokenize(sent2.lower())

    # sw contains the list of stopwords
    l1 =[];l2 =[]

    # remove stop words from the string
    X_set = {w for w in X_list }
    Y_set = {w for w in Y_list }

    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)

    return cosine



"""
list_of_pairs: literally all list of pairs (usually without nums - 748)
phrase_to_frequency_dict: mapping phrases to num of times they appear
represented_phrase_pair_counts: mapping phrase-pairs to num of times they appear
TODO: map?
"""
def create_dataset_of_potentially_contradictory_sentences(base_path, represented_phrase_pair_counts, thresh, sample_number=50, use_mesh=False):
    curr_file_name = base_path+"/all_files.tsv" 
    new_file_name = base_path+f"/SNOMED_dataset.tsv"

    total_mesh_terms = set()

    # labels - {"contradiction","non-contradiction"}
    # {lowercased phrase:[sentence1, sentence2, ...]}
    dict_of_phrases_to_sentences = {}
    split = "train"

    dict_of_phrases_to_sentences = get_dict_of_phrases_to_sentences(base_path)

    list_of_contradictions = []
    list_of_non_contradictions = []

    blunder_num = 0
    
    number_of_pairs_actually_represented = set()

    used_sentence_pairs = set()
    total_unsucessful_pairs = 0
    total_succesful_pairs = 0

    for idx, (phrase1,phrase2,_,label) in enumerate(represented_phrase_pair_counts):
        if phrase1 in dict_of_phrases_to_sentences and phrase2 in dict_of_phrases_to_sentences:    
            assert phrase1 != phrase2
            sentences1 = dict_of_phrases_to_sentences[phrase1]  

            sentences1_set = set(sentences1)
            sentences2 = dict_of_phrases_to_sentences[phrase2]
            sentences2_set = set(sentences2)

            # number of combos gets used here instead, so don't need it above
            num_combos = len(sentences1_set)*len(sentences2_set)

            iterations = num_combos
            curr_iter = 0
            success_sample_count = 0
            unsuccess_retry_count = 0 # in the case that we deal with MeSH terms, we may need to resample
            max_retries = 3000
            while curr_iter < iterations and unsuccess_retry_count < max_retries and success_sample_count < sample_number:

                if len(sentences1_set) <= 0 or len(sentences2_set) <= 0:
                    break
                
                sentence1, pmid1, mesh_headings1 = random.sample(sentences1_set,1)[0]
                sentence2, pmid2, mesh_headings2 = random.sample(sentences2_set,1)[0]

                if sentence1.strip() == sentence2.strip():
                    if len(sentences1_set) > len(sentences2_set):
                        sentences1_set.remove((sentence1, pmid1, mesh_headings1))
                    else:
                        sentences2_set.remove((sentence2, pmid2, mesh_headings2))
                    continue

                if (sentence1, sentence2) in used_sentence_pairs or (sentence2, sentence1) in used_sentence_pairs:
                    if len(sentences1_set) > len(sentences2_set):
                        sentences1_set.remove((sentence1, pmid1, mesh_headings1))
                    else:
                        sentences2_set.remove((sentence2, pmid2, mesh_headings2))
                    continue

                used_sentence_pairs.add((sentence1, sentence2))
                used_sentence_pairs.add((sentence2, sentence1))
                
                valid_relation=True
                
                if use_mesh:
                    valid_relation = determine_if_sentences_have_valid_relationship(phrase1, sentence1, pmid1, mesh_headings1, phrase2, sentence2, pmid2, mesh_headings2, thresh)
                if valid_relation:
                    success_sample_count+=1
                    unsuccess_retry_count=0
                    if label == "contradiction":
                        list_of_contradictions.append(("contradiction", (sentence1, sentence2)))
                    elif label == "non-contradiction":
                        list_of_non_contradictions.append(("non-contradiction", (sentence1, sentence2)))
                    number_of_pairs_actually_represented.add((phrase1,phrase2))
                    assert (phrase2,phrase1) not in number_of_pairs_actually_represented
                else:
                    unsuccess_retry_count+=1
                curr_iter+=1

            if unsuccess_retry_count >= max_retries or curr_iter >= iterations:
                total_unsucessful_pairs+=1
            else:
                total_succesful_pairs+=1

    all_instances = list_of_contradictions + list_of_non_contradictions

    all_instances_set = {}
    all_instances_prime_set = set()
    used_sentence_pairs = set()
    used_sentence_pairs_prime = set()
    for label, instance in all_instances:
        sent1, sent2 = instance
        used_sentence_pairs.add((sent1.strip(),sent2.strip()))
        used_sentence_pairs.add((sent2.strip(),sent1.strip()))
        if sent1.strip() in all_instances_set:
            all_instances_set[sent1].append((label,instance))
        else:
            all_instances_set[sent1] = [(label,instance)]

        if sent2.strip() in all_instances_set:
            all_instances_set[sent2].append((label,instance))
        else:
            all_instances_set[sent2] = [(label,instance)]

    print(f"all instances len: {len(all_instances)}")
    print(f"used_sentence_pairs len: {len(used_sentence_pairs)}")
    print(f"used_sentence_pairs len: {len(used_sentence_pairs_prime)}")
    print(f"set of all instances: {len(all_instances_set)}", f"set of all instances prime: {len(all_instances_prime_set)}")
    print(f"blunder_num: {blunder_num}")
    print(f"Length of total mesh terms: {len(total_mesh_terms)}")
    count = 0
    random.shuffle(all_instances)

    with open(new_file_name,"w") as fw:
        fw.write("\t".join(["split","dataset","filename","sentence1","sentence2","label"])+"\n")
        for label, instance in all_instances:
            sentence1,sentence2 = instance
            fw.write("\t".join([split,new_file_name,curr_file_name,sentence1,sentence2,label])+"\n")
            count+=1

    return dict_of_phrases_to_sentences

def evenly_class_sample_created_file(file):
    new_file = file.split(".tsv")[0]+"_even_sampled"+".tsv"

    contra_lines = []
    non_contra_lines = []
    header = None
    with open(file) as f:
        lines = f.readlines()
        header = lines[0]
        for line in lines[1:]:
            split,dataset,curr_file_name,sentence1,sentence2,label = [x.strip() for x in line.split("\t")]
            if label == 'contradiction':
                contra_lines.append(line)
            else:
                non_contra_lines.append(line)

    min_len = min(len(contra_lines),len(non_contra_lines))
    contra_lines = random.sample(contra_lines,min_len)
    non_contra_lines = random.sample(non_contra_lines,min_len)
    all_instances = contra_lines+non_contra_lines
    random.shuffle(all_instances)
    with open(new_file,"w") as fw:
        fw.write(header)
        for line in all_instances:
            fw.write(line)


"""
The goal is to even these out, by sampling randomly with replacement 
to make the lists equal length
{word: [[contra_list pairs], [non_contra_list pairs]]}
"""
def even_out_phrase_distribution_via_words(word_to_phrase_pairs):

    new_represented_phrase_pairs = []

    for word,lists in word_to_phrase_pairs.items():
        len1 = len(lists[0])
        len2 = len(lists[1])
        new_represented_phrase_pairs+=lists[0]
        new_represented_phrase_pairs+=lists[1]

        # lack of non-contras
        if len1 != 0 and len1 < len2:
            new_phrases = random.choices(lists[0], k=len2-len1)
            new_represented_phrase_pairs += new_phrases

        # lack of contras
        if len2 != 0 and len2 < len1:
            new_phrases = random.choices(lists[1], k=len1-len2)
            new_represented_phrase_pairs += new_phrases

    return new_represented_phrase_pairs
        
def handle_phrase_distribution(represented_phrase_pair_counts):

    mapping = {'non-contradiction':0, 'contradiction':1}

    # {word: [[contra_list pairs], [non_contra_list pairs]]}
    word_to_phrase_pairs = {}

    for phrase in represented_phrase_pair_counts:
        p1,p2,count,label = phrase
        phrase_words = p1.split(" ")
        phrase_words += p2.split(" ")
        idx = mapping[label]
        for word in set(phrase_words): # made this a set so that we don't double count

            if word not in word_to_phrase_pairs:
                word_to_phrase_pairs[word] = [[],[]]

            if word not in word_counts:
                word_counts[word]=[0,0]
            
            word_counts[word][idx]+=1
            word_to_phrase_pairs[word][idx].append((p1,p2,-1,label))

    new_represented_phrase_pairs = even_out_phrase_distribution_via_words(word_to_phrase_pairs)

    new_word_to_phrase_pairs={}
    for phrase in new_represented_phrase_pairs:
        p1,p2,_,label = phrase
        phrase_words = p1.split(" ")
        phrase_words += p2.split(" ")
        idx = mapping[label]
        for word in set(phrase_words): # made this a set so that we don't double count

            if word not in new_word_to_phrase_pairs:
                new_word_to_phrase_pairs[word] = [[],[]]

            if word not in word_counts:
                word_counts[word]=[0,0]
            
            word_counts[word][idx]+=1
            new_word_to_phrase_pairs[word][idx].append((p1,p2,-1,label))

    closer_to_even = 0

    for word in new_word_to_phrase_pairs:
        lists = word_to_phrase_pairs[word]
        new_lists = new_word_to_phrase_pairs[word]

        len1,len2 = len(lists[0]),len(lists[1])
        new_len1,new_len2 = len(new_lists[0]),len(new_lists[1])

        old_contra_dist = len2/(len1+len2) 
        new_contra_dist = new_len2/(new_len1+new_len2)
        print(f"{old_contra_dist} | {new_contra_dist}")

        if abs(old_contra_dist-0.5) >= abs(new_contra_dist-0.5):
            closer_to_even+=1

    with open("heuristic_predicted_labels_sampled.tsv","w") as fw:
        fw.write(f"Phrase A\tPhrase B]tPred Label\n")
        for p1,p2,_,label in new_represented_phrase_pairs:
            fw.write(f"{p1}\t{p2}\t{mapping[label]}\n")

    return new_represented_phrase_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', dest='thresh', type=float, required=False, default=0.35, help='The threshold for filtering')
    parser.add_argument('--num_samples', dest='num_samples', type=int, required=False, default=10, help='The number of samples')
    parser.add_argument('--use_mesh', action='store_true')
    args = parser.parse_args()

    base_path = "../pubmed/exact_match_snomed_phrases"
    base_path = "/home/davem/MeSH/ftp.ncbi.nlm.nih.gov/pubmed/exact_match_snomed_phrases"
    phrase_to_frequency_dict, dict_of_mults = get_statistics_about_data(base_path)

    list_of_phrase_pairs, labels, phrase_pairs_to_label = populated_no_num_snomed_phrase_pairs(keep_numbers=True)  
    represented_phrase_pair_counts = get_represented_phrase_pair_counts(list_of_phrase_pairs, labels, phrase_to_frequency_dict)

    dict_of_phrases_to_sentences = create_dataset_of_potentially_contradictory_sentences(base_path, represented_phrase_pair_counts, args.thresh, sample_number=args.num_samples, use_mesh=args.use_mesh)
    
    for a in dict_of_phrases_to_sentences:
        assert len(dict_of_phrases_to_sentences[a]) == phrase_to_frequency_dict[a]