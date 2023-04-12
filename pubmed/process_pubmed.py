import os
import sys
import glob
import multiprocessing as mp
import nltk
import re
import unicodedata
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('words')
from nltk.corpus import words
import random

# There are 3 different versions
def populate_snomed_phrases(version):
    set_of_snomed_phrases = set()
    with open(f"snomed_phrases_v{version}.txt","r") as f:
        phrases = f.readlines()
        for phrase in phrases:
            if phrase.strip() in set_of_snomed_phrases:
                print(phrase)
            set_of_snomed_phrases.add(phrase.strip())
    return set_of_snomed_phrases

def contains_nums(phrase):  
    return re.search('\d', phrase)

def remove_special_chars(set_of_terms):
    re.sub(r'[^\w]', ' ', s)

def extract_unique_words_from_phrases(set_of_snomed_phrases):
    stop_words = set(stopwords.words('english'))
    symbols = {'<','>',':','-'}
    set_of_unique_words = set()
    for phrase in set_of_snomed_phrases:
        word_tokens = phrase.split(" ")
        for word in word_tokens:

            for symbol in symbols:
                word = word.replace(symbol,"")

            try:
                attempt = int(word)
                break
            except:
                pass

            # remove stop words:
            if word not in stop_words:
                set_of_unique_words.add(word)

    return set_of_unique_words


def stemming_query(phrase):
    # tokenized_phrase = nltk.word_tokenize(phrase.lower())
    tokenized_phrase = phrase.lower().split(" ")
    all_fields = "[All Fields]"
    query = ""
    list_of_token_sets = []
    for token in tokenized_phrase:
        porter_stemmer  = PorterStemmer()
        current_set = {token}
        stemmed = porter_stemmer.stem(token)
        for word in words.words():
            # shouldn't be longer than original word
            if stemmed == word[:len(stemmed)] and len(word) <= len(token):
                current_set.add(word)
        list_of_token_sets.append(current_set)
    for i, token_set in enumerate(list_of_token_sets):
        query += "("
        for j, word in enumerate(list(token_set)):
            if j < len(token_set) - 1:
                query += f'"{word}"{all_fields} OR '
            else:
                query += f'"{word}"{all_fields}'


        if i < len(list_of_token_sets) - 1:
            query += ") AND "
        else:
            query += ")"
    return query, list_of_token_sets


def check_if_sentence_contains_stemming_query(sentence, list_of_token_sets):
    # need to check if 1 element from each set is in the sentence
    bool_presence = [False]*len(list_of_token_sets)
    set_of_words_in_sent = set(sentence.split(" "))
    for i, curr_token_set in enumerate(list_of_token_sets):
        if len(set_of_words_in_sent.intersection(curr_token_set)) > 0:
            bool_presence[i] = True
                
    return all(bool_presence)


def does_sentence_contain_phrase_not_in_order(sent, phrase_to_tokenized_phrase_map):
    phrases=[]
    for phrase, word_tokens in phrase_to_tokenized_phrase_map.items():
        all_found = True
        for word in word_tokens:
            # heuristic for whole-word-matching
            if len(word) < 3:
                word = " " + word + " "
            if word not in sent:
                all_found = False
        if all_found:     
            phrases.append(phrase)
    return phrases


"""
Self implementation of sentence tokenization due to issues with sent_tokenize
Long because try to keep the punctuation
"""
def sent_tokenize_manual(abstract):
    # sometimes double spaces are inserted during the scraping process (not my bug)
    abstract = abstract.replace("  "," ")
    split_on_dot = abstract.split(". ")

    for i in range(len(split_on_dot)-1):
        split_on_dot[i]+="."

    split_on_everything = []
    for i,l in enumerate(split_on_dot):
        split_on_q = l.split("?")
        for i in range(len(split_on_q)-1):
            split_on_q[i]+="?"
        split_on_everything+=split_on_q
    
    return split_on_everything


def does_sentence_contain_approximate_phrase(sent, phrase_to_query_map):
    phrases=[]
    for phrase, list_of_token_sets in phrase_to_query_map.items():
        if check_if_sentence_contains_stemming_query(sent, list_of_token_sets):
            phrases.append(phrase)
    
    return phrases


def find_qualifying_sentences_with_phrase(phrase, abstract):
    list_of_matches = []
    abstract = unicodedata.normalize('NFKD', abstract).lower()
    sentences = nltk.sent_tokenize(abstract)
    for sentence in sentences:
        if phrase in sentence:
            list_of_matches.append(sentence)
    return list_of_matches


def does_sentence_contain_exact_phrase(sent, phrase_to_tokenized_phrase_map):
    phrases=[]
    for phrase in phrase_to_tokenized_phrase_map:
        if phrase in sent:
            phrases.append(phrase)

    return phrases


def find_qualifying_sentences_with_word(abstract, set_of_unique_words):
    list_of_matches = []
    abstract = unicodedata.normalize('NFKD', abstract).lower()

    # TODO: need to choose....
    # sentences = nltk.sent_tokenize(abstract)
    sentences = sent_tokenize_manual(abstract)
    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        # word_tokens = sentence.split(" ")
        for word in word_tokens:
            if word in set_of_unique_words:
                list_of_matches.append(sentence)
                break
    return list_of_matches
    

def process_file_for_snomed_phrases(args):
    file_num, set_of_snomed_phrases = args
    perfect_matches = 0
    curr_file_name = "baseline/pubmed22n{:04}.tsv".format(file_num)
    count = 0
    if os.path.isfile(curr_file_name):
        with open(curr_file_name,"r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                count+=1
                line = line[:-1]
                arr = line.split("\t")
                _,title,abstract,labels,pub_types,date,file,mesh_headings,keywords = arr
                # for cardio_snomed_phrase in set_of_snomed_phrases[0]:
                for cardio_snomed_phrase in list(set_of_snomed_phrases)[:10]:
                    list_of_matches = find_qualifying_sentences_with_phrase(cardio_snomed_phrase,abstract)
                    perfect_matches+=len(list_of_matches)
    
    return perfect_matches

"""
SCRAPES ALL OF PUBMED TO FIND WHICH MESH PHRASES ARE PRESENT
"""
def process_for_unique_mesh_phrases(args):
    file_num = args
    print(file_num)
    perfect_matches = 0
    issue_parsing = 0
    curr_file_name = "baseline/pubmed22n{:04}.tsv".format(file_num)
    count = 0
    curr_set_mesh_terms = set()
    if os.path.isfile(curr_file_name):
        with open(curr_file_name,"r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                try:
                    count+=1
                    line = line[:-1]
                    arr = line.split("\t")
                    pmid,title,abstract,labels,pub_types,date,file,mesh_headings,keywords = arr
                    mesh_arr = [x.lower() for x in mesh_headings.split(";")]
                    for x in mesh_arr:
                        curr_set_mesh_terms.add(x)
                except:
                    print("Issue parsing")
                    issue_parsing += 1

    return curr_set_mesh_terms


"""
SCRAPES ALL OF PUBMED TO FIND SENTENCES WHICH CONTAIN INDIVIDUAL WORDS (OF PHRASES)
THIS IS A PREPROCESSING STEP, AFTERWARDS WE GO OVER THESE SENTENCES TO CHECK IF THEY
CONTAIN SNOMED PHRASES
"""
def process_file_for_snomed_unique_words(args):
    file_num, set_of_unique_words = args
    print(file_num)
    perfect_matches = 0
    issue_parsing = 0
    curr_file_name = "baseline/pubmed22n{:04}.tsv".format(file_num)
    unique_words_file_name = "unique_words/{:04}.tsv".format(file_num)
    count = 0
    if os.path.isfile(curr_file_name):
        with open(curr_file_name,"r") as f:
            with open(unique_words_file_name, "w") as fw:
                fw.write(f"sentence_match\tpmid\ttitle\tmesh_headings\tkeywords\n")
                lines = f.readlines()
                for line in lines[1:]:
                    try:
                        count+=1
                        line = line[:-1]
                        arr = line.split("\t")
                        pmid,title,abstract,labels,pub_types,date,file,mesh_headings,keywords = arr
                        # for cardio_snomed_phrase in set_of_snomed_phrases[0]:
                        list_of_matches = find_qualifying_sentences_with_word(abstract, set_of_unique_words)

                        for match in list_of_matches:
                            fw.write(f"{match}\t{pmid}\t{title}\t{mesh_headings}\t{keywords}\n")
                        perfect_matches+=len(list_of_matches)
                    except:
                        print("Issue parsing")
                        issue_parsing += 1

    return (perfect_matches, issue_parsing)


def process_transition_files_for_exact_snomed_phrases(args):
    file_num, phrase_to_its_pairs, pairs_of_phrases_to_label_dict,phrase_to_tokenized_phrase_map,exact = args
    print(file_num)
    perfect_matches = 0
    transition_file_name = "transitions/{:04}.tsv".format(file_num)
    # snomed_phrases_file_name = "transitions_non_exact_matches/{:04}.tsv".format(file_num)
    snomed_phrases_file_name = "transitions_non_exact_matches_all_snomed_under_12/{:04}.tsv".format(file_num)
    count = 0
    if os.path.isfile(transition_file_name):
        with open(transition_file_name,"r") as f:
            with open(snomed_phrases_file_name, "w") as fw:
                fw.write(f"sentence1\tsentence2\ttransition_word\tpmid\tmesh_headings\tlist_of_present_phrase_pairs\n")
                lines = f.readlines()
                for line in lines[1:]:
                    # try:
                        curr_phrase_dictionary = {} # of the format {phrase1: [phrase2_list], ...}
                        curr_pairs_of_phrases = []
                        combo_count = 0
                        found_a_phrase_1 = False
                        found_a_phrase_2 = False
                        sentence1,sentence2,transition_word,pmid,mesh_headings,keywords = line.split("\t")
                        sentence1 = sentence1.replace("  "," ") 
                        sentence2 = sentence2.replace("  "," ") 
                        count+=1

                        # find if sentence contains any of the phrases
                        if exact:
                            phrases_1 = does_sentence_contain_exact_phrase(sentence1, phrase_to_tokenized_phrase_map)
                        else:
                            phrases_1 = does_sentence_contain_phrase_not_in_order(sentence1, phrase_to_tokenized_phrase_map)

                        if len(phrases_1) > 0:
                            found_a_phrase_1 = True
                            potential_phrase_pairs = set()
                            for phrase in phrases_1: # iterate through the phrases which were found in the text
                                assert phrase not in curr_phrase_dictionary
                                its_pairs = phrase_to_its_pairs[phrase]
                                its_phrase_to_tokenized_phrase_map = {}
                                for p in its_pairs:
                                    its_phrase_to_tokenized_phrase_map[p] = phrase_to_tokenized_phrase_map[p]
                                potential_phrase_pairs.update(its_pairs)

                                if len(potential_phrase_pairs) > 0:
                                    if exact:
                                        phrases_2 = does_sentence_contain_exact_phrase(sentence2, its_phrase_to_tokenized_phrase_map)
                                    else:
                                        phrases_2 = does_sentence_contain_phrase_not_in_order(sentence2, its_phrase_to_tokenized_phrase_map)

                                    if len(phrases_2) > 0: # means that we have a match
                                        for p2 in phrases_2:
                                            # assert p2 in sentence2
                                            print(p2,phrase)
                                            assert p2 in phrase_to_its_pairs[phrase]
                                            assert phrase in phrase_to_its_pairs[p2]
                                            if phrase not in curr_phrase_dictionary:
                                                curr_phrase_dictionary[phrase] = [p2]
                                                combo_count+=1
                                            else:
                                                curr_phrase_dictionary[phrase].append(p2)
                                                combo_count+=1

                                            snomed_heuristic_label = pairs_of_phrases_to_label_dict[(phrase,p2)]
                                            curr_pairs_of_phrases.append((phrase,p2,snomed_heuristic_label))

                                        found_a_phrase_2 = True


                            # if len(potential_phrase_pairs) > 0:
                            #     phrases_2 = does_sentence_contain_exact_phrase(sentence2, potential_phrase_pairs)
                            #     if len(phrases_2) > 0: # means that we have a match
                            #         print("**************")
                            #         print(sentence1)
                            #         print(phrases_1)
                            #         print()
                            #         print(sentence2)
                            #         print(phrases_2)
                            #         print()
                            #         for p1 in phrases_1:
                            #             assert p1 in sentence1
                            #             for p2 in phrases_2:
                            #                 # assert p1!=p2
                            #                 assert p2 in sentence2
                            #                 assert p2 in phrase_to_its_pairs[p1]
                            #                 assert p1 in phrase_to_its_pairs[p2]
                                            
                            #         found_a_phrase_2 = True

                        if found_a_phrase_1 and found_a_phrase_2:
                            perfect_matches+=1
                            assert combo_count == len(curr_pairs_of_phrases)
                            print(combo_count)
                            fw.write(f"{sentence1}\t{sentence2}\t{transition_word}\t{pmid}\t{mesh_headings}\t{curr_pairs_of_phrases}\n")
                    # except:
                    #     print("Issue parsing")
                    
    return (perfect_matches,0)


"""
Helper function to normalize and clean input/output
"""
def clean_phrase(phrase):
    phrase = phrase.replace(":","")
    phrase = phrase.replace("  "," ")
    phrase = phrase.lower()
    phrase = phrase.strip()
    return phrase


"""
Only keep matches which are perfectly contain the snomed phrase.
This is in line with the main paper.
"""
def process_unique_words_files_for_exact_snomed_phrases(args):
    file_num, phrase_to_tokenized_phrase_map = args
    print(file_num)
    perfect_matches = 0
    unique_words_file_name = "unique_words/{:04}.tsv".format(file_num)
    snomed_phrases_file_name = "exact_match_snomed_phrases/{:04}.tsv".format(file_num)
    count = 0
    if os.path.isfile(unique_words_file_name):
        with open(unique_words_file_name,"r") as f:
            with open(snomed_phrases_file_name, "w") as fw:
                lines = f.readlines()
                for sent in lines[1:]:
                    try:
                        count+=1
                        sent = sent[:-1]

                        # TODO: sanity check if this is correct
                        match,pmid,title,mesh_headings,keywords = sent.split("\t")
                        # find if sentence contains any of the phrases
                        phrases = does_sentence_contain_exact_phrase(match, phrase_to_tokenized_phrase_map)
                        if len(phrases) > 0:
                            phrase_string = ",".join(phrases)
                            fw.write(f"{sent}\t{phrase_string}\n")
                            perfect_matches+=1
                    except:
                        print("Issue parsing")
                    
    return (perfect_matches,0)


"""
Look through already filtered sentences for snomed phrases.
A sentence is a 'match' if it contains a stemming-variation of every word in the phrase
For example, for the word "decreasing", the stemming variations may be:
    - "decreas"
    - "decr"
"""
def process_unique_words_files_for_approximate_snomed_phrases(args):
    file_num, phrase_to_query_map = args
    print(file_num)
    perfect_matches = 0
    unique_words_file_name = "unique_words/{:04}.tsv".format(file_num)
    snomed_phrases_file_name = "approximate_match_snomed_phrases/{:04}.tsv".format(file_num)
    count = 0
    if os.path.isfile(unique_words_file_name):
        with open(unique_words_file_name,"r") as f:
            with open(snomed_phrases_file_name, "w") as fw:
                lines = f.readlines()
                for sent in lines[1:]:
                    try:
                        count+=1
                        sent = sent[:-1]

                        # find if sentence contains any of the phrases
                        phrases = does_sentence_contain_approximate_phrase(sent, phrase_to_query_map)
                        # phrases = does_sentence_contain_exact_phrase(sent, phrase_to_query_map)

                        if len(phrases) > 0:
                            phrase_string = ",".join(phrases)
                            fw.write(f"{sent}\t{phrase_string}\n")
                            perfect_matches+=1

                    except:
                        print("Issue parsing")

    return perfect_matches


"""
Look through already filtered sentences for snomed phrases.
A sentence is a 'match' if it contains every word in the phrase, NOT necessarily
in order and sequentially...
Some tricks include
    - if word is of length < 3, then it must match the 'whole' word 
"""
def process_unique_words_files_for_snomed_phrases_without_order(args):
    file_num, phrase_to_tokenized_phrase_map = args
    print(file_num)
    perfect_matches = 0
    unique_words_file_name = "unique_words/{:04}.tsv".format(file_num)
    snomed_phrases_file_name = "out_of_order_match_cleaned_snomed_phrases/{:04}.tsv".format(file_num)
    count = 0
    if os.path.isfile(unique_words_file_name):
        with open(unique_words_file_name,"r") as f:
            with open(snomed_phrases_file_name, "w") as fw:
                lines = f.readlines()
                for sent in lines[1:]:
                    try:
                        count+=1
                        sent = sent[:-1]

                        # find if sentence contains any of the phrases
                        phrases = does_sentence_contain_phrase_not_in_order(sent, phrase_to_tokenized_phrase_map)

                        if len(phrases) > 0:
                            phrase_string = ",".join(phrases)
                            fw.write(f"{sent}\t{phrase_string}\n")
                            perfect_matches+=1

                    except:
                        print("Issue parsing")

    return perfect_matches


'''
Counts the number of abstracts across all downloaded pubmed files
'''
def run_count_number_of_abstracts():
    POOLSIZE = 80
    pool = mp.Pool(POOLSIZE)
    fnames = list(range(1,1115))
    total_perfect_matches=0
    args = fnames
    for x in pool.imap_unordered(count_number_of_abstracts, args, 1):
        total_perfect_matches+=x

    return total_perfect_matches

        
'''
Counts the number of abstracts for the given file num suffix
'''
def count_number_of_abstracts(file_num):  
    print(file_num)  
    curr_file_name = "baseline/pubmed22n{:04}.tsv".format(file_num)
    set_of_pmids = set()
    if os.path.isfile(curr_file_name):
        with open(curr_file_name,"r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line[:-1]
                arr = line.split("\t")
                try:
                    pmid,title,abstract,labels,pub_types,date,file,mesh_headings,keywords = arr
                    set_of_pmids.add(pmid)

                except:
                    pass

            
    return len(set_of_pmids)

    
if __name__ == "__main__":

    # run_transition_dataset_creation()
    run_count_number_of_abstracts()
    # run_to_find_transition_files_with_snomed_phrases("/home/davem/MeSH/ftp.ncbi.nlm.nih.gov/pubmed/generated_contradicting_pairs_shuffled_all_under_12_scratch.csv",exact=False)
    # run_to_find_transition_files_with_exact_snomed_phrases("/home/davem/MeSH/ftp.ncbi.nlm.nih.gov/pubmed/generated_contradicting_pairs_shuffled_scratch.csv")
    assert False
    POOLSIZE = 80
    pool = mp.Pool(POOLSIZE)
    fnames = list(range(1,1115))

    arguments = fnames
    total_perfect_matches=0
    set_of_mesh_terms = set()
    total_sentences=0
    phrases = set()
    with open("/home/davem/MeSH/ftp.ncbi.nlm.nih.gov/pubmed/generated_contradicting_pairs_shuffled_all_cardio_scratch.csv") as f:
        lines = f.readlines()[1:]
        for line in lines:
            _,p1,_,p2,label = line.split(",")
            phrases.add(p1.lower())
            phrases.add(p2.lower())

    phrase_to_tokenized_phrase_map = {}
    for phrase in phrases:
        phrase_words = phrase.split(" ")
        phrase_to_tokenized_phrase_map[phrase]=phrase_words

    phrase_to_its_pairs = {} # {phrase1: [phrase2_list]}
    pairs_of_phrases_to_label_dict = {} # {(phrase1, phrase2): label}
    with open("/home/davem/MeSH/ftp.ncbi.nlm.nih.gov/pubmed/generated_contradicting_pairs_shuffled_all_cardio_scratch.csv") as f:
        lines = f.readlines()[1:]
        for line in lines:
            _,p1,_,p2,label = line.split(",")
            p1 = p1.lower()
            p2 = p2.lower()

            if (p1,p2) in pairs_of_phrases_to_label_dict or (p2,p1) in pairs_of_phrases_to_label_dict:
                # due to EKG cleaning, this can happen...
                assert label.strip() == pairs_of_phrases_to_label_dict[p1,p2]

            pairs_of_phrases_to_label_dict[(p1,p2)] = label.strip()
            pairs_of_phrases_to_label_dict[(p2,p1)] = label.strip()

            if p1 not in phrase_to_its_pairs:
                phrase_to_its_pairs[p1] = [p2]
            else:
                phrase_to_its_pairs[p1].append(p2)

            if p2 not in phrase_to_its_pairs:
                phrase_to_its_pairs[p2] = [p1]
            else:
                phrase_to_its_pairs[p2].append(p1)

    print(len(phrase_to_its_pairs))
    print(sum([len(y) for x,y in phrase_to_its_pairs.items()])/2)

    args = zip(fnames,[phrase_to_its_pairs]*len(fnames),[pairs_of_phrases_to_label_dict]*len(fnames),[phrase_to_tokenized_phrase_map]*len(fnames))
    total_perfect_matches = 0
    for x, y in pool.imap_unordered(process_transition_files_for_exact_snomed_phrases, args, 1):
        total_perfect_matches+=x
        
    print(total_perfect_matches)           
    assert False

    # Note that the filenames continue in numbering from one directory
    # to the other (but do not overlap)
    file_num_to_positive_pairs = {}
    file_num_to_negative_pairs = {}
    total_sentences = 0
    total_perfect_matches = 0
    total_issues_parsing = 0
    VERSION = 5 # TODO: THIS MUST BE CHANGED DEPENDING ON THE VERSION

#     for x, y, file_num, total in pool.imap_unordered(process_file_for_contrastive, fnames, 1):

#         file_num_to_positive_pairs[file_num]=x
#         file_num_to_negative_pairs[file_num]=y
#         total_sentences+=total
    # for x, y, file_num, total in pool.imap_unordered(process_file_for_contrastive_clinical, fnames, 1):

    #     file_num_to_positive_pairs[file_num]=x
    #     file_num_to_negative_pairs[file_num]=y
    #     total_sentences+=total
    set_of_snomed_phrases = populate_snomed_phrases(VERSION)
    set_of_unique_words = extract_unique_words_from_phrases(set_of_snomed_phrases)

    list_of_set_of_snomed_phrases = [set_of_snomed_phrases]*len(fnames)
    list_of_set_of_unique_words = [set_of_unique_words]*len(fnames)
    phrase_to_query_map = {}
    phrase_to_tokenized_phrase_map = {}

    # with open("phrase_to_tokenized_phrase_v3.tsv", "r") as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         phrase, tokenized_phrase = [x.strip() for x in line.split("\t")]

    #         # add spaces before and after
    #         phrase_to_tokenized_phrase_map[phrase] = eval(tokenized_phrase)

    phrase_to_tokenized_phrase_map = {}
    for phrase in set_of_snomed_phrases:
        phrase_words = phrase.split(" ")
        phrase_to_tokenized_phrase_map[phrase]=phrase_words

    with open("phrase_to_stemming_query_map_v3.tsv", "r") as f:
        lines = f.readlines()
        for line in lines:
            phrase, list_of_token_sets = [x.strip() for x in line.split("\t")]
            phrase_to_query_map[phrase] = eval(list_of_token_sets)

    list_of_phrase_to_tokenized_phrase_maps_v1 = [phrase_to_tokenized_phrase_map]*len(fnames)
    # list_of_phrase_to_query_maps = [phrase_to_query_map]*len(fnames)
    # list_of_phrase_to_tokenized_phrase_maps = [phrase_to_tokenized_phrase_map]*len(fnames)

    print(len(set_of_unique_words))
    print(len(phrase_to_tokenized_phrase_map))

    # arguments = [(x,y) for (x,y) in zip(fnames, list_of_set_of_snomed_phrases)]
    # arguments = [(x,y) for (x,y) in zip(fnames, list_of_set_of_unique_words)]
    arguments = [(x,y) for (x,y) in zip(fnames, list_of_phrase_to_tokenized_phrase_maps_v1)]
    # arguments = [(x,y) for (x,y) in zip(fnames, list_of_phrase_to_query_maps)]
    # arguments = [(x,y) for (x,y) in zip(fnames, list_of_phrase_to_tokenized_phrase_maps)]

    # for x in pool.imap_unordered(process_file_for_snomed_phrases, arguments, 1):
    # for x in pool.imap_unordered(process_file_for_snomed_unique_words, arguments,1):
    #     total_perfect_matches+=x[0]
    #     total_issues_parsing+=x[1]
    for x in pool.imap_unordered(process_unique_words_files_for_exact_snomed_phrases, arguments,1):
        total_perfect_matches+=x[0]

    # for x in pool.imap_unordered(process_unique_words_files_for_approximate_snomed_phrases, arguments,1):
    # for x in pool.imap_unordered(process_unique_words_files_for_snomed_phrases_without_order, arguments,1):
    #     total_perfect_matches+=x
        
    print(f"Total perfect matches: {total_perfect_matches}")
    # print(f"Total issues parsing: {total_issues_parsing}")
    total_sentences_2 = 0
    total_positive = 0
    total_negative = 0
    for k,v in file_num_to_positive_pairs.items():
        total_sentences_2+=len(v)
        total_positive+=len(v)
    for k,v in file_num_to_negative_pairs.items():
        total_sentences_2+=len(v)
        total_negative+=len(v)
        
    print(total_sentences)
    print(total_sentences_2)
    print(total_positive)
    print(total_negative)
    assert total_sentences_2 == total_sentences