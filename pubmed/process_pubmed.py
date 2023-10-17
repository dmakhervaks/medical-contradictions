import argparse
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

def populate_snomed_phrases():
    set_of_snomed_phrases = set()
    with open(f"../snomed/snomed_phrases.txt","r") as f:
        phrases = f.readlines()
        for phrase in phrases:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter_by_unique_words', action='store_true')
    parser.add_argument('--filter_by_exact_phrases', action='store_true')
    args = parser.parse_args()

    POOLSIZE = 80
    pool = mp.Pool(POOLSIZE)
    fnames = list(range(1,1115))

    total_perfect_matches = 0
    total_issues_parsing = 0

    set_of_snomed_phrases = populate_snomed_phrases()

    phrase_to_tokenized_phrase_map = {}
    for phrase in set_of_snomed_phrases:
        phrase_words = [x.lower() for x in phrase.split(" ")]
        phrase_to_tokenized_phrase_map[phrase.lower()]=phrase_words


    # NOTE: processes all of the baseline pubmed files locally by filtering first the sentences containing unique words
    if args.filter_by_unique_words:
        set_of_unique_words = extract_unique_words_from_phrases(set_of_snomed_phrases)
        list_of_set_of_unique_words = [set_of_unique_words]*len(fnames)
        arguments = [(x,y) for (x,y) in zip(fnames, list_of_set_of_unique_words)]
        for x in pool.imap_unordered(process_file_for_snomed_unique_words, arguments,1):
            total_perfect_matches+=x[0]

    # NOTE: takes the unique word files and finds exact matches
    if args.filter_by_exact_phrases:
        list_of_phrase_to_tokenized_phrase = [phrase_to_tokenized_phrase_map]*len(fnames)
        arguments = [(x,y) for (x,y) in zip(fnames, list_of_phrase_to_tokenized_phrase)]
        for x in pool.imap_unordered(process_unique_words_files_for_exact_snomed_phrases, arguments,1):
            total_perfect_matches+=x[0]

    print(f"Total perfect matches: {total_perfect_matches}")