from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from wordhoard import Synonyms
from wordhoard import Antonyms
import random
import json
import os


RERUN_INIT=False

default_world.set_backend(filename = "pym.sqlite3")
# import_umls("umls-2019AA-metathesaurus.zip", terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])
if RERUN_INIT:  
    import_umls("2022AA.zip", terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])
PYM = get_ontology("http://PYM/").load()
SNOMEDCT_US = PYM["SNOMEDCT_US"]
GROUP_SIZE= 250
one = PYM["MESH"]
ICD10 = PYM["ICD10"]
CUI = PYM["CUI"]

file = "all_list_of_dicts.tsv"
list_of_dicts = []
lines = []
with open(file) as f:
    lines = f.readlines()
    
count = 0
for d in lines:
    count+=1
    curr_dict = eval(d.strip())
    list_of_dicts.append(curr_dict)


cui_to_snomed_id = {} # one to many mapping
phrase_to_cui = {} # one to many mapping
with open("snomed_all_phrases_cui.tsv") as f:
    lines = f.readlines()
    for line in lines[1:]:
        phrase,cui,synonyms,snomed_id = [x.strip() for x in line.split("\t")]
        cui = eval(cui)
        for c in cui:  
            if c in cui_to_snomed_id:
                cui_to_snomed_id[c].add(snomed_id)
            else:
                cui_to_snomed_id[c] = {snomed_id}
            if phrase.lower() in phrase_to_cui:
                phrase_to_cui[phrase.lower()].add(c)
            else:
                phrase_to_cui[phrase.lower()] = {c}

filtered_list_of_dicts = []
key_count = {'prefLabel':0, 'synonym':0, 'cui':0, 'semanticType':0, 'children':0, 'parents':0, 
         'descendants':0, 'ancestors':0}
for d in list_of_dicts:
    num_keys = 0
    for k in d.keys():
        if k in key_count:
            num_keys+=1
    if num_keys == len(key_count):
        at_least_one = False
        for c in d['cui']:
            if c in cui_to_snomed_id:
                at_least_one=True
        if at_least_one:
            filtered_list_of_dicts.append(d)

# write filtered_list_of_dict
with open("filtered_list_of_dicts.tsv","w") as fw:
    for d in filtered_list_of_dicts:
        fw.write(f"{d}\n")

lengths = 0
lengths2 = 0
for i, d in enumerate(filtered_list_of_dicts):
    all_sids = []
    all_inters = []
    all_interprets = []
    all_mapped = []
    for c in d['cui']:
        sids = cui_to_snomed_id[c]
        # all_sids.append(sid)
        all_sids.extend(sids)
        
        # there may be multiple sids - i.e. Edentulous and Absence of teeth 
        # map to same cui 
        inter = []
        interprets = []
        mapped_to = []
        
        for sid in sids:
            inter.extend(SNOMEDCT_US[int(sid)].has_interpretation)
            interprets.extend(SNOMEDCT_US[int(sid)].interprets)
            mapped_to.extend(SNOMEDCT_US[int(sid)].mapped_to)


        for curr_inter in inter: all_inters.append(curr_inter.label[0])
        for curr_interprets in interprets: all_interprets.append(curr_interprets.label[0])
        for curr_mapped in mapped_to: all_mapped.append(curr_mapped.label[0])


    filtered_list_of_dicts[i]['interpretations'] = all_inters
    filtered_list_of_dicts[i]['interprets'] = all_interprets
    filtered_list_of_dicts[i]['mappedTo'] = all_mapped
    filtered_list_of_dicts[i]['sids'] = all_sids

    set_of_children = set()
    snomed_roots = {123037004,404684003,308916002,272379006,363787002,410607006,373873005,78621006,260787004,71388002,362981000,419891008,243796009,900000000000441003,48176007,370115009,123038009,254291000,105590001 }

    def traverse_tree(root,set_of_interest):
        if root is None:
            return
        set_of_interest.add(root.name)
        children = root.children
        for child in root.children:
            traverse_tree(child,set_of_interest)
        return 
    
    # traverse over the entire tree to get all nodes
all_nodes = set()
for curr_root in snomed_roots:
    curr_root_concept = SNOMEDCT_US[curr_root]

cardio_words = set()
with open("cardiology_words.csv") as f:
    lines = f.readlines()
    for line in lines:
        word = line.strip()
        if len(word)>1:
            cardio_words.add(word)

def get_words_and_acronyms(field):
    words = set()
    acronyms = set()
    with open(f"field_words/{field}_words.tsv") as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip().lower()
            if len(word)>1:
                words.add(word)

    if os.path.exists(f"field_acronyms/{field}_acronyms.tsv"):
        with open(f"field_acronyms/{field}_acronyms.tsv") as f:
            lines = f.readlines()
            for line in lines:
                word = line.strip().lower()
                if len(word)>1:
                    acronyms.add(word)
                
    return words, acronyms

def get_field_terms(words, acronyms):
    # only care about nodes relating to cardio
    term_count = 0
    terms = set()
    for node in all_nodes:
        assert len(SNOMEDCT_US[node].label) == 1
        curr_label = SNOMEDCT_US[node].label[0]
        curr_label_words = curr_label.split(" ")
        for w in words:
            for c in curr_label_words:
                if w.lower() in c.lower():
                    terms.add(node)
                    
        for w in acronyms:
            for c in curr_label_words:
                if w == c:
                    terms.add(node)
                    
    return terms

# only care about nodes relating to cardio
cardio_term_count = 0
cardio_terms = set()
for node in all_nodes:
    assert len(SNOMEDCT_US[node].label) == 1
    curr_label = SNOMEDCT_US[node].label[0]
    for w in cardio_words:
        # if w in curr_label.lower():
        #     cardio_terms.add(node)
        if 'heart' in curr_label.lower() or 'cardia' in curr_label.lower() or 'cardio' in curr_label.lower() or 'aerobic' in curr_label.lower() or 'cardiac' in curr_label.lower():
            cardio_terms.add(node)


# only care about nodes relating to cardio
cardio_term_count = 0
cardio_terms_2 = set()
for node in all_nodes:
    assert len(SNOMEDCT_US[node].label) == 1
    curr_label = SNOMEDCT_US[node].label[0]
    for w in cardio_words:
        if w in curr_label.lower():
            cardio_terms_2.add(node)
        # if 'heart' in curr_label.lower() or 'cardia' in curr_label.lower() or 'cardio' in curr_label.lower() or 'aerobic' in curr_label.lower() or 'cardiac' in curr_label.lower():
        #     cardio_terms.add(node)


# but we don't want just them, we also want all cardio node children
all_cardio_set_of_children = set()
for cardio_term in cardio_terms_2:
    curr_root_concept = SNOMEDCT_US[cardio_term]
    traverse_tree(curr_root_concept,all_cardio_set_of_children)

with open('synonym_dict.json', 'r') as file:
    synonym_dict = json.load(file)
    
with open('antonym_dict.json', 'r') as file:
    antonym_dict = json.load(file)

useless_set = {'\x1b', '[', '3', '8', ';', '2', ';', '2', '5', '5', ';', '0', ';', '2', '5', '5', 'm', 'N', 'o', ' ', 's', 'y', 'n', 'o', 'n', 'y', 'm', 's', ' ', 'w', 'e', 'r', 'e', ' ', 'f', 'o', 'u', 'n', 'd', ' ', 'f', 'o', 'r', ' ', 't', 'h', 'e', ' ', 'w', 'o', 'r', 'd', ':', ' ', 'h', 'y', 'p', 'o', 'k', 'i', 'n', 'e', 't', 'i', 'c', ' ', '\n', 'P', 'l', 'e', 'a', 's', 'e', ' ', 'v', 'e', 'r', 'i', 'f', 'y', ' ', 't', 'h', 'a', 't', ' ', 't', 'h', 'e', ' ', 'w', 'o', 'r', 'd', ' ', 'i', 's', ' ', 's', 'p', 'e', 'l', 'l', 'e', 'd', ' ', 'c', 'o', 'r', 'r', 'e', 'c', 't', 'l', 'y', '.', ' ', '\x1b', '[', '3', '8', ';', '2', ';', '2', '5', '5', ';', '2', '5', '5', ';', '2', '5', '5', 'm'}
synonym_dict_updated = {}
antonym_dict_updated = {}
for word,values in synonym_dict.items():
    new_values = set()
    for val in values:
        if val not in useless_set and len(val) > 1:
            new_values.add(val.lower())
    assert word not in synonym_dict_updated
    synonym_dict_updated[word] = list(new_values)
    
for word,values in antonym_dict.items():
    new_values = set()
    for val in values:
        if val not in useless_set and len(val) > 1:
            new_values.add(val.lower())
    assert word not in antonym_dict_updated
    antonym_dict_updated[word] = list(new_values)


import nltk
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_common_words(phrase_1_words, phrase_2_words):
    intersection = set(phrase_1_words).intersection(set(phrase_2_words))
    
    uniq_phrase_1_words = []
    uniq_phrase_2_words = []
    
    for word in phrase_1_words:
        if word not in intersection:
            uniq_phrase_1_words.append(word)
            
    for word in phrase_2_words:
        if word not in intersection:
            uniq_phrase_2_words.append(word) 
            
    return uniq_phrase_1_words, uniq_phrase_2_words

def find_intersection(phrase_1, phrase_2):
    phrase_1, phrase_2 = phrase_1.lower(), phrase_2.lower()
    phrase_1_words = phrase_1.split(" ")
    phrase_2_words = phrase_2.split(" ")
    
    uniq_phrase_1_words, uniq_phrase_2_words = remove_common_words(phrase_1_words,phrase_2_words)
    
    return uniq_phrase_1_words, uniq_phrase_2_words

def remove_stop_words(phrase):  
    word_tokens = word_tokenize(phrase)

    filtered_phrase = [w for w in word_tokens if not w.lower() in stop_words]

    filtered_phrase = []
    filtered_string = phrase 
    for i, w in enumerate(word_tokens):
        if w not in stop_words:
            filtered_phrase.append(w)
        else:
            filtered_string = filtered_string.replace(w+" ","")
            filtered_string = filtered_string.replace(w,"")

    return filtered_string


def remove_stop_words_v2(phrase):  
    word_tokens = phrase.split(" ")
    
    stop_words_v2 = {"in", "of", "or", "due","to", "on","at"}

    filtered_phrase = [w for w in word_tokens if not w.lower() in stop_words_v2]
    filtered_phrase = []
    filtered_string = phrase.lower() 
    for i, w in enumerate(word_tokens):
        w = w.lower()
        if w not in stop_words_v2:
            filtered_phrase.append(w)
        else:
            filtered_string = filtered_string.replace(w+" ","")
            filtered_string = filtered_string.replace(w,"")
            
    # remove ekg/ecg stuff
    filtered_string=filtered_string.replace("ekg: ","")
    filtered_string=filtered_string.replace("ecg: ","")
    filtered_string=filtered_string.replace("ekg:","")
    filtered_string=filtered_string.replace("ecg:","")
    return filtered_string


def take_care_of_special_cases(uniq_phrase_1_words, uniq_phrase_2_words, word1_l, word2_l, auto_label):
    if "|" in word1_l or "|" in word2_l:
        word1_l_arr = [x.strip() for x in word1_l.split("|")]
        word2_l_arr = [x.strip() for x in word2_l.split("|")]
        word1_l_set = set(word1_l_arr)
        word2_l_set = set(word2_l_arr)

        intersection = word1_l_set.intersection(word2_l_set)

        # at least one of them completely matches.. not a contradiction by our heuristic
        if len(intersection) == len(word1_l_set) or len(intersection) == len(word2_l_set):
            auto_label = "0"
        # both of them have extra information... most likely contradiction
        else:
            auto_label = "1"
            
    if ("right" in uniq_phrase_1_words and "left" in uniq_phrase_2_words) or ("right" in uniq_phrase_2_words and "left" in uniq_phrase_1_words):
        auto_label="0"
    
    return auto_label


def use_synonym_heuristic(uniq_phrase_1_words,uniq_phrase_2_words,auto_label,phrase1,phrase2):
    if len(uniq_phrase_1_words) == 1 and len(uniq_phrase_2_words) == 1:
        word1 = uniq_phrase_1_words[0]
        word2 = uniq_phrase_2_words[0]
        is_synonym = False
        is_antonym = False

        # assert word1 in synonym_dict
        if word1 not in synonym_dict:
            synonym_dict[word1] = []
            for syn in wn.synsets(word1):
                for lemma in syn.lemma_names():
                    synonym_dict[word1].append(lemma)
                    if lemma == word2 and lemma != word1:
                        is_synonym = True
                        auto_label = "0"
            for syn in Synonyms(word1).find_synonyms():
                synonym_dict[word1].append(syn)
                if syn == word2:
                    is_synonym = True
                    auto_label = "0"
        else: # already have populated synonyms
            for syn in synonym_dict[word1]:
                if syn == word2:
                    is_synonym = True
                    auto_label = "0"

        # assert word1 in antonym_dict
        if word1 not in antonym_dict:
            antonym_dict[word1] = []
            for syn in wn.synsets(word1):
                for i in syn.lemmas():
                    if i.antonyms():
                        ant = i.antonyms()[0].name()
                        antonym_dict[word1].append(ant)
                        if ant == word2:
                            is_antonym = True
            for ant in Antonyms(word1).find_antonyms():
                antonym_dict[word1].append(ant)
                if ant == word2:
                    is_antonym = True
                    auto_label = "1"
        else: # already have populated antonyms
            for ant in antonym_dict[word1]:
                if ant == word2:
                    is_antonym = True
                    auto_label = "1"
        
    return auto_label


def determine_label(phrases, interpretations, use_synonyms=True, rid_of_stop_words=True, special_cases=True):
    phrase1,phrase2 = phrases
    interp1,interp2 = interpretations
    
    naive_intersection = interp1.intersection(interp2)
    naive_label = "1" if len(naive_intersection) == 0 else "0"
    
    reconciled_label = naive_label
    
    if rid_of_stop_words:
        phrase1 = remove_stop_words_v2(phrase1)
        phrase2 = remove_stop_words_v2(phrase2)

    uniq_phrase_1_words, uniq_phrase_2_words = find_intersection(phrase1,phrase2)

    if len(uniq_phrase_1_words) == 1 and len(uniq_phrase_2_words) == 1:
        set_of_unique_words.add(uniq_phrase_1_words[0])
        set_of_unique_words.add(uniq_phrase_2_words[0])
    
    if use_synonyms:
        reconciled_label = use_synonym_heuristic(uniq_phrase_1_words,uniq_phrase_2_words,reconciled_label,phrase1,phrase2)
        
    if special_cases:
        reconciled_label = take_care_of_special_cases(uniq_phrase_1_words, uniq_phrase_2_words, interp1, interp2, reconciled_label)
                 
            
    # IMPORTANT!: when reconciled claims its a contradiction, but naive claims it isnt... most of the time
    # reconciled is correct (i.e. lots of types 'Abnormal')
    # However... when reconciled claims non-contradiction, but naive_label claims contradiction... naive is
    # correct most of the time.. this usually happens because of 'special case'.. maybe should just get rid of it
    
    if str(reconciled_label)=='0' and str(naive_label) =='1':
        final_label = naive_label
    elif str(reconciled_label)=='1' and str(naive_label) =='0':
        final_label = reconciled_label
    else: # they are the same
        final_label = naive_label
        
    return final_label
    
set_of_unique_words = set()

# find the tree ancestry pattern of the given i_by's
import math

# only cardio focused...
def clean_phrases_of_colon_prefixes(i_by):
    phrase = i_by.label[0].lower()
    phrase = phrase.replace("ekg: ","")
    phrase = phrase.replace("ecg: ","")
    i_by.label[0]=phrase
    return i_by

def clean_phrases_of_colon_prefixes_phrase(phrase):
    phrase = phrase.lower()
    phrase = phrase.replace("ekg: ","")
    phrase = phrase.replace("ecg: ","")
    return phrase
    
def determine_interpretation_relationship(interpreted_by, single_word_dict, phrase_dict, sample=True):
    # {i_by: self and parents}
    dict_interpreted_by = {}
    count = 0
    # i_by is a phrase which describes a certain observed outcome.. i.e. 
    # Increased vascular resistance
    for i_by in interpreted_by:
        # for the i_by example above.. i.e. increase (usually single word, but sometimes 2+)
        has_interpretations = [x.label[0] for x in i_by.has_interpretation]
        
        # if any of the parents are the same between the i_by, then they are same category (of contra spectrum)
        assert i_by not in dict_interpreted_by
        modified_parents = []
        
        modified_parents = i_by.parents
                
        parents_and_self = modified_parents + [i_by] 
        dict_interpreted_by[i_by] = parents_and_self
        
    # TODO: HERE we can sample
    keys = list(dict_interpreted_by.keys())
    pairs_of_keys = set()
    pairs_of_contra_keys = set()
    pairs_of_non_contra_keys = set()
    without_interpretations = 0
    without_interpretations_list = []
    for key_i in keys:
        for key_j in keys:
            if key_i != key_j:
                parents_i = set(dict_interpreted_by[key_i])
                parents_j = set(dict_interpreted_by[key_j])
                descendents_i = key_i.descendant_concepts()
                descendents_j = key_j.descendant_concepts()
                
                parent_intersection = parents_i.intersection(parents_j)
                
                i_interpretation = set([x.label[0] for x in key_i.has_interpretation])
                j_interpretation = set([x.label[0] for x in key_j.has_interpretation])
                i_j_interpretation_intersection = i_interpretation.intersection(j_interpretation)
                
                # not all of them will have an 'interpretation' word
                if len(i_interpretation) > 0 and len(j_interpretation) > 0:
                    if (key_i,key_j) not in pairs_of_keys and (key_j,key_i) not in pairs_of_keys:
                        pairs_of_keys.add((key_i,key_j))
                        
                        assert len(key_i.label) == 1
                        assert len(key_j.label) == 1
                        
                        predicted_label = determine_label((key_i.label[0],key_j.label[0]),(i_interpretation,j_interpretation))

                        if predicted_label=="1":
                                pairs_of_contra_keys.add((key_i,key_j))
                        elif predicted_label=="0":
                                pairs_of_non_contra_keys.add((key_i,key_j))
                        else:
                            assert False
                else:
                    without_interpretations+=1
                    without_interpretations_list.append((key_i.name,key_j.name))
    
    assert pairs_of_keys == pairs_of_contra_keys.union(pairs_of_non_contra_keys)

    if sample:
        # MAYBE should do random.choice instead if the number is not 0 (to make even number of samples)
        if len(pairs_of_keys) < 5:
            sampled_keys = pairs_of_keys
        elif min(len(pairs_of_contra_keys),len(pairs_of_non_contra_keys)) == 0: 
            contra_sample = random.sample(pairs_of_contra_keys, min(5,len(pairs_of_contra_keys)))
            non_contra_sample = random.sample(pairs_of_non_contra_keys, min(5,len(pairs_of_non_contra_keys)))
            sampled_keys = set(contra_sample).union(set(non_contra_sample))
            assert len(set(contra_sample).intersection(set(non_contra_sample))) == 0
        else:
            min_num_samples = min(len(pairs_of_contra_keys), len(pairs_of_non_contra_keys))
            contra_sample = random.sample(pairs_of_contra_keys, min_num_samples)
            non_contra_sample = random.sample(pairs_of_non_contra_keys, min_num_samples)
            sampled_keys = set(contra_sample).union(set(non_contra_sample))
            assert len(set(contra_sample).intersection(set(non_contra_sample))) == 0
    else:
        sampled_keys=pairs_of_keys
        
    contra_count = 0
    non_contra_count = 0
    # iterate over all the i_by to see if there is intersection between the parents
    # if there is, then they are same side of contra spectrum
    for key_i,key_j in sampled_keys:
        assert key_i != key_j
        parents_i = set(dict_interpreted_by[key_i])
        parents_j = set(dict_interpreted_by[key_j])
        descendents_i = key_i.descendant_concepts()
        descendents_j = key_j.descendant_concepts()

        parent_intersection = parents_i.intersection(parents_j)

        i_interpretation = set([x.label[0] for x in key_i.has_interpretation])
        j_interpretation = set([x.label[0] for x in key_j.has_interpretation])
        i_j_interpretation_intersection = i_interpretation.intersection(j_interpretation)
        
        assert len(i_interpretation) > 0 and len(j_interpretation) > 0
        # in theory: dont contradict?
        # if len(parent_intersection) == 0:
        assert len(key_i.label) == 1
        assert len(key_j.label) == 1
        count+=1

        i_cui_ids = tuple([x.name for x in SNOMEDCT_US[key_i.name].unifieds])
        j_cui_ids = tuple([x.name for x in SNOMEDCT_US[key_j.name].unifieds])

        key_i_label = f"{str(i_interpretation)}: {key_i.label[0]}|{i_cui_ids}|{key_i.name}"
        key_j_label = f"{str(j_interpretation)}: {key_j.label[0]}|{j_cui_ids}|{key_j.name}"
        assert len(i_cui_ids) > 0
        assert len(j_cui_ids) > 0

        predicted_label = determine_label((key_i.label[0],key_j.label[0]),(i_interpretation,j_interpretation))

        if predicted_label == "1":
            assert (key_i,key_j) in pairs_of_contra_keys or (key_j,key_i) in pairs_of_contra_keys

            if (key_i_label,key_j_label) not in phrase_dict["contra"] and \
            (key_j_label,key_i_label) not in phrase_dict["contra"]:
                phrase_dict["contra"].add((key_i_label,key_j_label))
                contra_count+=1
                
            for i_int in i_interpretation:
                for j_int in j_interpretation:
                    pair1 = (i_int,j_int)
                    pair2 = (j_int,i_int)

                    if pair1 not in single_word_dict["contra"] and pair2 not in single_word_dict["contra"]:
                        single_word_dict["contra"].add(pair1)                
        elif predicted_label == "0":
            assert (key_i,key_j) in pairs_of_non_contra_keys or (key_j,key_i) in pairs_of_non_contra_keys


            # sometimes the pairs are already populated by means of getting there through a different
            # ancestor
            if (key_i_label,key_j_label) not in phrase_dict["non-contra"] and \
            (key_j_label,key_i_label) not in phrase_dict["non-contra"]:
                phrase_dict["non-contra"].add((key_i_label,key_j_label))
                non_contra_count+=1
                
            for i_int in i_interpretation:
                for j_int in j_interpretation:
                    pair1 = (i_int,j_int)
                    pair2 = (j_int,i_int)

                    if pair1 not in single_word_dict["non-contra"] and pair2 not in single_word_dict["non-contra"]:
                        single_word_dict["non-contra"].add(pair1) 
                        
        else:
            assert False

elements_with_interpretations = 0
set_of_has_interpretations = set()
single_word_dict = {"contra":set(),"non-contra":set()}
phrase_dict = {"contra":set(),"non-contra":set()}

num_interpretations= []
inter_parent_to_sid = {}

num_children = 0
relev_children = []

# can iteratate over subfields as well...
# for child in all_nodes:
for child in all_cardio_set_of_children:
    num_children+=1
    child_concept = SNOMEDCT_US[child]
    interpreted_by = child_concept.is_interpreted_by
    inter_parent_to_sid[child_concept.label[0]] = child
        
    if interpreted_by != []:
        num_interpretations.append((len(interpreted_by),child_concept.label[0]))

    # can also filter by group size here, by adding additional conditional.  
    # len(interpreted_by) is the group size
    if interpreted_by != [] and len(interpreted_by) < 25:
        relev_children.append(child_concept.label[0])
        determine_interpretation_relationship(interpreted_by, single_word_dict, phrase_dict,sample=False)
        for i_by in interpreted_by:
            assert len(i_by.label) == 1
            has_interpretations = [x.label[0] for x in i_by.has_interpretation]
            for item in has_interpretations: set_of_has_interpretations.add(item)
            if len(has_interpretations) > 0:
                elements_with_interpretations += 1