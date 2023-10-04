import re
import urllib.request, urllib.error, urllib.parse
import json
import os
from pprint import pprint
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
from itertools import repeat
from concurrent import futures
from urllib.error import HTTPError,URLError
import time

REST_URL = "http://data.bioontology.org"
API_KEY = "fc1a9972-02d6-4b17-9346-4a37240f6847"

def get_json(opener,url):
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())


def get_snomed_terms():
    snomed_set = set()
    # with open("mesh_to_snomed.tsv") as f:
    #     lines = f.readlines()[1:]
    #     for line in lines:
    #         snomed = line.split("\t")[1].strip()
    #         snomed_set.add(snomed)

    with open("snomed_all_phrases_cui.tsv") as f:
        lines = f.readlines()[1:]
        for line in lines:
            snomed = line.split("\t")[3].strip()
            snomed_set.add(snomed)
    return snomed_set


def write_snomed_pool(arguments): 
    list_of_terms,idx=arguments
    set_of_current_ids = set()
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    results_list = []
    with open(f"/home/davem/MeSH/snomed_results/{idx}.tsv","w") as fw:
        with open(f'list_of_dicts_{idx}.tsv', 'w') as fout:
            fw.write("UI\tterm\tprefLabel\tcui\tsemanticType\tsynonym\n")
            for i,curr in enumerate(list_of_terms):
                if i%100 == 0:
                    print(f"{idx}: {i}")

                for retry in range(1,4):
                    try:
                        # results = get_json(opener,REST_URL + f"/search?q={curr}")["collection"]
                        # print(results)
                        results = populate_new_dict_based_on_url(curr,opener)
                        results_list.append(results)
                        break
                    except HTTPError as err:
                        if err.code == 429:
                            print(f"sleeping: {20*retry}")
                            time.sleep(20*retry)
                        else:
                            print("ERROR")
                            raise

                if results != {}:
                    set_of_current_ids.add(curr)
                    fout.write(str(results) +"\n")

            # if len(results)==1:
            #     pref_label,cui,semantic_type,synonym = "","","",""
            #     if 'prefLabel' in results[0]:
            #         pref_label = results[0]['prefLabel']
            #     if 'cui' in results[0]:
            #         cui = results[0]['cui']
            #     if 'semanticType' in results[0]:
            #         semantic_type = results[0]['semanticType']
            #     if 'synonym' in results[0]:
            #         synonym = results[0]['synonym']
            #     fw.write(f"{curr}\t{curr}\t{pref_label}\t{cui}\t{semantic_type}\t{synonym}\n")

            # else:
            #     fw.write(f"MORE THAN 1 RESULTS for: {curr}\n")

    return set_of_current_ids


def get_mesh_cuis_pool(arguments): 
    term_to_identifier,list_of_terms,idx=arguments
    set_of_current_ids = set()
    opener = urllib.request.build_opener()

    with open(f"/home/davem/MeSH/results/{idx}.tsv","w") as fw:
        fw.write("UI\tterm\tprefLabel\tcui\tsemanticType\tsynonym\n")
        for i,term in enumerate(list_of_terms):
            if i%100 == 0:
                print(f"{idx}: {i}")
            curr = term_to_identifier[term]
            set_of_current_ids.add(curr)

            for retry in range(1,4):
                try:
                    results = get_json(opener,REST_URL + f"/search?q={curr}")["collection"]
                    break
                except HTTPError as err:
                    if err.code == 429:
                        time.sleep(20*retry)
                    else:
                        raise

            if len(results)==1:
                pref_label,cui,semantic_type,synonym = "","","",""
                if 'prefLabel' in results[0]:
                    pref_label = results[0]['prefLabel']
                if 'cui' in results[0]:
                    cui = results[0]['cui']
                if 'semanticType' in results[0]:
                    semantic_type = results[0]['semanticType']
                if 'synonym' in results[0]:
                    synonym = results[0]['synonym']
                fw.write(f"{curr}\t{term}\t{pref_label}\t{cui}\t{semantic_type}\t{synonym}\n")

            else:
                fw.write(f"MORE THAN 1 RESULTS for: {curr}\n")
                
    return set_of_current_ids


def get_mesh_cuis(term_to_identifier): 
    set_of_current_ids = set()
    opener = urllib.request.build_opener()

    with open(f"/home/davem/MeSH/results/total.tsv","w") as fw:
        fw.write("UI\tterm\tprefLabel\tcui\tsemanticType\tsynonym\n")
        for term in term_to_identifier:
            curr = term_to_identifier[term]
            print(curr)
            set_of_current_ids.add(curr)
            try:
                results = get_json(opener,REST_URL + f"/search?q={curr}")["collection"]
            except HTTPError as err:
                if err.code == 429:
                    time.sleep(60)
                else:
                    raise
            if len(results)==1:
                pref_label,cui,semantic_type,synonym = "","","",""
                if 'prefLabel' in results[0]:
                    pref_label = results[0]['prefLabel']
                if 'cui' in results[0]:
                    cui = results[0]['cui']
                if 'semanticType' in results[0]:
                    semantic_type = results[0]['semanticType']
                if 'synonym' in results[0]:
                    synonym = results[0]['synonym']
                # print(f"{curr}\t{term}\t{pref_label}\t{cui}\t{semantic_type}\t{synonym}")
                fw.write(f"{curr}\t{term}\t{pref_label}\t{cui}\t{semantic_type}\t{synonym}\n")
            else:
                fw.write(f"MORE THAN 1 RESULTS for: {curr}\n")

def process_mesh_ascii_file():
    terms = {}
    numbers = {}
    identifiers = set()
    term_to_identifier = {}
    mesh_terms_list = []
    identifiers_list = []

    meshFile = 'd2023.bin'

    diseases = {}

    mesh_terms = set()

    with open(meshFile, mode='rb') as file:

        mesh = file.readlines()

        outputFile = open('mesh.txt', 'w')
        count = 0
        for line in mesh:
            meshTerm = re.search(b'MH = (.+)$', line)


            if meshTerm:
                term = meshTerm.group(1)   
                mesh_terms.add(term.decode('utf-8').lower())
                mesh_terms_list.append(term.decode('utf-8').lower())
                
            meshNumber = re.search(b'MN = (.+)$', line)
            meshIdentifier = re.search(b'UI = (.+)$', line)
            if meshNumber:

                number = meshNumber.group(1)
                numbers[number.decode('utf-8')] = term.decode('utf-8')
                
                if number.decode('utf-8')[0] == 'C':
                    if term in diseases:
                        diseases[term] = diseases[term] + ' ' + number.decode('utf-8')

                    else:
                        diseases[term] = number.decode('utf-8')
                        
                if term in terms:

                    terms[term] = terms[term] + ' ' + number.decode('utf-8')
            
                else:

                    terms[term] = number.decode('utf-8')
                    
            if meshIdentifier:
                identifier = meshIdentifier.group(1)
                identifiers.add(identifier.decode('utf-8'))
                identifiers_list.append(identifier.decode('utf-8'))

        meshNumberList = []

        meshTermList = terms.keys()

        for term in meshTermList:
            item_list = terms[term].split(' ')
            for phrase in item_list:
                meshNumberList.append(phrase)


        meshNumberList.sort()

        for term,id in zip(mesh_terms_list,identifiers_list):
            assert term not in term_to_identifier
            term_to_identifier[term] = id

        return term_to_identifier

def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]


def parse_list(lst):
    pref_label_lst = []
    for elem in lst:
        pref_label_lst.append(elem["prefLabel"])

    return pref_label_lst


def try_link(opener,link):
    val_info = None
    for retry in range(1,5):
        try:
            val_info = json.loads(opener.open(link).read())
            break
        except HTTPError as err:
            if err.code == 429:
                print(f"HTTPError: sleeping for {link}: {20*retry}")
                time.sleep(20*retry)
            else:
                print("ERROR")
                raise
        except URLError as err:
            print(f"URLError: sleeping for {link}: {20*retry}")
            time.sleep(20*retry)

    return val_info


def populate_new_dict_based_on_url(u_id,opener):
    # result = get_json(opener,REST_URL + f"/search?q={u_id}")["collection"]
    result = try_link(opener,REST_URL + f"/search?q={u_id}")["collection"]
    new_dict = {}
    if result != []:
        for key,val in result[0].items():
            if key == "prefLabel":
                new_dict[key] = val
            elif key == "synonym":
                new_dict[key] = val
            elif key == "cui":
                new_dict[key] = val
            elif key == "semanticType":
                new_dict[key] = val
            elif key == "links":
                for key_links,val_links in val.items():
                    if key_links == "parents":
                        # val_info = json.loads(opener.open(val_links).read())
                        val_info = try_link(opener, val_links)
                        if val_info is not None and val_info != []:
                            new_dict[key_links] =  parse_list(val_info) 
                        else:
                            new_dict[key_links] = []
                    elif key_links == "children":
                        # val_info = json.loads(opener.open(val_links).read())
                        val_info = try_link(opener, val_links)
                        if val_info is not None and val_info != []:
                            new_dict[key_links] = parse_list(val_info["collection"])  
                        else:
                            new_dict[key_links] = []
                    elif key_links == "descendants":
                        # val_info = json.loads(opener.open(val_links).read())
                        val_info = try_link(opener, val_links)
                        if val_info is not None and val_info != []:
                            new_dict[key_links] = parse_list(val_info["collection"])  
                        else:
                            new_dict[key_links] = []
                    elif key_links == "ancestors":
                        # val_info = json.loads(opener.open(val_links).read())
                        val_info = try_link(opener, val_links)
                        if val_info is not None and val_info != []:
                            new_dict[key_links] = parse_list(val_info)   
                        else:
                            new_dict[key_links] = []

    return new_dict

# get_mesh_cuis(term_to_identifier)


if __name__ == "__main__":
    # TODO: WITH POOLING
    POOLSIZE  = 8 # number of CPUs
    POOLSIZE = 16
    pool = mp.Pool(POOLSIZE)
    opener = urllib.request.build_opener()
    

    # term_to_identifier = process_mesh_ascii_file()
    # list_of_terms = list(term_to_identifier.keys())
    # chunks = chunkify(list_of_terms,POOLSIZE)

    # assert set(chunks[0]) != set(chunks[1])

    # arguments = []

    # assert chunks[0]!= chunks[1]
    # for i,chunk in enumerate(chunks):
    #     arguments.append((term_to_identifier,chunk,i))

    snomed_set = get_snomed_terms()

    list_of_terms = list(snomed_set)
    chunks = chunkify(list_of_terms,POOLSIZE)
    assert set(chunks[0]) != set(chunks[1])

    arguments = []

    assert chunks[0]!= chunks[1]
    for i,chunk in enumerate(chunks):
        arguments.append((chunk,i))

    set_of_ids = set()

    pool = mp.Pool(POOLSIZE)

    total_perfect_matches = 0
    for x in pool.imap_unordered(write_snomed_pool, arguments, 1):
        total_perfect_matches+=len(x)

    print(total_perfect_matches)





    # with futures.ThreadPoolExecutor(POOLSIZE) as executor:
    #     curr_ids = executor.map(write_snomed_pool, arguments)
    # print(len(set_of_ids))
