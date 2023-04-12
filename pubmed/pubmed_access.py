import ftplib
import gzip
import xml.etree.ElementTree as ET
import pandas as pd
import os 
import sys
import multiprocessing as mp
import time
EMAIL="dmakhervaks@hmc.edu"

def download_pubmed(to_folder):
    host = 'ftp.ncbi.nlm.nih.gov'
    ftp = ftplib.FTP(host, passwd=EMAIL)
    ftp.login()
    ftp.cwd("pubmed/baseline/")
    for i in range(1, 973):
    #for i in range(5, 6):
        fname = "pubmed22n{:04}.xml.gz".format(i)
        print("downloading file {}".format(fname))
        ftp.retrbinary('RETR {}'.format(fname), open(os.path.join(to_folder, fname), "wb").write)
		
def get_first(xml_object, xpath):
    results = xml_object.findall(xpath)
    if len(results) == 0 or results[0].text is None:
        return ""
    return results[0].text.replace("\t", " ")


def get_values_list(xml_object, xpath):
    results = xml_object.findall(xpath)
    if len(results) == 0:
        return []
    # return results[0].getchildren()
    return list(results[0])


def extract_date(xml_object):
    date_objs = xml_object.findall(".//PubMedPubDate")
    if len(date_objs) == 0:
        return ""
    chosen_date = None
    if len(date_objs) == 1:
        chosen_date = date_objs[0]
    if len(date_objs) > 1:
        for date in date_objs:
            if 'PubStatus' in date.attrib.keys() and date.attrib['PubStatus'] == 'pubmed':
                chosen_date = date
        if not chosen_date:
            chosen_date = date_objs[0]
    # Create a date string
    year = get_first(chosen_date, 'Year')
    month = get_first(chosen_date, 'Month')
    day = get_first(chosen_date, 'Day')
    return "{}/{}/{}".format(year, month, day)
	

def sanitize(string, forbidden_chars=["\r","\n",",", ";"]):
    s = string
    if s is None:
        return ""
    for c in forbidden_chars:
        s = s.replace(c, " ")
    return s


def parse_pubmed_xml_to_dataframe(idx, folder="baseline", search_for_pmid=None):
    just_clinical_trials = True
    num_clinical_trials = 0
    idx = int(idx)
    start_idx, end_idx = idx, idx
    for i in range(start_idx, end_idx + 1):
        data = {}
        folder = "baseline"
        fname = os.path.join(folder, "pubmed22n{:04}.xml.gz".format(i))
        print("working on file: {}".format(fname))
        with gzip.open(fname, "rb") as g:
            content = g.read()
            retry_count = 0
            while retry_count < 3:
                try:
                    root = ET.fromstring(content)
                    break
                except:
                    print("Retrying")
                    time.sleep(10)
                    retry_count+=1
        # for pubmed_article in root.getchildren():
        for pubmed_article in list(root):
            pmid = pd.to_numeric(get_first(pubmed_article, "MedlineCitation/PMID"))
            if just_clinical_trials:
                stringified = ET.tostring(pubmed_article, encoding="unicode")
                if "NCT0" in stringified or "NCT1" in stringified or "NCT2" in stringified or "NCT3" in stringified or "NCT4" in stringified or "NCT5" in stringified or "NCT6" in stringified or "NCT7" in stringified or "NCT8" in stringified or "NCT9" in stringified:
                    num_clinical_trials+=1
                else:
                    continue
            # abstract_text = get_first(pubmed_article, "MedlineCitation/Article/Abstract/AbstractText")
            abstract_texts = get_values_list(pubmed_article, "MedlineCitation/Article/Abstract") 
            if search_for_pmid is not None and pmid == search_for_pmid:
                return pubmed_article
            if not abstract_texts:  # Skip empty abstracts.
                #print("skipping empty abstract")
                continue
            if len(abstract_texts) > 1:
                print("fname:{} pmid:{} has more than one abstracttexts".format(fname, pmid))
            #print("num abs texts: {}".format(len(abstract_texts)))
            abstract_text_as_string = ";".join([sanitize(item.text) for item in abstract_texts if item is not None])
            abstract_labels = ";".join([item.attrib.get('Label', "") for item in abstract_texts if item is not None])
            title = get_first(pubmed_article, "MedlineCitation/Article/ArticleTitle")
            pub_types = ";".join([sanitize(item.text) for item in
                                  pubmed_article.findall(".//PublicationType")
                                  if item is not None])
            date = pd.to_datetime(extract_date(pubmed_article))
            keywords = pubmed_article.findall(".//Keyword")
            kw_list = []
            for item in keywords:
                if item is None or item.text is None:
                    continue
                words = item.text.split(",")
                if len(words) > 1:
                    kw_list.extend(words)
                elif len(words[0])>0:
                    kw_list.append(words[0])
            kw_as_text = ";".join([sanitize(word) for word in kw_list])
            desc = pubmed_article.findall(".//MeshHeading/DescriptorName")
            mesh = ";".join([sanitize(d.text) for d in desc])
            
            title = title.replace("\t", " ")
            abstract_text_as_string = abstract_text_as_string.replace("\t", " ")
            abstract_labels = abstract_labels.replace("\t", " ")
            pub_types = pub_types.replace("\t", " ")
            fname = fname.replace("\t", " ")
            mesh = mesh.replace("\t", " ")
            kw_as_text = kw_as_text.replace("\t", " ")
            
            data[pmid] = {'title': title, 'abstract': abstract_text_as_string, 'labels':abstract_labels,
                          'pub_types':pub_types, 'date': date, 'file': fname,
                          'mesh_headings': mesh, 'keywords': kw_as_text}
        df = pd.DataFrame.from_dict(data, orient='index')
        if just_clinical_trials:
            df.to_csv(os.path.join(folder+"_clinical_trials", "pubmed22n{:04}.tsv".format(i)),sep='\t')
        else:
            df.to_csv(os.path.join(folder, "pubmed22n{:04}.tsv".format(i)),sep='\t')
            
    return num_clinical_trials
            
POOLSIZE  = 8 # number of CPUs
pool = mp.Pool(POOLSIZE)
fnames = [str(x) for x in list(range(1,1115))]
total_trials = 0
for x in pool.imap_unordered(parse_pubmed_xml_to_dataframe, fnames, 1):
   total_trials+=x
print(f"Total trials: {total_trials}")

# NOTE: to run on single core
# start = int(sys.argv[1])
# num_trials = parse_pubmed_xml_to_dataframe(start,start)
# print(start, num_trials)
