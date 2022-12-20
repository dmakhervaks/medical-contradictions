import gzip, csv
import random

header_line = None
contra_lines = []
non_contra_lines = []
header = None
train_count = 0
dev_count = 0
test_count = 0
new_lines = []
#714640
contra_count = 0
entail_count = 0
new_lines = []
# with open("data/However_Moreover_Balanced_Copy.tsv","r") as f:
#     mapping = {'contradiction':'contradiction','non-contradiction':'entailment'}
#     lines = f.readlines()
#     header = lines[0]
#     new_lines.append(header)
#     for line in lines[1:]:
#         new_lines.append(line)
#         split, dataset, filename, sent1, sent2, label = [x.strip() for x in line.split("\t")]
#         if mapping[label] == 'entailment':
#             entail_count+=1
#             if entail_count > 357320:
#                 sample = random.sample(lines,1)[0]
#                 _,_,_,s1,s2,_ = [x.strip() for x in sample.split("\t")]
#                 if entail_count%2 == 0:
#                     assert s1 != sent2
#                     new_lines.append("\t".join([split, dataset, filename, s1, sent2, "neutral"]) +"\n")
#                 else:
#                     assert s2 != sent1
#                     new_lines.append("\t".join([split, dataset, filename, sent1, s2, "neutral"]) +"\n")

#         elif mapping[label] == 'contradiction':
#             contra_count+=1
#             if contra_count > 357320:
#                 sample = random.sample(lines,1)[0]
#                 _,_,_,s1,s2,_ = [x.strip() for x in sample.split("\t")]
#                 if contra_count%2 == 0:
#                     assert s1 != sent2
#                     new_lines.append("\t".join([split, dataset, filename, s1, sent2, "neutral"]) +"\n")
#                 else:
#                     assert s2 != sent1
#                     new_lines.append("\t".join([split, dataset, filename, sent1, s2, "neutral"]) +"\n")
#         else:
#             assert False

# snomed_phrases = set()
# with open("snomed_phrases_v3.txt","r") as f:
#     lines = f.readlines()  
#     for line in lines:
#         snomed_phrases.add(line.strip())

# phrase_to_tokenized_phrase_map = {}
# for phrase in snomed_phrases:
#         phrase_words = phrase.split(" ")
#         phrase_to_tokenized_phrase_map[phrase]=phrase_words


# def does_sentence_contain_phrase_not_in_order(sent, phrase_to_tokenized_phrase_map):
#     phrases=[]
#     for phrase, word_tokens in phrase_to_tokenized_phrase_map.items():
#         all_found = True
#         for word in word_tokens:
#             # heuristic for whole-word-matching
#             if len(word) < 3:
#                 word = " " + word + " "
#             if word not in sent:
#                 all_found = False
#         if all_found:     
#             phrases.append(phrase)
#     return phrases

# def does_sentence_contain_exact_phrase(sent, phrase_to_tokenized_phrase_map):
#     phrases=[]
#     for phrase in phrase_to_tokenized_phrase_map:
#         phrase_words = phrase_to_tokenized_phrase_map[phrase]
#         sent_words = sent.split(" ")
#         window_size = len(phrase_words)
#         if len(sent_words) > len(phrase_words):
#             for i in range(len(sent_words) - window_size + 1):
#                 if phrase_words == sent_words[i: i + window_size]:
#                     phrases.append(phrase)
#         # if phrase in sent:
#         #     phrases.append(phrase)
    
#     return phrases

# cardio_words = {"cardio","heart","caridi","blood pressure","bp","myocard","ecg","pulse","vascular","artery","systol","diastol"}
# total_count = 0    
# cardio_count = 0
# with open("data/However_Moreover_Balanced_Copy.tsv","r") as f:
#     with open("data/However_Moreover_Balanced_Exact_Phrase.tsv","w") as fw:
#         new_lines = f.readlines()
#         total_count = len(new_lines)
#         fw.write(new_lines[0])
#         for line in new_lines[1:]:
#             split, dataset, filename, sent1, sent2, label = [x.strip() for x in line.split("\t")]
#             phrase1_list = does_sentence_contain_exact_phrase(sent1.lower(), phrase_to_tokenized_phrase_map)
#             phrase2_list = does_sentence_contain_exact_phrase(sent2.lower(), phrase_to_tokenized_phrase_map)
#             if len(phrase1_list) > 0 or len(phrase2_list) > 0:
#                 cardio_count+=1
#                 fw.write(line)


cardio_set = set()
with open("data/Cardio_Copy.tsv") as f:
    lines = f.readlines()[1:]
    for line in lines:
        split, dataset, filename, sent1, sent2, label = [x.strip() for x in line.split("\t")]
        cardio_set.add(sent1.lower())
        cardio_set.add(sent2.lower())

however_set = set()
with open("data/However_Moreover_Balanced_Copy.tsv") as f:
    lines = f.readlines()[1:]
    for line in lines:
        split, dataset, filename, sent1, sent2, label = [x.strip() for x in line.split("\t")]
        however_set.add(sent1.lower())
        however_set.add(sent2.lower())

print(len(cardio_set))
print(len(however_set))
print(however_set.intersection(cardio_set))