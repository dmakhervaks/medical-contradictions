def has_suffix(sentence, suffix):
    for w in sentence.split(" "):
        if suffix == w[-len(suffix):]:
            return True
    return False


def filter_mednli_by_field_single_sentence(path_to_mednli, path_to_words, path_to_acronyms=None):
    words = set()
    with open(path_to_words) as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip()
            if len(word)>1:
                words.add(word)

    if path_to_acronyms is not None:
        acryonms = set()
        with open(path_to_acronyms) as f:
            lines = f.readlines()
            for line in lines:
                acryonm = line.strip()
                if len(acryonm)>1:
                    acryonms.add(acryonm)

    field_lines = []
    with open(path_to_mednli) as f:
        lines = f.readlines()
        field_lines.append(lines[0])
        lines = lines[1:]
        for i in range(0,len(lines)):
            line = lines[i]

            split,_,_,s1,s2,label = line.split("\t")
            s1_words, s2_words = s1.split(" "), s2.split(" ")

            label = label.strip()

            field_bool=False
            if 'surgery' not in path_to_words:
                for w in words:
                    if w.lower() in s1.lower() or w.lower() in s2.lower():
                        field_bool=True

            else:    
                # surgery words are composed of suffixes
                for w in words:
                    if has_suffix(s1,w) or has_suffix(s2,w):
                        field_bool=True

            
            # check for acronyms... MUST be case-sensitive...
            if path_to_acronyms is not None:
                for a in acryonms:
                    if a in s1_words or w in s2_words:
                        field_bool=True

            if field_bool:
                field_lines.append(line)

    return field_lines


all_field_lines = []

# TODO: update path
path_to_mednli = ""
all_field_lines.append((filter_mednli_by_field_single_sentence("field_words/cardiology_words.tsv","field_acronyms/cardiology_acronyms.tsv"),"Cardio"))
all_field_lines.append((filter_mednli_by_field_single_sentence("field_words/endocrinology_words.tsv","field_acronyms/endocrinology_acronyms.tsv"),"Endocrinology"))
all_field_lines.append((filter_mednli_by_field_single_sentence("field_words/surgery_words.tsv"),"Surgery"))
all_field_lines.append((filter_mednli_by_field_single_sentence("field_words/gynecology_words.tsv","field_acronyms/gynecology_acronyms.tsv"),"Gynecology"))
all_field_lines.append((filter_mednli_by_field_single_sentence("field_words/obstetrics_words.tsv","field_acronyms/obstetrics_acronyms.tsv"),"Obstetrics"))

all_unique_field_lines = []
field_intersections_with_cardio_lines = []
for i in range(len(all_field_lines)):
    sets = []
    for x,y in all_field_lines[0:i]+all_field_lines[i+1:]:
        sets.append(set(x))

    filtered = set(all_field_lines[i][0]).difference(*sets)
    all_unique_field_lines.append((filtered,all_field_lines[i][1]))

for field_lines in all_field_lines[1:]:
    with open(f"MedNLI_{field_lines[1]}.tsv","w") as fw:
        for line in field_lines[0]:
            fw.write(line)