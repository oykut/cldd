'''
- Extacts vocabulary of a dataset to extract their concepts
- Lemmatize them
- Concept extraction is performed in Java
- Preprocess concepts that is obtained from Java

:param:
    dataset (uni, movie, title)


:default:
    lang  (de)
    embedding (fasttext)
    flag (1)   -> with OOV version
'''

import pprint
import treetaggerwrapper
import re
import six
import pickle
from nltk.tokenize import word_tokenize
import pandas as pd

def lemmatize(topic, main_lang, file_lang):
    # topic = "uni"
    # main_lang = "fr"
    # file_lang = "fr"
    in_path = "/Users/oyku/Desktop/blocked/vocabulary/" + main_lang + "/" + topic + "_" + file_lang + ".p"
    out_path = "/Users/oyku/Desktop/blocked/lemmatized/" + main_lang + "/" + topic + "_" + file_lang + "_lemmatized.txt"


    fileout = open(out_path, "w")
    tagger = treetaggerwrapper.TreeTagger(TAGLANG=file_lang)


    vocab = pickle.load(open(in_path, "rb"))
    print(file_lang + " vocabulary: " + str(len(vocab)))

    for line in vocab:

        if (line.strip().isdigit()):
            continue

        else:
            tags = tagger.tag_text(line.strip())
            tags2 = treetaggerwrapper.make_tags(tags)

            fileout.write(line.strip() + "," + tags2[0][2] + "\n")


def pickle_wordnet(topic, main_lang, file_lang):
    path = "/Users/oyku/Desktop/blocked/"
    in_path = path + "wordnet/" + main_lang + "/" + topic + "_" + file_lang + "_wordnet.txt"
    out_path = path + "uwn/" + main_lang + "/" + topic + "_" + file_lang + ".p"

    f = open(in_path, "r")
    content = f.readlines()

    wordnet_dict = {}
    for line in content:
        senses = {}
        data = line.split(",")

        i = 1
        while i < len(data):
            senses[data[i]] = data[i + 1]
            i += 2

        wordnet_dict[data[0]] = senses

    pickle.dump(wordnet_dict, open(out_path, "wb"))


def is_correct_type(field, dataset):
    # Checking for english

    col = "ltable_" + field

    en_type = "sth_else_en"
    de_type = "sth_else_de"
    column = dataset[col]
    column = column.dropna()

    type_list = list(set(column.map(type).tolist()))

    if len(type_list) == 1:
        returned_type = type_list[0]

        if returned_type == str or returned_type == six.unichr or returned_type == six.text_type:
            en_type = "str"

    # Checking for german

    col = "rtable_" + field

    column = dataset[col]
    column = column.dropna()

    type_list = list(set(column.map(type).tolist()))

    if len(type_list) == 1:
        returned_type = type_list[0]

        if returned_type == str or returned_type == six.unichr or returned_type == six.text_type:
            de_type = "str"

    return en_type == de_type



# param is dataframe. to be given by the function extract_vocabulary
def get_vocabulary(dataset):
    vocabulary = set()
    for column in dataset:
        for row in dataset[column]:

            row = re.sub(r'[^\w\s]', ' ', str(row))
            words = word_tokenize(row)

            for word in words:
                vocabulary.add(word.lower())

    return vocabulary



def extract_vocabulary(topic, lang):
    path = "/Users/oyku/Desktop/blocked/"
    # topic = "uni"
    # lang = "fr"
    data_path = path + topic + "_" + lang + "_blocked_original.csv"

    dataset = pd.read_csv(data_path)

    dropped = ['_id', 'ltable_id', 'rtable_id', 'Label']
    dataset = dataset.drop(dropped, axis=1)

    corres = set()

    for col in list(dataset):
        field = col[7:]
        if "ltable_" + field in list(dataset) and "rtable_" + field in list(dataset):
            if is_correct_type(field, dataset):
                corres.add(field)

    ## Arranging types

    en_cols = ["ltable_" + col for col in corres]
    en = dataset[en_cols]

    de_cols = ["rtable_" + col for col in corres]
    de = dataset[de_cols]

    en_vocabulary = get_vocabulary(en)
    de_vocabulary = get_vocabulary(de)

    pickle.dump(de_vocabulary, open(path + "vocabulary/" + lang + "/" + topic + "_" + lang + ".p", "wb"))
    pickle.dump(en_vocabulary, open(path + "vocabulary/" + lang + "/" + topic + "_en.p", "wb"))


def main(argv):
    topic = argv[0]
    lang = argv[1]

    # extract_vocabulary(topic, lang)
    # lemmatize(topic, lang, lang)
    # lemmatize(topic, lang, "en")
    pickle_wordnet(topic, lang, lang)
    pickle_wordnet(topic, lang, "en")
