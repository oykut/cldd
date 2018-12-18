'''
- Takes cell value pairs and matching betweens terms and concepts
- Creates multilingual lexical knowledge base features
'''


import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic


# This class is for UWN

class Similarity_Uwn():

    def __init__(self):
        self.en_vocab = {}
        self.de_vocab = {}

        self.brown_ic = {}

    def prepare(self, embedding, lang, dataset, en_oov_pckle, de_oov_pckle, sense_pickle_en, sense_pickle_de):
        print("UWN features are preparing")

        self.en_vocab = pickle.load(open(sense_pickle_en, "rb"))
        self.de_vocab = pickle.load(open(sense_pickle_de, "rb"))

        self.brown_ic = wordnet_ic.ic('ic-brown.dat')

    def common_sense_weights(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return 0

        sum = 0
        for en_lemma in en_tokens:
            if en_lemma in self.en_vocab:

                for de_lemma in de_tokens:
                    if de_lemma in self.de_vocab:

                        for key, value in self.en_vocab[en_lemma].items():

                            if key in self.de_vocab[de_lemma]:
                                sum += min(float(self.en_vocab[en_lemma][key]), float(self.de_vocab[de_lemma][key]))

        return float(sum) / min(len(en_tokens), len(de_tokens))

    # Helper funtion to return the sense with max weight
    def get_max_sense(self, en_tokens, de_tokens):
        en_synsets = []
        for en_lemma in en_tokens:
            if en_lemma in self.en_vocab:
                score = 0
                synset = ""
                for key, value in self.en_vocab[en_lemma].items():
                    if float(value) > score:
                        score = float(value)
                        synset = key

                if synset[3:].strip().isdigit():
                    wn_synset = wn.synset_from_pos_and_offset(synset[2], int(synset[3:].strip()))
                    en_synsets.append((wn_synset, score))

        de_synsets = []
        for de_lemma in de_tokens:
            if de_lemma in self.de_vocab:
                score = 0
                synset = ""
                for key, value in self.de_vocab[de_lemma].items():
                    if float(value) > score:
                        score = float(value)
                        synset = key

                if synset[3:].strip().isdigit():
                    wn_synset = wn.synset_from_pos_and_offset(synset[2], int(synset[3:].strip()))
                    de_synsets.append((wn_synset, score))

        return en_synsets, de_synsets

    def sense_similarity_wup(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return 0

        en_synsets, de_synsets = self.get_max_sense(en_tokens, de_tokens)

        if len(en_tokens) <= len(de_tokens):
            outer = en_synsets
            inner = de_synsets
            small_length = len(en_tokens)

        else:
            outer = de_synsets
            inner = en_synsets
            small_length = len(de_tokens)

        weight_sum = 0
        for out_syn in outer:
            synset_sim = 0
            w1 = 0
            w2 = 0
            for in_syn in inner:
                sim = out_syn[0].wup_similarity(in_syn[0])
                if sim is None:
                    sim = 0
                if sim > synset_sim:
                    synset_sim = sim
                    w1 = out_syn[1]
                    w2 = in_syn[1]

                weight_sum += float(w1) * float(w2) * synset_sim

        return float(weight_sum) / small_length

    def sense_similarity_lch(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return 0

        en_synsets, de_synsets = self.get_max_sense(en_tokens, de_tokens)

        if len(en_tokens) <= len(de_tokens):
            outer = en_synsets
            inner = de_synsets
            small_length = len(en_tokens)

        else:
            outer = de_synsets
            inner = en_synsets
            small_length = len(de_tokens)

        weight_sum = 0
        for out_syn in outer:
            synset_sim = 0
            w1 = 0
            w2 = 0
            for in_syn in inner:

                if out_syn[0].pos() == in_syn[0].pos():
                    sim = out_syn[0].lch_similarity(in_syn[0])
                else:
                    sim = 0

                if sim is None:
                    sim = 0

                if sim > synset_sim:
                    synset_sim = sim
                    w1 = out_syn[1]
                    w2 = in_syn[1]

                weight_sum += float(w1) * float(w2) * synset_sim

        return float(weight_sum) / small_length

    def sense_similarity_path(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return 0

        en_synsets, de_synsets = self.get_max_sense(en_tokens, de_tokens)

        if len(en_tokens) <= len(de_tokens):
            outer = en_synsets
            inner = de_synsets
            small_length = len(en_tokens)

        else:
            outer = de_synsets
            inner = en_synsets
            small_length = len(de_tokens)

        weight_sum = 0
        for out_syn in outer:
            synset_sim = 0
            w1 = 0
            w2 = 0
            for in_syn in inner:
                sim = out_syn[0].path_similarity(in_syn[0])
                if sim is None:
                    sim = 0
                if sim > synset_sim:
                    synset_sim = sim
                    w1 = out_syn[1]
                    w2 = in_syn[1]

                weight_sum += float(w1) * float(w2) * synset_sim

        return float(weight_sum) / small_length

    def sense_similarity_resnik(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return 0

        en_synsets, de_synsets = self.get_max_sense(en_tokens, de_tokens)

        if len(en_tokens) <= len(de_tokens):
            outer = en_synsets
            inner = de_synsets
            small_length = len(en_tokens)

        else:
            outer = de_synsets
            inner = en_synsets
            small_length = len(de_tokens)

        weight_sum = 0
        for out_syn in outer:
            synset_sim = 0
            w1 = 0
            w2 = 0
            for in_syn in inner:

                if out_syn[0].pos() == in_syn[0].pos():
                    try:
                        sim = out_syn[0].res_similarity(in_syn[0], self.brown_ic)
                    except:
                        sim = 0

                else:
                    sim = 0

                if sim is None:
                    sim = 0

                if sim > synset_sim:
                    synset_sim = sim
                    w1 = out_syn[1]
                    w2 = in_syn[1]

                weight_sum += float(w1) * float(w2) * synset_sim

        return float(weight_sum) / small_length

    def sense_similarity_jcn(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return 0

        en_synsets, de_synsets = self.get_max_sense(en_tokens, de_tokens)

        if len(en_tokens) <= len(de_tokens):
            outer = en_synsets
            inner = de_synsets
            small_length = len(en_tokens)

        else:
            outer = de_synsets
            inner = en_synsets
            small_length = len(de_tokens)

        weight_sum = 0
        for out_syn in outer:
            synset_sim = 0
            w1 = 0
            w2 = 0
            for in_syn in inner:

                if out_syn[0].pos() == in_syn[0].pos():
                    try:
                        sim = out_syn[0].jcn_similarity(in_syn[0], self.brown_ic)
                    except:
                        sim = 0
                else:
                    sim = 0

                if sim is None:
                    sim = 0

                if sim > synset_sim:
                    synset_sim = sim
                    w1 = out_syn[1]
                    w2 = in_syn[1]

                weight_sum += float(w1) * float(w2) * synset_sim

        return float(weight_sum) / small_length

    def sense_similarity_lin(self, s1, s2):
        en_tokens = word_tokenize(s1.lower())
        de_tokens = word_tokenize(s2.lower())

        if len(en_tokens) == 0 or len(de_tokens) == 0:
            return 0

        en_synsets, de_synsets = self.get_max_sense(en_tokens, de_tokens)

        if len(en_tokens) <= len(de_tokens):
            outer = en_synsets
            inner = de_synsets
            small_length = len(en_tokens)

        else:
            outer = de_synsets
            inner = en_synsets
            small_length = len(de_tokens)

        weight_sum = 0
        for out_syn in outer:
            synset_sim = 0
            w1 = 0
            w2 = 0
            for in_syn in inner:

                if out_syn[0].pos() == in_syn[0].pos():
                    try:
                        sim = out_syn[0].lin_similarity(in_syn[0], self.brown_ic)
                    except:
                        sim = 0
                else:
                    sim = 0

                if sim is None:
                    sim = 0

                if sim > synset_sim:
                    synset_sim = sim
                    w1 = out_syn[1]
                    w2 = in_syn[1]

                weight_sum += float(w1) * float(w2) * synset_sim

        return float(weight_sum) / small_length
