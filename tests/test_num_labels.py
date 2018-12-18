'''

To test the system performance when smaller subsets of data (labels) used.

:param:
    dataset (uni, movie, title)


:default:
    lang  (de)
    embedding (fasttext)
    flag (1)   -> with OOV version
'''

import sys
import datetime
import warnings
import pandas as pd

from sklearn import model_selection
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def main(argv):
    topic = argv[0]
    flag = argv[1]

    lang = "de"
    # lang = "es"
    # lang = "fr"

    flag = "1"
    # flag = "0"

    # topic = "uni"
    # topic = "movie"
    # topic = "title"

    embedding = "fasttext"
    # embedding = "babylon"
    # embedding = "fbmuse"
    # embedding = "multicca"
    # embedding = "multiskip"
    # embedding = "multicluster"
    # embedding = "translationInvariance"
    # embedding = "bilbowa"

    if topic == "uni":
        path = "/home/oyku/datasets/University/"

    elif topic == "movie":
        path = "/home/oyku/datasets/Movie/"

    elif topic == "title":
        path = "/home/oyku/datasets/Article/"

    else:
        print("Wrong dataset is given. It should be either uni, movie or title.")
        return

    # --------------------------------------

    if lang not in ["de", "es", "fr"]:
        print("Wrong language is given. It should be either de, es or fr.")
        return

    # --------------------------------------
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    if flag == "0":
        method = embedding
        result_path = "/home/oyku/datasets/newexperiments/num_labels/" + topic + "_" + lang + "_baseline.csv"

    elif flag == "1":
        method = embedding + "_oov"
        result_path = "/home/oyku/datasets/newexperiments/num_labels/" + topic + "_oov_" + "_" + lang + "_baseline.csv"

    else:
        print("Specify flag as 0 or 1.")
        return

    # --------------------------------------

    labeled = path + topic + "_" + lang + "_blocked_translated.csv"
    labeled = pd.read_csv(labeled)
    print(labeled.shape)
    embedding_features = ['crosslang_mean_sim',  # 0
                          'crosslang_tfidf_mean_sim',  # 1
                          'crosslang_max_sim',  # 2
                          'crosslang_tfidf_max_sim',  # 3
                          'crosslang_tfidf_max_weight',  # 4
                          'crosslang_vector_composition',  # 5
                          'crosslang_greedy_aligned_words',  # 6
                          'crosslang_weighted_greedy_aligned_words',  # 7
                          'crosslang_optimal_alignment',  # 8
                          'crosslang_sif']  # 9

    hybrid_features = ['crosslang_aligned_words_senses_jaccard',  # 10
                       'crosslang_weighted_aligned_words_senses_jaccard',  # 11
                       'crosslang_aligned_words_senses_path_sim',  # 12
                       'crosslang_weighted_aligned_words_senses_path_sim']  # 13

    uwn_features = ['crosslang_uwn_common_sense_weights',  # 14
                    'crosslang_uwn_sense_similarity_path',  # 15
                    'crosslang_uwn_sense_similarity_lch',  # 16
                    'crosslang_uwn_sense_similarity_wup',  # 17
                    'crosslang_uwn_sense_similarity_resnik',  # 18
                    'crosslang_uwn_sense_similarity_jcn',  # 19
                    'crosslang_uwn_sense_similarity_lin']  # 20

    oov_features = ['crosslang_sim_oov',  # 21
                    'crosslang_number_difference']  # 22

    # extra_features = embedding_features + hybrid_features + uwn_features
    # features_index = [[el for el in range(0, len(extra_features))]]
    extra_features = []
    features_index = []

    if flag == "1":
        print("Adding extra OOV treatment for numbers and still-OOV words")
        new_index = len(extra_features)
        features_index = [version + [new_index, new_index + 1] for version in features_index]
        extra_features = extra_features + oov_features

    #magellan_fts_path = path + "features/" + topic + "_" + lang + "_translated_magellan_features.csv"
    magellan_fts_path = path + "features/" + topic + "_" + lang + "_baseline_features.csv"
    wordembed_fts_path = path + "features/" + topic + "_" + lang + "_" + method + "_features.csv"
    uwn_fts_path = path + "features/" + topic + "_" + lang + "_uwn_features.csv"

    magellan_features = pd.read_csv(magellan_fts_path)
    wordembed_features = pd.read_csv(wordembed_fts_path)
    uwn_features = pd.read_csv(uwn_fts_path)

    # train_features = pd.concat([magellan_features, wordembed_features], axis=1)
    # train_features = pd.concat([train_features, uwn_features], axis=1)

    train_features = magellan_features

    print("Running features experiment on " + topic + " dataset!")
    print("Training features:  " + str(len(list(train_features))))

    exclude = ["_id", "ltable_id", "rtable_id"]
    cols = [col for col in list(train_features) if col not in exclude]
    train_features = train_features[cols]

    ## Getting the names of the functions for different combinations
    feature_names_version = []
    for features in features_index:
        temp = []
        for index in features:
            temp.append(extra_features[index])
        feature_names_version.append(temp)

    feature_version = {}
    version_explanation = {}


    # Todo: fix here when you want to put it on Github
    # In this one, I do not need any experiment with Magellan
    # base_features = [col for col in list(train_features) if "crosslang" not in col]
    base_features = [col for col in list(train_features) if "zzzzz" not in col]
    features = train_features[base_features]
    feature_version["Version 0"] = features
    version_explanation["Version 0"] = "basic"
    for ind, feats in enumerate(feature_names_version):
        version = "Version " + str(ind + 1)
        version_explanation[version] = feats

        cols = [col for col in list(train_features) if any(col.endswith(feat) for feat in feats)]
        cols = base_features + cols
        feature_version[version] = train_features[cols]

    gold = pd.DataFrame(labeled["Label"])

    cols = ['Features', 'Label_perc', 'Label_num', 'Precision', 'Recall', 'F1']
    df = pd.DataFrame(columns=cols)

    percentages = [0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2]

    for i in range(0, len(feature_version)):

        version = "Version " + str(i)
        features = feature_version[version]

        for perc in percentages:
            X_train, X_test, y_train, y_test = train_test_split(features, gold, stratify=gold, test_size=perc,
                                                                random_state=7)

            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            scale = StandardScaler()
            imp.fit(X_train)
            imp.statistics_[pd.np.isnan(imp.statistics_)] = 0
            X_train = scale.fit_transform(imp.transform(X_train))

            # Cross Validation
            model = XGBClassifier(random_state=7, n_estimators=350)
            kfold = model_selection.StratifiedKFold(n_splits=5, random_state=7)
            scoring = ['f1', 'recall', 'precision']
            scores = model_selection.cross_validate(model, X_train, y_train.values.ravel(), cv=kfold, scoring=scoring)
            f1 = "%.3f (%.3f)" % (scores['test_f1'].mean() * 100, scores['test_f1'].std() * 100)
            recall = "%.3f (%.3f)" % (scores['test_recall'].mean() * 100, scores['test_recall'].std() * 100)
            precision = "%.3f (%.3f)" % (scores['test_precision'].mean() * 100, scores['test_precision'].std() * 100)

            print("%s:  --- Precision: %s     Recall: %s      F1: %s" % (version, precision, recall, f1))
            label_percentage = 1 - perc
            label_num = int(len(X_train) * 0.8)
            version_results = [version_explanation[version], label_percentage, label_num, precision, recall, f1]
            df.loc[len(df)] = version_results

        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        scale = StandardScaler()
        imp.fit(features)
        imp.statistics_[pd.np.isnan(imp.statistics_)] = 0
        features = scale.fit_transform(imp.transform(features))

        # Cross Validation
        model = XGBClassifier(random_state=7, n_estimators=350)
        kfold = model_selection.StratifiedKFold(n_splits=5, random_state=7)
        scoring = ['f1', 'recall', 'precision']
        scores = model_selection.cross_validate(model, features, gold.values.ravel(), cv=kfold, scoring=scoring)
        f1 = "%.3f (%.3f)" % (scores['test_f1'].mean() * 100, scores['test_f1'].std() * 100)
        recall = "%.3f (%.3f)" % (scores['test_recall'].mean() * 100, scores['test_recall'].std() * 100)
        precision = "%.3f (%.3f)" % (scores['test_precision'].mean() * 100, scores['test_precision'].std() * 100)

        print("%s:  --- Precision: %s     Recall: %s      F1: %s" % (version, precision, recall, f1))
        label_percentage = 1
        label_num = int(len(features) * 0.8)
        version_results = [version_explanation[version], label_percentage, label_num, precision, recall, f1]
        df.loc[len(df)] = version_results

    df.to_csv(result_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
