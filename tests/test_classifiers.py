'''
Tests performance of classifiers on CLDD.

:param:
    dataset (uni, movie, title)

:default:
    lang  (de)
    embedding (fasttext)
    OOV (1)   -> with OOV version
'''

import sys
import datetime
import warnings
import pandas as pd
from itertools import chain

import lightgbm as lgbm
from xgboost import XGBClassifier
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def split_cols_into_set_features(X):
    '''
    Creates a list of list containing a list of features per column. i.e dataset = [name_a, name_b, country_a, country_b],
    the result is a list of a list with: [[name_a,name_b], [country_a, country_b]]
    :param X: the dataset
    :return: a list of list with the names of features per column in each list.
    '''
    columns = list(X)
    cols = []
    cols_indexes = []

    for i in range(0, len(columns)):
        c = columns[i].split('_')[0]
        group = []
        group_indexes = []
        con = lambda c2: c2.split('_')[0] == c and c2 not in list(chain.from_iterable(cols))
        for c2 in filter(con, columns):
            group.append(c2)
            group_indexes.append(X.columns.get_loc(c2))
        if len(group) != 0:
            cols.append(group)
            cols_indexes.append(group_indexes)

    return cols, cols_indexes


def main(argv):
    topic = argv[0]

    flag = "1"
    # flag = "0"

    lang = "de"
    # lang = "es"
    # lang = "fr"

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


    labeled = path + topic + "_" + lang + "_blocked_original.csv"
    labeled = pd.read_csv(labeled)
    print(labeled.shape)

    date = datetime.datetime.today().strftime('%Y-%m-%d')

    if flag == "1":
        method = embedding + "_oov"
        result_path = "/home/oyku/datasets/newexperiments/classifiers/" + topic + "_oov_" + lang + ".csv"
    else:
        method = embedding
        result_path = "/home/oyku/datasets/newexperiments/classifiers/" + topic + "_" + lang + ".csv"

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

    extra_features = embedding_features + hybrid_features + uwn_features
    features_index = [[el for el in range(0, len(extra_features))]]

    if flag == "1":
        print("Adding extra OOV treatment for numbers and still-OOV words")
        new_index = len(extra_features)
        features_index = [version + [new_index, new_index + 1] for version in features_index]
        extra_features = extra_features + oov_features

    magellan_fts_path = path + "features/" + topic + "_" + lang + "_magellan_features.csv"
    wordembed_fts_path = path + "features/" + topic + "_" + lang + "_" + method + "_features.csv"
    uwn_fts_path = path + "features/" + topic + "_" + lang + "_uwn_features.csv"

    magellan_features = pd.read_csv(magellan_fts_path)
    wordembed_features = pd.read_csv(wordembed_fts_path)
    uwn_features = pd.read_csv(uwn_fts_path)

    train_features = pd.concat([magellan_features, wordembed_features], axis=1)
    train_features = pd.concat([train_features, uwn_features], axis=1)

    print("Running classifiers experiment on " + topic + " dataset!")
    print("Training features:  " + str(len(list(train_features))))

    exclude = ["_id", "ltable_id", "rtable_id"]
    cols = [col for col in list(train_features) if col not in exclude]
    train_features = train_features[cols]

    ## Getting the names of the function names for versions
    feature_names_version = []
    for features in features_index:
        temp = []
        for index in features:
            temp.append(extra_features[index])
        feature_names_version.append(temp)

    feature_version = {}
    version_explanation = {}
    base_features = [col for col in list(train_features) if "crosslang" not in col]
    # features = train_features[base_features]
    # feature_version["Version 0"] = features
    # version_explanation["Version 0"] = "basic"
    for ind, feats in enumerate(feature_names_version):
        version = "Version " + str(ind + 1)
        version_explanation[version] = feats

        cols = [col for col in list(train_features) if any(col.endswith(feat) for feat in feats)]
        cols = base_features + cols
        feature_version[version] = train_features[cols]

    gold = pd.DataFrame(labeled["Label"])

    cols = ['Version', 'Classifier', 'F1', 'Recall', 'Precision']
    df = pd.DataFrame(columns=cols)

    for i in range(0, len(feature_version)):
        version = "Version " + str(i+1)
        features = feature_version[version]

        ### This part is necessary for stacking ###
        list_columns_with_features, list_indexes_columns_with_features = split_cols_into_set_features(features)
        pipelines = []
        for lst in list_indexes_columns_with_features:
            pipe = make_pipeline(ColumnSelector(cols=lst),
                                 XGBClassifier(random_state=7, n_estimators=350))
            pipelines.append(pipe)
        ####################################################

        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        scale = StandardScaler()
        imp.fit(features)
        imp.statistics_[pd.np.isnan(imp.statistics_)] = 0
        features = scale.fit_transform(imp.transform(features))

        models = []
        models.append(('XGB', XGBClassifier(random_state=7, n_estimators=350)))
        models.append(('LR', LogisticRegression(random_state=7)))
        models.append(('DT', DecisionTreeClassifier(random_state=7)))
        models.append(('RF', RandomForestClassifier(random_state=7)))
        models.append(('Ada', AdaBoostClassifier(random_state=7, n_estimators=350)))
        models.append(('LGBM', lgbm.LGBMClassifier(objective='binary', random_state=7)))
        models.append(('SVM', SVC(random_state=7, C=10, gamma=0.001)))
        models.append(('Stacking_probs', StackingClassifier(classifiers=pipelines,
                                                            meta_classifier=XGBClassifier(random_state=7,
                                                                                          use_probas=True,
                                                                                          average_probas=False))))


        print(version)

        for name, model in models:
            kfold = model_selection.StratifiedKFold(n_splits=5, random_state=7)
            scoring = ['f1', 'recall', 'precision']
            scores = model_selection.cross_validate(model, features, gold.values.ravel(), cv=kfold, scoring=scoring)
            f1 = "%.3f (%.3f)" % (scores['test_f1'].mean() * 100, scores['test_f1'].std() * 100)
            recall = "%.3f (%.3f)" % (scores['test_recall'].mean() * 100, scores['test_recall'].std() * 100)
            precision = "%.3f (%.3f)" % (scores['test_precision'].mean() * 100, scores['test_precision'].std() * 100)

            print("Classifier: %s --- F1: %s     Recall: %s      Precision: %s" % (name, f1, recall, precision))
            version_results = [version, name, f1, recall, precision]
            df.loc[len(df)] = version_results

    df.to_csv(result_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
