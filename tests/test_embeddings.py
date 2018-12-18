'''
Tests performance of cross-language word embedding models on CLDD.

:param:
    dataset (uni, movie, title)

:default:
    lang  (de)
    classifier (XGBoost)
    flag (0) -> not OOV version
'''

import sys
import datetime
import warnings
import pandas as pd

from sklearn import model_selection
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def main(argv):
    topic = argv[0]

    lang = "de"
    # lang = "es"
    # lang = "fr"

    # flag = "1"
    flag = "0"

    # topic = "uni"
    # topic = "movie"
    # topic = "title"

    # embedding = "fasttext"
    # embedding = "babylon"
    # embedding = "fbmuse"
    # embedding = "multicca"
    # embedding = "multiskip"
    # embedding = "multicluster"
    # embedding = "translationInvariance"
    # embedding = "shuffle_5_300"

    embeddings = ["fasttext", "babylon", "fbmuse", "multicca", "multiskip", "multicluster",
                  "translationInvariance", "shuffle_3_300", "shuffle_5_300", "shuffle_7_300", "shuffle_10_300",
                  "shuffle_15_300", "shuffle_5_40", "shuffle_5_100", "shuffle_5_200", "shuffle_5_512"]

    # embeddings = ["shuffle_3_300", "shuffle_5_300", "shuffle_7_300", "shuffle_10_300", "shuffle_15_300",
    #               "shuffle_5_40", "shuffle_5_100", "shuffle_5_200", "shuffle_5_300", "shuffle_5_512"]

    # --------------------------------------
    if topic == "uni":
        path = "/home/oyku/datasets/University/"

    elif topic == "movie":
        path = "/home/oyku/datasets/Movie/"

    elif topic == "title":
        path = "/home/oyku/datasets/Article/"

    else:
        print("Wrong dataset is given. It should be either uni, movie or title!")
        return
    # --------------------------------------




    date = datetime.datetime.today().strftime('%Y-%m-%d')
    result_path = "/home/oyku/datasets/newexperiments/shuffle_embeddings/" + topic + "_" + date + ".csv"

    labeled = path + topic + "_" + lang + "_blocked_original.csv"
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

    extra_features = embedding_features + hybrid_features + uwn_features
    features_index = [[el for el in range(0, len(extra_features))]]

    print("Running embeddings test on " + topic + " dataset!")

    # Reading Magellan and UWN features here. They do not depend on embeddings.
    magellan_fts_path = path + "features/" + topic + "_" + lang + "_magellan_features.csv"
    uwn_fts_path = path + "features/" + topic + "_" + lang + "_uwn_features.csv"


    magellan_features = pd.read_csv(magellan_fts_path)
    uwn_features = pd.read_csv(uwn_fts_path)

    exclude = ["_id", "ltable_id", "rtable_id"]
    gold = pd.DataFrame(labeled["Label"])

    cols = ['Embedding', 'F1', 'Recall', 'Precision']
    df = pd.DataFrame(columns=cols)

    for embedding in embeddings:
        wordembed_fts_path = path + "features/" + topic + "_" + lang + "_" + embedding + "_features.csv"
        wordembed_features = pd.read_csv(wordembed_fts_path)

        train_features = pd.concat([magellan_features, wordembed_features], axis=1)
        train_features = pd.concat([train_features, uwn_features], axis=1)

        print("Training features:  " + str(len(list(train_features))))

        cols = [col for col in list(train_features) if col not in exclude]
        train_features = train_features[cols]

        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        scale = StandardScaler()
        imp.fit(train_features)
        imp.statistics_[pd.np.isnan(imp.statistics_)] = 0
        features = scale.fit_transform(imp.transform(train_features))

        # Cross Validation
        model = XGBClassifier(random_state=7, n_estimators=350)
        kfold = model_selection.StratifiedKFold(n_splits=5, random_state=7)
        scoring = ['f1', 'recall', 'precision']
        scores = model_selection.cross_validate(model, features, gold.values.ravel(), cv=kfold, scoring=scoring)
        f1 = "%.3f (%.3f)" % (scores['test_f1'].mean() * 100, scores['test_f1'].std() * 100)
        recall = "%.3f (%.3f)" % (scores['test_recall'].mean() * 100, scores['test_recall'].std() * 100)
        precision = "%.3f (%.3f)" % (scores['test_precision'].mean() * 100, scores['test_precision'].std() * 100)

        print("Embedding: %s --- F1: %s     Recall: %s      Precision: %s" % (embedding, f1, recall, precision))
        version_results = [embedding, f1, recall, precision]
        df.loc[len(df)] = version_results

    df.to_csv(result_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
