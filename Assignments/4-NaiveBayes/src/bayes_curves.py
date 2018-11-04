import pandas as pd
import numpy as np
import bayes
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold


def import_data(dataset_arff_path):
    """
    Importing the arff file
    :param dataset_arff_path: String, the path of the arff file to be imported
    :return: DataFrame and Metadata for the file content
    """
    data, meta = arff.loadarff(dataset_arff_path)
    return pd.DataFrame(data), meta


def main():
    data, meta = import_data('../dataset/chess-KingRookVKingPawn.arff')

    num_folds = 10
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=7)
    # n_bayes_correct_preds = 0
    # n_bayes_total_preds = 0
    # tan_correct_preds = 0
    # tan_total_preds = 0
    for train_indices, test_indices in skf.split(data.iloc[:, :-1], data.iloc[:, -1]):

        train_set = data.iloc[train_indices]
        test_set = data.iloc[test_indices]

        nb = bayes.NaiveBayes(train_set, meta, test_set)
        nb.train()

        # n_bayes_correct_preds += nb.test()
        # n_bayes_total_preds += len(test_set)

        tan = bayes.TAN(train_set, meta, test_set)
        tan.train()
        # tan_correct_preds += tan.test()
        # tan_total_preds += len(test_set)
        # print 'Fold: '
        fold_nb_correct_preds = nb.test()
        fold_tan_correct_preds = tan.test()
        print 'Naive Bayes:', fold_nb_correct_preds, len(test_set)  # , float(nb.test()) / len(test_set)
        print 'TAN:', fold_tan_correct_preds, len(test_set)  # , float(tan_correct_preds) / tan_total_preds
    # print 'Overall: '
    # print n_bayes_correct_preds, n_bayes_total_preds, float(n_bayes_correct_preds) / n_bayes_total_preds
    # print tan_correct_preds, tan_total_preds, float(tan_correct_preds) / tan_total_preds


if __name__ == '__main__':
    main()
