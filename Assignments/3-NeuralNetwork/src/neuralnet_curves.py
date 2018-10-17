import sys
import numpy as np
import pandas as pd
import neuralnet
import matplotlib.pyplot as plt
import os.path
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold


def plot_curve(x_vals, y_vals, x_label, y_label, title):
    df = pd.DataFrame({'x_vals': x_vals, 'y_vals': y_vals})
    plt.plot('x_vals', 'y_vals', data=df, marker='o', color='olive', linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def main():
    training_data_file_path = 'dataset/sonar.arff'
    data, meta = arff.loadarff(training_data_file_path)
    data = np.asarray(data.tolist())

    # Part B (1)
    print 'Epochs curve'
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=73)

    epochs = [25, 50, 75, 100]
    accuracy = []
    for epoch in epochs:
        predicted_confidences = np.zeros(len(data))
        fold_numbers = np.zeros(len(data), dtype=int)
        curr_fold_num = 0
        for train_indexes, test_indexes in skf.split(data[:, :-1].astype(float), data[:, -1]):
            fold_numbers[test_indexes] = curr_fold_num
            curr_fold_num += 1

            X_train, X_test = data[train_indexes, :-1].astype(float), data[test_indexes, :-1].astype(float)
            y_train, y_test = data[train_indexes, -1], data[test_indexes, -1]
            neural_net = neuralnet.NeuralNetwork(len(meta.names()) - 1, len(meta.names()) - 1, 1, 0.1, epoch)
            for _ in range(neural_net.epoch):
                for i in range(len(X_train)):
                    neural_net.train(X_train[i], [0.0] if y_train[i] == 'Rock' else [1.0])
            # fold_accuracy = neural_net.test_neural_net(data, test_indexes, predicted_confidences)
            neural_net.test_neural_net(data, test_indexes, predicted_confidences)
            # print 'Fold accuracy:', fold_accuracy

        correct_preds = 0
        for i in range(len(data)):
            predicted_label = meta[meta.names()[-1]][1][0] if predicted_confidences[i] < 0.5 else meta[meta.names()[-1]][1][
                1]
            actual_label = data[i, -1]
            if predicted_label == actual_label:
                correct_preds += 1
        # print '(', epoch, ',', correct_preds * 1.0 / len(data), ')'
        accuracy.append(correct_preds * 1.0 / len(data))
    plot_curve(epochs, accuracy, 'Epoch', 'Accuracy', 'Accuracy vs Epoch')

    # Part B (2)
    print 'Folds curve'
    num_folds = [5, 10, 15, 20, 25]
    accuracy = []
    for folds in num_folds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=73)
        predicted_confidences = np.zeros(len(data))
        fold_numbers = np.zeros(len(data), dtype=int)
        curr_fold_num = 0
        for train_indexes, test_indexes in skf.split(data[:, :-1].astype(float), data[:, -1]):
            fold_numbers[test_indexes] = curr_fold_num
            curr_fold_num += 1

            X_train, X_test = data[train_indexes, :-1].astype(float), data[test_indexes, :-1].astype(float)
            y_train, y_test = data[train_indexes, -1], data[test_indexes, -1]
            neural_net = neuralnet.NeuralNetwork(len(meta.names()) - 1, len(meta.names()) - 1, 1, 0.1, 50)
            for _ in range(neural_net.epoch):
                for i in range(len(X_train)):
                    neural_net.train(X_train[i], [0.0] if y_train[i] == 'Rock' else [1.0])
            # fold_accuracy = neural_net.test_neural_net(data, test_indexes, predicted_confidences)
            neural_net.test_neural_net(data, test_indexes, predicted_confidences)
            # print 'Fold accuracy:', fold_accuracy

        correct_preds = 0

        for i in range(len(data)):
            predicted_label = meta[meta.names()[-1]][1][0] if predicted_confidences[i] < 0.5 else \
            meta[meta.names()[-1]][1][
                1]
            actual_label = data[i, -1]
            if predicted_label == actual_label:
                correct_preds += 1
        # print '(', folds, ',', correct_preds * 1.0 / len(data), ')'
        accuracy.append(correct_preds * 1.0 / len(data))
    plot_curve(num_folds, accuracy, 'Number of folds', 'Accuracy', 'Accuracy vs Number of folds')

    # Part B (3)
    print 'ROC curve'
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=59)

    predicted_confidences = np.zeros(len(data))
    fold_numbers = np.zeros(len(data), dtype=int)
    curr_fold_num = 0
    for train_indexes, test_indexes in skf.split(data[:, :-1].astype(float), data[:, -1]):
        fold_numbers[test_indexes] = curr_fold_num
        curr_fold_num += 1

        X_train, X_test = data[train_indexes, :-1].astype(float), data[test_indexes, :-1].astype(float)
        y_train, y_test = data[train_indexes, -1], data[test_indexes, -1]
        neural_net = neuralnet.NeuralNetwork(len(meta.names()) - 1, len(meta.names()) - 1, 1, 0.1, 50)
        for _ in range(neural_net.epoch):
            for i in range(len(X_train)):
                neural_net.train(X_train[i], [0.0] if y_train[i] == 'Rock' else [1.0])
        # fold_accuracy = neural_net.test_neural_net(data, test_indexes, predicted_confidences)
        neural_net.test_neural_net(data, test_indexes, predicted_confidences)
        # print 'Fold accuracy:', fold_accuracy

    roc_input = []
    for i in range(len(data)):
        predicted_label = meta[meta.names()[-1]][1][0] if predicted_confidences[i] < 0.5 else meta[meta.names()[-1]][1][
            1]
        actual_label = data[i, -1]
        # print fold_numbers[i], predicted_label, actual_label, predicted_confidences[i]
        roc_input.append((predicted_confidences[i], actual_label))

    # Sort in decreasing order of positive confidence
    x_fpr_vals = y_trp_vals = []
    roc_input.sort(reverse=True)
    num_neg = len([val for val in roc_input if val[1] == 'Rock'])
    num_pos = len([val for val in roc_input if val[1] == 'Mine'])
    tp = fp = last_tp = 0
    for i in range(len(roc_input)):
        if i > 0 and roc_input[i][0] != roc_input[i - 1][0] and roc_input[i][1] == 'Rock' and tp > last_tp:
            # print '(', fp * 1.0 / num_neg, ',', tp * 1.0 / num_pos, ')'
            x_fpr_vals.append(fp * 1.0 / num_neg)
            y_trp_vals.append(tp * 1.0 / num_pos)
            last_tp = tp
        if roc_input[i][1] == 'Mine':
            tp += 1
        else:
            fp += 1
    # print '(', fp * 1.0 / num_neg, ',', tp * 1.0 / num_pos, ')'
    x_fpr_vals.append(fp * 1.0 / num_neg)
    y_trp_vals.append(tp * 1.0 / num_pos)

    plot_curve(x_fpr_vals, y_trp_vals, 'False Positive Rate', 'True Positive Rate', 'ROC Curve')


if __name__ == '__main__':
    main()
