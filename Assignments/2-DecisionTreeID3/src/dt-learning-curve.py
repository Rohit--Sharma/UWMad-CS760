# import dt_learn as dt
import sys
import numpy as np
import pandas as pd
from dt_learn import dt_learn_id3, dt_test, import_data
import matplotlib.pyplot as plt
import os.path


# Driver Code
def main():
    if len(sys.argv) != 4:
        print 'Usage: dt-learning-curve <dataset> <trainset> m'
        sys.exit(1)

    for domain in ['heart', 'diabetes']:
        if not os.path.isfile(domain + '_learning.pkl'):
            training_data_file_path = 'dataset/' + domain + '_train.arff'  # + sys.argv[1]
            testing_data_file_path = 'dataset/' + domain + '_test.arff'  # + sys.argv[2]
            m = 4  # int(sys.argv[3])

            dataset, metadata = import_data(training_data_file_path)
            # features = [feature for feature in metadata.names()[:-1] if metadata[feature][0] != 'numeric']
            features = metadata.names()[:-1]
            target_attrib = metadata.names()[-1]

            training_set_sizes = [0.05, 0.10, 0.20, 0.50]

            mean_accuracies = []
            max_accuracies = []
            min_accuracies = []
            for s in training_set_sizes:
                accuracy = []
                for sampling_iter in range(10):
                    sub_dataset = dataset.sample(frac=s)
                    decision_tree = dt_learn_id3(sub_dataset, m, metadata, features, target_attrib, None)
                    # print decision_tree
                    accuracy.append(dt_test(decision_tree, testing_data_file_path, False))
                print str(s) + ': (' + str(np.max(accuracy)) + ', ' + str(np.mean(accuracy)) + ', ' + str(
                    np.min(accuracy)) + ')'
                mean_accuracies.append(np.mean(accuracy))
                max_accuracies.append(np.max(accuracy))
                min_accuracies.append(np.min(accuracy))
            training_set_sizes.append(1.00)
            decision_tree = dt_learn_id3(dataset, m, metadata, features, target_attrib, None)
            full_data_accuracy = dt_test(decision_tree, testing_data_file_path, False)
            mean_accuracies.append(full_data_accuracy)
            max_accuracies.append(full_data_accuracy)
            min_accuracies.append(full_data_accuracy)

            training_set_sizes = [s * dataset.shape[0] for s in training_set_sizes]

            # Data
            df = pd.DataFrame({'x': training_set_sizes, 'min': min_accuracies, 'mean': mean_accuracies,
                               'max': max_accuracies})
            df.to_pickle(domain + '_learning.pkl')

    for domain in ['heart', 'diabetes']:
        plot(domain, 'learning')


def pred_accuracy_vs_tree_size():
    m_values = [2, 5, 10, 20]
    for domain in ['heart', 'diabetes']:
        if not os.path.isfile(domain + '_pred_accuracy.pkl'):
            training_data_file_path = 'dataset/' + domain + '_train.arff'  # + sys.argv[1]
            testing_data_file_path = 'dataset/' + domain + '_test.arff'  # + sys.argv[2]

            dataset, metadata = import_data(training_data_file_path)
            # features = [feature for feature in metadata.names()[:-1] if metadata[feature][0] != 'numeric']
            features = metadata.names()[:-1]
            target_attrib = metadata.names()[-1]

            test_set_accuracy = []
            for m in m_values:
                decision_tree = dt_learn_id3(dataset, m, metadata, features, target_attrib, None)
                test_set_accuracy.append(dt_test(decision_tree, testing_data_file_path, False))

            df = pd.DataFrame({'x': m_values, 'acc': test_set_accuracy})
            df.to_pickle(domain + '_pred_accuracy.pkl')

    for domain in ['heart', 'diabetes']:
        plot(domain, 'pred_acc')


def plot(domain, type):
    if type == 'learning':
        df = pd.read_pickle(domain + '_learning.pkl')
        plt.plot('x', 'min', data=df, marker='x', color='olive', linewidth=2, linestyle='dashed')
        plt.plot('x', 'mean', data=df, marker='o', color='olive', linewidth=2)
        plt.plot('x', 'max', data=df, marker='x', color='olive', linewidth=2, linestyle='dashed')
        plt.legend()
        plt.xlabel("Sample set size")
        plt.ylabel("Accuracy")
        plt.title("Decision Tree Learning Curve - " + domain)
        plt.show()
    elif type == 'pred_acc':
        df = pd.read_pickle(domain + '_pred_accuracy.pkl')
        plt.plot('x', 'acc', data=df, marker='o', color='olive', linewidth=2)
        plt.xlabel("Tree size (m)")
        plt.ylabel("Accuracy")
        plt.title("Decision Tree Accuracy vs Tree Size - " + domain)
        plt.show()


# Calling the main function
if __name__ == '__main__':
    # main()
    pred_accuracy_vs_tree_size()
