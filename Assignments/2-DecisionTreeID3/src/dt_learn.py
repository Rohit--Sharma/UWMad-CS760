from scipy.io import arff
import numpy as np
import pandas as pd
import math
import sys


def import_data(dataset_arff_path):
    """
    Importing the arff file
    :param dataset_arff_path: String, the path of the arff file to be imported
    :return: DataFrame and Metadata for the file content
    """
    data, meta = arff.loadarff(dataset_arff_path)
    return pd.DataFrame(data), meta


def entropy_nominal(samples, meta, target_attribute, attribute=None):
    # type: (pd.DataFrame, arff.arffread.MetaData, str, str) -> float
    """
    Computes the entropy of nominal attributes
    If an attribute name is also passed, computes the weighted average or the conditional entropy of the attribute
    :param samples: DataFrame, the input dataset
    :param meta: MetaData, the metadata of the dataset
    :param target_attribute: String, the class attribute
    :param attribute: String, the attribute for which entropy is to be found. For class entropy, it is None
    :return: float, the entropy of the nominal attribute if passed or the class entropy
    """
    if attribute is None:
        target_att_vals = meta[target_attribute][1]
        target_att_val_counts = samples[target_attribute].value_counts(dropna=False)
        num_samples = len(samples)

        result = 0
        for class_val in target_att_vals:
            num_att_vals = target_att_val_counts.get(class_val, 0)
            if num_att_vals != 0:
                result += - float(num_att_vals) / num_samples * math.log(float(num_att_vals) / num_samples, 2)
        return result
    else:
        result = 0
        for val in meta[attribute][1]:
            # samples with attribute val = val
            samples_attr_val = samples[samples[attribute] == val]
            result += float(len(samples_attr_val)) / len(samples) * entropy_nominal(samples_attr_val, meta,
                                                                                    target_attribute)
        return result


def entropy_numeric(samples, meta, target_attribute, attribute=None):
    """
    Computes the entropy of the numeric attribute.
    The thresholds are chosen at each point where the attribute's value changes and the threshold which
        maximizes the entropy is chosen.
    :param samples: DataFrame, the input dataset
    :param meta: MetaData, the metadata of the dataset
    :param target_attribute: String, the class attribute
    :param attribute: String, a numeric attribute name
    :return: (float, float), tuple with the threshold and the entropy of the numeric attribute
    """
    result = (0, float('inf'))
    att_vals = np.sort(samples[attribute].unique())
    split_points = [(att_vals[i] + att_vals[i + 1]) / 2.0 for i in range(len(att_vals) - 1)]
    for split_point in split_points:
        samples_lte = samples[samples[attribute] <= split_point]
        samples_gt = samples[samples[attribute] > split_point]
        curr_split_entropy = float(len(samples_lte)) / len(samples) * entropy_nominal(samples_lte, meta,
                                                                                      target_attribute) + float(
            len(samples_gt)) / len(samples) * entropy_nominal(samples_gt, meta, target_attribute)
        if curr_split_entropy < result[1]:
            result = (split_point, curr_split_entropy)
    return result


def dt_learn_id3(dataset, m, metadata, features, target_attribute, current_max_class):
    """
    Decision Tree Learning (ID3 like algorithm):
        Recursively choose a feature which maximizes the information gain and split the dataset
        according to it's values.
    :param dataset: DataFrame, the input dataset
    :param m: int, stopping criteria indicating the number of instances to create a leaf node
    :param metadata: MetaData, the metadata of the dataset
    :param features: list, the list of features that can be chosen at a given level for creating a subtree
    :param target_attribute: String, the class attribute
    :param current_max_class: String, the mode of the parent node. Used in stopping criteria while creating leaf nodes
    :return: Dictionary {'feature': ..'value': ..'label'}, the trained decision tree model
    """
    if dataset[target_attribute].nunique() == 1:
        return dataset[target_attribute].unique()[0]
    if len(dataset) < m:
        # Assuming target attribute is binary
        if len(dataset) == 0 or dataset[target_attribute].value_counts()[0] == dataset[target_attribute].value_counts()[1]:
            return current_max_class
        else:
            return dataset[target_attribute].mode()[0]
    if len(features) == 0:
        return current_max_class

    # Start growing the tree
    current_max_class = dataset[target_attribute].mode().get(0, 0)

    nominal_features = [feature for feature in features if metadata[feature][0] not in ['numeric', 'real']]
    numeric_features = [feature for feature in features
                        if metadata[feature][0] in ['numeric', 'real']]

    info_gain_values = [
        (entropy_nominal(dataset, metadata, target_attribute) - entropy_nominal(dataset, metadata, target_attribute, feature))
        for feature in nominal_features]
    info_gain_numeric = [((entropy_nominal(dataset, metadata, target_attribute) -
                           entropy_numeric(dataset, metadata, target_attribute, feature)[1]),
                          entropy_numeric(dataset, metadata, target_attribute, feature)[0])
                         for feature in numeric_features]

    info_gain_max = -1
    index = 0
    for info_gain_val in info_gain_values:
        if info_gain_val > info_gain_max:
            info_gain_max = info_gain_val
            best_feature = nominal_features[index]
        index += 1

    info_gain_numeric_max = -1
    index = 0
    info_gain_split_val = 0
    for info_gain_val in info_gain_numeric:
        if info_gain_val[0] > info_gain_numeric_max:
            info_gain_numeric_max = info_gain_val[0]
            info_gain_split_val = info_gain_val[1]
            best_feature_numeric = numeric_features[index]
        index += 1

    if info_gain_values is not None and len(info_gain_values) > 0 and \
            info_gain_numeric is not None and len(info_gain_numeric) > 0:
        if info_gain_max == info_gain_numeric_max:
            if metadata.names().index(best_feature) > metadata.names().index(best_feature_numeric):
                best_feature = best_feature_numeric
                info_gain_max = info_gain_numeric_max
        elif info_gain_numeric_max > info_gain_max:
            best_feature = best_feature_numeric
            info_gain_max = info_gain_numeric_max
    elif info_gain_values is None or not len(info_gain_values):
        best_feature = best_feature_numeric
        info_gain_max = info_gain_numeric_max

    if info_gain_max == 0:
        return current_max_class

    features = [feature for feature in nominal_features if feature != best_feature]
    features += numeric_features

    tree = {best_feature: {}}

    if metadata[best_feature][0] == 'nominal':
        for value in metadata[best_feature][1]:
            sub_dataset = dataset[dataset[best_feature] == value]
            subtree = dt_learn_id3(sub_dataset, m, metadata, features, target_attribute, current_max_class)
            tree[best_feature][value] = subtree
    elif metadata[best_feature][0] in ['numeric', 'real']:
        split_val = info_gain_split_val

        sub_dataset_lte = dataset[dataset[best_feature] <= split_val]
        subtree_lte = dt_learn_id3(sub_dataset_lte, m, metadata, features, target_attribute, current_max_class)
        tree[best_feature]['<= ' + '{:.6f}'.format(split_val)] = subtree_lte

        sub_dataset_gt = dataset[dataset[best_feature] > split_val]
        subtree_gt = dt_learn_id3(sub_dataset_gt, m, metadata, features, target_attribute, current_max_class)
        tree[best_feature]['> ' + '{:.6f}'.format(split_val)] = subtree_gt

    return tree


def print_tree(root, metadata, dataset, depth=0):
    """
    Prints the learned decision tree in the desired format
    :param root: Dictionary, the learned decision tree
    :param metadata:  MetaData, the metadata of the dataset
    :param dataset: DataFrame, the input dataset
    :param depth: int, a recursive variable used to print tab characters
    :return: void
    """
    if root is None:
        return
    feature = root.items()[0][0]
    if metadata[feature][0] not in ['numeric', 'real']:
        for value in metadata[feature][1]:
            count_str = ' [' + str(dataset[dataset[feature] == value][dataset['class'] == 'negative'].shape[0]) + ' ' + \
                      str(dataset[dataset[feature] == value][dataset['class'] == 'positive'].shape[0]) + ']'
            if isinstance(root[feature][value], dict):
                print ('|' + '\t') * depth + str(feature) + ' = ' + str(value) + count_str
                print_tree(root[feature][value], metadata, dataset[dataset[feature] == value], depth + 1)
            else:
                print ('|' + '\t') * depth + str(feature) + ' = ' + str(value) + count_str + \
                      ': ' + str(root[feature][value])
    else:
        split_val = float(root[feature].items()[0][0].split()[-1])
        for value in ['<= ' + '{:.6f}'.format(split_val), '> ' + '{:.6f}'.format(split_val)]:
            count_str_lte = ' [' + str(dataset[dataset[feature] <= split_val][dataset['class'] == 'negative'].shape[0]) + ' ' + \
                      str(dataset[dataset[feature] <= split_val][dataset['class'] == 'positive'].shape[0]) + ']'
            count_str_gt = ' [' + str(
                        dataset[dataset[feature] > split_val][dataset['class'] == 'negative'].shape[0]) + ' ' + \
                          str(dataset[dataset[feature] > split_val][dataset['class'] == 'positive'].shape[0]) + ']'
            if isinstance(root[feature][value], dict):
                if value.split()[0] == '<=':
                    print ('|' + '\t') * depth + str(feature) + ' ' + str(value) + count_str_lte
                    print_tree(root[feature][value], metadata, dataset[dataset[feature] <= split_val], depth + 1)
                else:
                    print ('|' + '\t') * depth + str(feature) + ' ' + str(value) + count_str_gt
                    print_tree(root[feature][value], metadata, dataset[dataset[feature] > split_val], depth + 1)
            else:
                if value.split()[0] == '<=':
                    print ('|' + '\t') * depth + str(feature) + ' ' + str(value) + count_str_lte + \
                          ': ' + str(root[feature][value])
                else:
                    print ('|' + '\t') * depth + str(feature) + ' ' + str(value) + count_str_gt + \
                          ': ' + str(root[feature][value])


def predict(learned_tree, metadata, testing_query, default=None):
    """
    Predicts the class of a sample test instance using the learned decision tree
    :param learned_tree: Dictionary, the learned decision tree
    :param metadata: MetaData, the metadata of the dataset
    :param testing_query: Series, a sample row for which the class has to be predicted
    :param default: String, this value is returned if a prediction can't be made for some reason
    :return: String, the class label predicted for the testing instance passed as a query
    """
    if learned_tree is None:
        return default

    feature = learned_tree.items()[0][0]
    val = testing_query[feature]

    if metadata[feature][0] not in ['numeric', 'real']:
        if isinstance(learned_tree[feature][val], dict):
            return predict(learned_tree[feature][val], metadata, testing_query, default)
        else:
            return learned_tree[feature][val]
    else:
        split_val = float(learned_tree[feature].items()[0][0].split()[-1])
        if val <= split_val:
            if isinstance(learned_tree[feature]['<= ' + '{:.6f}'.format(split_val)], dict):
                return predict(learned_tree[feature]['<= ' + '{:.6f}'.format(split_val)], metadata, testing_query, default)
            else:
                return learned_tree[feature]['<= ' + '{:.6f}'.format(split_val)]
        else:
            if isinstance(learned_tree[feature]['> ' + '{:.6f}'.format(split_val)], dict):
                return predict(learned_tree[feature]['> ' + '{:.6f}'.format(split_val)], metadata, testing_query, default)
            else:
                return learned_tree[feature]['> ' + '{:.6f}'.format(split_val)]


def dt_test(decision_tree, test_set_path, print_res=True):
    """
    Iterate through all the rows in the test data and make predictions for each instance using the decision tree
    :param decision_tree: Dictionary, the learned decision tree
    :param test_set_path: String, the path of the testing set arff file
    :param print_res: Boolean, determines if the result is to be printed or not
    :return: float, the test set accuracy
    """
    training_set, metadata = import_data(test_set_path)

    predicted_vals = [(predict(decision_tree, metadata, training_sample[1], 'default'), training_sample[1][-1])
                      for training_sample in training_set.iterrows()]
    correct_preds = 0
    total_preds = 1
    for prediction in predicted_vals:
        if print_res:
            print str(total_preds) + ': Actual: ' + prediction[1] + ' Predicted: ' + prediction[0]
        if prediction[0] == prediction[1]:
            correct_preds += 1
        total_preds += 1
    if print_res:
        print 'Number of correctly classified: ' + str(correct_preds) + \
              ' Total number of test instances: ' + str(len(predicted_vals))
    return float(correct_preds) / len(predicted_vals)   # accuracy


def main():
    """
    The driver method which takes in command line arguments, trains the decision tree, prints it, and performs
        predictions on the test data
    :return: void
    """
    if len(sys.argv) != 4:
        print 'Usage: dt-learn <dataset> <trainset> m'
        sys.exit(1)

    training_data_file_path = 'dataset/' + sys.argv[1]
    testing_data_file_path = 'dataset/' + sys.argv[2]
    m = int(sys.argv[3])

    dataset, metadata = import_data(training_data_file_path)
    features = metadata.names()[:-1]
    target_attrib = metadata.names()[-1]

    decision_tree = dt_learn_id3(dataset, m, metadata, features, target_attrib, None)
    print_tree(decision_tree, metadata, dataset)

    print '<Predictions for the Test Set Instances>'
    dt_test(decision_tree, testing_data_file_path)


# Calling the main function
if __name__ == '__main__':
    main()
