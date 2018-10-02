from scipy.io import arff
import numpy as np
import pandas as pd
import math
from pprint import pprint
import sys


# Importing dataset
def import_data(dataset_arff):
    data, meta = arff.loadarff('dataset/' + dataset_arff)
    return pd.DataFrame(data), meta


def entropy_pds(samples, meta, target_attribute, attribute=None):
    # type: (pd.DataFrame, arff.arffread.MetaData, str, str) -> float
    """
    Computes the entropy of nominal attributes
    If an attribute name is also passed, computes the weighted average or the conditional entropy of the attribute
    :param samples: DataFrame, the input dataset
    :param meta:
    :param target_attribute:
    :param attribute:
    :return:
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
            result += float(len(samples_attr_val)) / len(samples) * entropy_pds(samples_attr_val, meta,
                                                                                target_attribute)
        return result


def entropy_num(samples, meta, target_attribute, attribute=None):
    """

    :param samples:
    :param meta:
    :param target_attribute:
    :param attribute: String, a numeric attribute name
    :return:
    """
    result = (0, float('inf'))
    att_vals = np.sort(samples[attribute].unique())
    split_points = [(att_vals[i] + att_vals[i + 1]) / 2.0 for i in range(len(att_vals) - 1)]
    for split_point in split_points:
        samples_lte = samples[samples[attribute] <= split_point]
        samples_gt = samples[samples[attribute] > split_point]
        curr_split_entropy = float(len(samples_lte)) / len(samples) * entropy_pds(samples_lte, meta,
                                                                                  target_attribute) + float(
            len(samples_gt)) / len(samples) * entropy_pds(samples_gt, meta, target_attribute)
        if curr_split_entropy < result[1]:
            result = (split_point, curr_split_entropy)
    return result


def dt_learn_id3(dataset, m, metadata, features, target_attribute, current_max_class):
    # Stopping criteria:
    if dataset[target_attribute].nunique() == 1:
        # print dataset[target_attribute].unique()
        return dataset[target_attribute].unique()[0]
    if len(dataset) < m or len(dataset) == 0:
        # print 'Reached empty dataset'
        return current_max_class
    if len(features) == 0:
        return current_max_class

    # Start growing the tree
    current_max_class = dataset[target_attribute].mode().get(0, 0)

    nominal_features = [feature for feature in features if metadata[feature][0] != 'numeric']
    numeric_features = [feature for feature in features
                        if metadata[feature][0] == 'numeric' or metadata[feature][0] == 'real']

    info_gain_values = [
        (entropy_pds(dataset, metadata, target_attribute) - entropy_pds(dataset, metadata, target_attribute, feature))
        for feature in nominal_features]
    info_gain_numeric = [((entropy_pds(dataset, metadata, target_attribute) -
                           entropy_num(dataset, metadata, target_attribute, feature)[1]),
                          entropy_num(dataset, metadata, target_attribute, feature)[0])
                         for feature in numeric_features]
    # print info_gain_numeric

    # if info_gain_values is not None and len(info_gain_values) > 0:
    #     best_feature = nominal_features[np.argmax(info_gain_values)]
    info_gain_max = -1
    index = 0
    for info_gain_val in info_gain_values:
        if info_gain_val > info_gain_max:
            info_gain_max = info_gain_val
            best_feature = nominal_features[index]
        index += 1

    # if info_gain_numeric is not None and len(info_gain_numeric) > 0:
    #     best_feature_numeric = numeric_features[np.argmax([info_gain[0] for info_gain in info_gain_numeric])]
    info_gain_numeric_max = -1
    index = 0
    info_gain_split_val = 0
    for info_gain_val in info_gain_numeric:
        if info_gain_val[0] > info_gain_numeric_max:
            info_gain_numeric_max = info_gain_val[0]
            info_gain_split_val = info_gain_val[1]
            best_feature_numeric = numeric_features[index]
        index += 1
    # print best_feature_numeric, info_gain_numeric[np.argmax([info_gain[0] for info_gain in info_gain_numeric])]
    # print best_feature, info_gain_values[np.argmax(info_gain_values)]

    if info_gain_values is not None and len(info_gain_values) > 0 and \
            info_gain_numeric is not None and len(info_gain_numeric) > 0:
        if info_gain_max == info_gain_numeric_max:
            if metadata.names().index(best_feature) > metadata.names().index(best_feature_numeric):
                best_feature = best_feature_numeric
                info_gain_max = info_gain_numeric_max
        elif info_gain_numeric_max > info_gain_max:
            best_feature = best_feature_numeric
            info_gain_max = info_gain_numeric_max
        '''
        best_feature = best_feature if info_gain_values[np.argmax(info_gain_values)] >= \
                                       info_gain_numeric[np.argmax([info_gain[0] for info_gain in info_gain_numeric])][
                                           0] else best_feature_numeric
        '''
    elif info_gain_values is None or not len(info_gain_values):
        best_feature = best_feature_numeric
        info_gain_max = info_gain_numeric_max

    # print best_feature_overall, metadata[best_feature_overall][0]
    # print best_feature_numeric, info_gain_numeric[np.argmax([info_gain[0] for info_gain in info_gain_numeric])][1]

    features = [feature for feature in nominal_features if feature != best_feature]
    features += numeric_features
    # print features

    tree = {best_feature: {}}

    if metadata[best_feature][0] == 'nominal':
        for value in metadata[best_feature][1]:  # dataset[best_feature].unique():
            sub_dataset = dataset[dataset[best_feature] == value]
            subtree = dt_learn_id3(sub_dataset, m, metadata, features, target_attribute, current_max_class)
            tree[best_feature][value] = subtree
    elif metadata[best_feature][0] in ['numeric', 'real']:
        split_val = info_gain_split_val  # info_gain_numeric[np.argmax([info_gain[0] for info_gain in info_gain_numeric])][1]

        sub_dataset_lte = dataset[dataset[best_feature] <= split_val]
        subtree_lte = dt_learn_id3(sub_dataset_lte, m, metadata, features, target_attribute, current_max_class)
        tree[best_feature]['<= ' + '{:.6f}'.format(split_val)] = subtree_lte

        sub_dataset_gt = dataset[dataset[best_feature] > split_val]
        subtree_gt = dt_learn_id3(sub_dataset_gt, m, metadata, features, target_attribute, current_max_class)
        tree[best_feature]['> ' + '{:.6f}'.format(split_val)] = subtree_gt

    return tree


def print_tree(root, metadata, dataset, depth=0):
    if root is None:
        return
    feature = root.items()[0][0]
    if metadata[feature][0] not in ['numeric', 'real']:
        for value in metadata[feature][1]:
            dataset = dataset[dataset[feature] == value]
            count_str = ' [' + str(dataset[dataset['class'] == 'negative'].shape[0]) + ' ' + \
                      str(dataset[dataset['class'] == 'positive'].shape[0]) + ']'
            if isinstance(root[feature][value], dict):
                print ('|' + '\t') * depth + str(feature) + ' = ' + str(value) + count_str
                print_tree(root[feature][value], metadata, dataset, depth + 1)
            else:
                print ('|' + '\t') * depth + str(feature) + ' = ' + str(value) + count_str + \
                      ': ' + str(root[feature][value])
    else:
        split_val = float(root[feature].items()[0][0].split()[-1])
        dataset_lte = dataset[dataset[feature] <= split_val]
        dataset_gt = dataset[dataset[feature] > split_val]
        for value in ['<= ' + '{:.6f}'.format(split_val), '> ' + '{:.6f}'.format(split_val)]:
            count_str_lte = ' [' + str(dataset_lte[dataset['class'] == 'negative'].shape[0]) + ' ' + \
                      str(dataset_lte[dataset_lte['class'] == 'positive'].shape[0]) + ']'
            count_str_gt = ' [' + str(
                dataset_gt[dataset['class'] == 'negative'].shape[0]) + ' ' + \
                          str(dataset_gt[dataset_gt['class'] == 'positive'].shape[0]) + ']'
            if isinstance(root[feature][value], dict):
                if value.split()[0] == '<=':
                    print ('|' + '\t') * depth + str(feature) + ' ' + str(value) + count_str_lte
                    print_tree(root[feature][value], metadata, dataset_lte, depth + 1)
                else:
                    print ('|' + '\t') * depth + str(feature) + ' ' + str(value) + count_str_gt
                    print_tree(root[feature][value], metadata, dataset_gt, depth + 1)
            else:
                if value.split()[0] == '<=':
                    print ('|' + '\t') * depth + str(feature) + ' ' + str(value) + count_str_lte + \
                          ': ' + str(root[feature][value])
                else:
                    print ('|' + '\t') * depth + str(feature) + ' ' + str(value) + count_str_gt + \
                          ': ' + str(root[feature][value])


def predict(learned_tree, metadata, testing_query, default=None):
    # print testing_query
    if learned_tree is None:
        return default

    feature = learned_tree.items()[0][0]
    # print feature
    val = testing_query[feature]
    # print val

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


def dt_test(decision_tree, metadata):
    training_set, metadata = import_data(sys.argv[2])

    predicted_vals = [(predict(decision_tree, metadata, training_sample[1], 'default'), training_sample[1][-1])
                      for training_sample in training_set.iterrows()]
    correct_preds = 0
    total_preds = 1
    for prediction in predicted_vals:
        print str(total_preds) + ': Actual: ' + prediction[1] + ' Predicted: ' + prediction[0]
        if prediction[0] == prediction[1]:
            correct_preds += 1
        total_preds += 1
    print 'Number of correctly classified: ' + str(correct_preds) + \
          ' Total number of test instances: ' + str(len(predicted_vals))
    # print 'Accuracy: ' + str(float(correct_preds) / len(predicted_vals))


# Driver Code
def main():
    dataset, metadata = import_data(sys.argv[1])
    # features = [feature for feature in metadata.names()[:-1] if metadata[feature][0] != 'numeric']
    features = metadata.names()[:-1]
    target_attrib = metadata.names()[-1]

    # print(entropy_num(dataset, metadata, target_attrib, 'age'))

    decision_tree = dt_learn_id3(dataset, int(sys.argv[3]), metadata, features, target_attrib, None)
    # pprint(decision_tree)
    print_tree(decision_tree, metadata, dataset)

    # print 'Predicted value: ' +
    print '<Predictions for the Test Set Instances>'
    dt_test(decision_tree, metadata)


# Calling the main function
if __name__ == '__main__':
    main()
