from scipy.io import arff
import numpy as np
import pandas as pd
import math
from pprint import pprint


# Importing dataset
def import_data(dataset_arff):
    data, meta = arff.loadarff('../' + dataset_arff)
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
        curr_split_entropy = float(len(samples_lte)) / len(samples) * entropy_pds(samples_lte, meta, target_attribute) + float(len(samples_gt)) / len(samples) * entropy_pds(samples_gt, meta, target_attribute)
        if curr_split_entropy < result[1]:
            result = (split_point, curr_split_entropy)
    return result


def dt_learn_id3(dataset, metadata, features, target_attribute, current_max_class):
    # Stopping criteria:
    if dataset[target_attribute].nunique() == 1:
        # print dataset[target_attribute].unique()
        return dataset[target_attribute].unique()[0]
    if len(dataset) == 0:
        # print 'Reached empty dataset'
        return current_max_class
    if len(features) == 0:
        return current_max_class

    # Start growing the tree
    current_max_class = dataset[target_attribute].mode().get(0, 0)

    info_gain_values = [
        (entropy_pds(dataset, metadata, target_attribute) - entropy_pds(dataset, metadata, target_attribute, feature))
        for feature in features if metadata[feature][0] != 'numeric']
    best_feature = features[np.argmax(info_gain_values)]
    features = [feature for feature in features if feature != best_feature]

    tree = {best_feature: {}}

    for value in metadata[best_feature][1]:  # dataset[best_feature].unique():
        sub_dataset = dataset[dataset[best_feature] == value]
        subtree = dt_learn_id3(sub_dataset, metadata, features, target_attribute, current_max_class)
        tree[best_feature][value] = subtree

    return tree


# Driver Code
def main():
    dataset, metadata = import_data('heart_train.arff')
    features = [feature for feature in metadata.names()[:-1] if metadata[feature][0] != 'numeric']
    target_attrib = metadata.names()[-1]

    # print(entropy_num(dataset, metadata, target_attrib, 'age'))

    decision_tree = dt_learn_id3(dataset, metadata, features, target_attrib, None)
    pprint(decision_tree)


# Calling the main function
if __name__ == '__main__':
    main()
