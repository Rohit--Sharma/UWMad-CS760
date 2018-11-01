import sys
import pprint
import pandas as pd
from scipy.io import arff


def import_data(dataset_arff_path):
    """
    Importing the arff file
    :param dataset_arff_path: String, the path of the arff file to be imported
    :return: DataFrame and Metadata for the file content
    """
    data, meta = arff.loadarff(dataset_arff_path)
    return pd.DataFrame(data), meta


class NaiveBayes:
    def __init__(self, dataset, metadata, testset):
        self.dataset = dataset
        self.metadata = metadata
        self.testset = testset

        self.target_attribute = self.metadata.names()[-1]

        self.parameters = {}    # Initialize the parameters dict
        self.prior_parameters = {}
        self.initialize_params()

    def initialize_params(self):
        for target_val in self.metadata[self.target_attribute][1]:
            self.parameters[target_val] = {}
            for attrib in self.metadata.names():
                if attrib != self.target_attribute and self.metadata[attrib][0] == 'nominal':
                    self.parameters[target_val][attrib] = {}
                    for attrib_val in self.metadata[attrib][1]:
                        self.parameters[target_val][attrib][attrib_val] = 0.0   # initialize all the probabilities to 0
                else:
                    # Handle numeric attributes
                    pass
            self.prior_parameters[target_val] = 0

    def train(self):
        num_samples = len(self.dataset)
        for key in self.prior_parameters:
            count = len(self.dataset[self.dataset[self.target_attribute] == key])
            # MLE Estimate using Laplace estimates
            self.prior_parameters[key] = (float(count) + 1) / (float(num_samples) + len(self.prior_parameters))

        for target_val in self.parameters:
            dataset_given_target_val = self.dataset[self.dataset[self.target_attribute] == target_val]
            num_samples_given_target_val = len(dataset_given_target_val)
            for attrib in self.parameters[target_val]:
                for attrib_val in self.parameters[target_val][attrib]:
                    num_attrib_given_target_val = len(dataset_given_target_val[dataset_given_target_val[attrib] == attrib_val])
                    self.parameters[target_val][attrib][attrib_val] = (float(num_attrib_given_target_val) + 1) / (float(num_samples_given_target_val) + len(self.parameters[target_val][attrib]))
        # pprint.pprint(self.parameters)
        # pprint.pprint(self.prior_parameters)

    def test(self):
        for _, row in self.testset.iterrows():
            predictions = {}
            for tgt_val in self.metadata[self.target_attribute][1]:
                predictions[tgt_val] = self.prior_parameters[tgt_val]
            for attrib in self.metadata.names():
                if attrib != self.target_attribute:
                    for tgt_val in self.metadata[self.target_attribute][1]:
                        predictions[tgt_val] = predictions[tgt_val] * self.parameters[tgt_val][attrib][row[attrib]]

            predicted_target = ''
            max_prob = -1
            predictor_prior_prob = 0.0
            for tgt_val in self.metadata[self.target_attribute][1]:
                if predictions[tgt_val] > max_prob:
                    predicted_target = tgt_val
                    max_prob = predictions[tgt_val]
                predictor_prior_prob += predictions[tgt_val]
            print predicted_target, predictions[predicted_target] / predictor_prior_prob


class TAN:
    def __init__(self, dataset, metadata):
        self.dataset = dataset
        self.metadata = metadata

    def train(self):
        pass


def main():
    if len(sys.argv) != 4:
        print 'Usage: bayes <train-set-file> <test-set-file> <n|t>'
        sys.exit(1)

    training_data_file_path = 'dataset/' + sys.argv[1]
    testing_data_file_path = 'dataset/' + sys.argv[2]

    naive_bayes = True if sys.argv[3] == 'n' else False

    dataset, metadata = import_data(training_data_file_path)
    testset = import_data(testing_data_file_path)[0]

    if naive_bayes:
        bayes = NaiveBayes(dataset, metadata, testset)
        print 'Training:'
        bayes.train()
        print 'Testing:'
        bayes.test()
    else:
        TAN(dataset, metadata)


if __name__ == '__main__':
    main()
