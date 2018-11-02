import sys
import math
import pprint
import collections
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


def laplace_mle_estimate(count_attrib, count_all_attribs, count_total):
    return (float(count_attrib) + 1.0) / (float(count_total) + count_all_attribs)


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
    def __init__(self, dataset, metadata, testset):
        self.dataset = dataset
        self.metadata = metadata
        self.testset = testset

        self.target_attribute = self.metadata.names()[-1]

        n_bayes = NaiveBayes(self.dataset, self.metadata, self.testset)
        n_bayes.train()
        self.parameters = n_bayes.parameters
        self.prior_parameters = n_bayes.prior_parameters
        self.prior_parameters_parent = {}
        self.conditional_parameters = {}

    def train(self):
        print 'Computing info gains for features'
        mutual_info_gains_mat = self.conditional_mutual_info_gain()
        print 'Info gains for features computed'

        # Construct graph for prim input:
        print 'Constructing graph'
        graph = {}
        for i in range(len(mutual_info_gains_mat)):
            graph[i] = {}
            for j in range(len(mutual_info_gains_mat[i])):
                if i != j:
                    graph[i][j] = mutual_info_gains_mat[i][j]
        print 'Graph constructed'

        # pprint.pprint(graph)

        print 'Constructing MST'
        maximal_spanning_tree = self.construct_mst_prims(graph)
        print 'MST Constructed'
        # pprint.pprint(maximal_spanning_tree)

        print 'Computing conditional probabilities with TAN'
        self.compute_tan_conditional_parameters(maximal_spanning_tree)
        # pprint.pprint(self.conditional_parameters)

        self.test(maximal_spanning_tree)

    def test(self, mst):
        print 'Testing TAN:'
        for _, row in self.testset.iterrows():
            predictions = {}
            for tgt_val in self.metadata[self.target_attribute][1]:
                predictions[tgt_val] = self.prior_parameters[tgt_val] * self.parameters[tgt_val][self.metadata.names()[0]][row[self.metadata.names()[0]]]

            for parent in mst:
                parent_attr = self.metadata.names()[parent]
                for child in mst[parent]:
                    child_attr = self.metadata.names()[child]
                    for tgt_val in self.metadata[self.target_attribute][1]:
                        predictions[tgt_val] *= self.conditional_parameters[tgt_val][parent_attr][row[parent_attr]][child_attr][row[child_attr]]

            predicted_target = ''
            max_prob = -1
            predictor_prior_prob = 0.0
            for tgt_val in self.metadata[self.target_attribute][1]:
                if predictions[tgt_val] > max_prob:
                    predicted_target = tgt_val
                    max_prob = predictions[tgt_val]
                predictor_prior_prob += predictions[tgt_val]
            print predicted_target, predictions[predicted_target] / predictor_prior_prob

    def compute_tan_conditional_parameters(self, mst):
        for tgt_val in self.metadata[self.target_attribute][1]:
            dataset_gvn_tgt_val = self.dataset[self.dataset[self.target_attribute] == tgt_val]
            self.conditional_parameters[tgt_val] = {}
            self.prior_parameters_parent[tgt_val] = {}
            for parent in mst:
                parent_attr = self.metadata.names()[parent]
                if parent_attr not in self.conditional_parameters[tgt_val]:
                    self.conditional_parameters[tgt_val][parent_attr] = {}
                for parent_val in self.metadata[parent_attr][1]:
                    dataset_gvn_tgt_parent_vals = dataset_gvn_tgt_val[dataset_gvn_tgt_val[parent_attr] == parent_val]
                    self.conditional_parameters[tgt_val][parent_attr][parent_val] = {}
                    for child in mst[parent]:
                        child_attr = self.metadata.names()[child]
                        if child_attr not in self.conditional_parameters[tgt_val][parent_attr][parent_val]:
                            self.conditional_parameters[tgt_val][parent_attr][parent_val][child_attr] = {}
                        for child_val in self.metadata[child_attr][1]:
                            count_child_val = len(dataset_gvn_tgt_parent_vals[dataset_gvn_tgt_parent_vals[child_attr] == child_val])
                            self.conditional_parameters[tgt_val][parent_attr][parent_val][child_attr][child_val] = laplace_mle_estimate(count_child_val, len(self.metadata[child_attr][1]), len(dataset_gvn_tgt_parent_vals))

    def conditional_mutual_info_gain(self):
        adj_matrix = []
        for attrib_i in self.metadata.names()[:-1]:
            list_xi = []
            for attrib_j in self.metadata.names()[:-1]:
                info_gain = 0.0
                if attrib_i == attrib_j:
                    list_xi.append(-1.0)
                else:
                    for attrib_i_val in self.metadata[attrib_i][1]:
                        for attrib_j_val in self.metadata[attrib_j][1]:
                            count_i_j_y = len(self.dataset[(self.dataset[attrib_i] == attrib_i_val) & (self.dataset[attrib_j] == attrib_j_val) & (self.dataset[self.target_attribute] == self.metadata[self.target_attribute][1][0])])
                            count_i_j_noty = len(self.dataset[(self.dataset[attrib_i] == attrib_i_val) & (self.dataset[attrib_j] == attrib_j_val) & (self.dataset[self.target_attribute] == self.metadata[self.target_attribute][1][1])])

                            count_all_attribs_i_j_y = len(self.metadata[attrib_i][1]) * len(self.metadata[attrib_j][1])
                            prob_i_j_y = laplace_mle_estimate(count_i_j_y, count_all_attribs_i_j_y * 2, len(self.dataset))
                            prob_i_j_given_y = laplace_mle_estimate(count_i_j_y, count_all_attribs_i_j_y, len(self.dataset[self.dataset[self.target_attribute] == self.metadata[self.target_attribute][1][0]]))
                            prob_i_gvn_y_j_gvn_y = self.parameters[self.metadata[self.target_attribute][1][0]][attrib_i][attrib_i_val] * self.parameters[self.metadata[self.target_attribute][1][0]][attrib_j][attrib_j_val]

                            prob_i_j_noty = laplace_mle_estimate(count_i_j_noty, count_all_attribs_i_j_y * 2, len(self.dataset))
                            prob_i_j_given_noty = laplace_mle_estimate(count_i_j_noty, count_all_attribs_i_j_y, len(self.dataset[self.dataset[self.target_attribute] == self.metadata[self.target_attribute][1][1]]))
                            prob_i_gvn_noty_j_gvn_noty = self.parameters[self.metadata[self.target_attribute][1][1]][attrib_i][attrib_i_val] * self.parameters[self.metadata[self.target_attribute][1][1]][attrib_j][attrib_j_val]

                            info_gain += prob_i_j_y * math.log(prob_i_j_given_y / prob_i_gvn_y_j_gvn_y, 2) + prob_i_j_noty * math.log(prob_i_j_given_noty / prob_i_gvn_noty_j_gvn_noty, 2)
                    list_xi.append(info_gain)
            adj_matrix.append(list_xi)
        # print adj_matrix
        return adj_matrix

    def construct_mst_prims(self, graph):
        vertices_new = set([0]) # set([self.metadata.names()[0]])
        # TODO: Do we need ordered dict?
        edges_new = collections.OrderedDict()       # Ordered dict to preserve the order of edges (directions)
        while len(vertices_new) != len(self.metadata.names()) - 1:
            candidate_src = ''
            candidate_vertex = ''
            edge_wt = -1
            for old_vertex in vertices_new:
                for vertex in range(len(self.metadata.names()[:-1])):
                    if vertex not in vertices_new and graph[old_vertex][vertex] > edge_wt:
                        edge_wt = graph[old_vertex][vertex]
                        candidate_src = old_vertex
                        candidate_vertex = vertex
            # add the new vertex to V_new
            vertices_new.add(candidate_vertex)
            # add the new edge to E_new
            if candidate_src not in edges_new:
                edges_new[candidate_src] = collections.OrderedDict()
            edges_new[candidate_src][candidate_vertex] = edge_wt
        return edges_new


def main():
    if len(sys.argv) != 4:
        print 'Usage: bayes <train-set-file> <test-set-file> <n|t>'
        sys.exit(1)

    training_data_file_path = 'dataset/' + sys.argv[1]
    testing_data_file_path = 'dataset/' + sys.argv[2]

    if sys.argv[3] == 'n':
        naive_bayes = True
    elif sys.argv[3] == 't':
        naive_bayes = False
    else:
        print 'Unknown arg at position 3. Please enter n for Naive Bayes or t for TAN'
        sys.exit(1)

    dataset, metadata = import_data(training_data_file_path)
    testset = import_data(testing_data_file_path)[0]

    if naive_bayes:
        bayes = NaiveBayes(dataset, metadata, testset)
        print 'Training:'
        bayes.train()
        print 'Testing:'
        bayes.test()
    else:
        tan = TAN(dataset, metadata, testset)
        tan.train()


if __name__ == '__main__':
    main()
