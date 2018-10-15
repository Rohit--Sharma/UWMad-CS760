import sys
import numpy as np
from scipy.io import arff
from scipy.special import expit as sigmoid
from sklearn.model_selection import StratifiedKFold


class NeuralNetwork:
    def __init__(self, in_nodes, hidden_nodes, out_nodes, learn_rate, epoch):
        self.num_input_nodes = in_nodes
        self.num_hidden_nodes = hidden_nodes
        self.num_output_nodes = out_nodes
        self.learning_rate = learn_rate
        self.epoch = epoch

        self.weights_input_hidden = np.random.uniform(-1.0, 1.0, (hidden_nodes, in_nodes + 1))  # add 1 for bias
        self.weights_hidden_output = np.random.uniform(-1.0, 1.0, (out_nodes, hidden_nodes + 1))  # add 1 for bias

    def train(self, input_vec, target_vec):
        # input_vec and target_vec can be tuple, list or ndarray
        input_vec = np.array(input_vec, ndmin=2).T
        input_vec = np.concatenate((input_vec, [[1]]))      # Concatenate a row of 1 for bias
        target_vec = np.array(target_vec, ndmin=2).T

        output_vector1 = np.dot(self.weights_input_hidden, input_vec)
        output_vector_hidden = sigmoid(output_vector1)
        output_vector_hidden = np.concatenate((output_vector_hidden, [[1]]))      # Concatenate a row of 1 for bias

        output_vector2 = np.dot(self.weights_hidden_output, output_vector_hidden)
        output_vector_network = sigmoid(output_vector2)

        output_errors = target_vec - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_output += tmp
        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        self.weights_input_hidden += self.learning_rate * np.dot(tmp, input_vec.T)[:-1, :]  # ???? last element cut off, ???

    def forward_propagate(self, input_vec):
        # input_vector can be tuple, list or ndarray

        input_vec = np.array(input_vec, ndmin=2).T
        input_vec = np.concatenate((input_vec, [[1]]))  # Concatenate a row of 1 for bias
        output_vector = np.dot(self.weights_input_hidden, input_vec)
        output_vector = sigmoid(output_vector)
        output_vector = np.concatenate((output_vector, [[1]]))

        output_vector = np.dot(self.weights_hidden_output, output_vector)
        output_vector = sigmoid(output_vector)

        return output_vector

    @staticmethod
    def test_neural_net(test_set, indices, predictions):
        correct_pred = 0
        for i in range(len(test_set)):
            # actual_label = 0 if data[i, -1] == 'Rock' else 1
            # predicted_label = 0 if neural_net.forward_propagate(data[i, :-1].astype(float))[0][0] < 0.5 else 1
            # print 'Actual:', actual_label, 'Predicted:', predicted_label
            actual_label = test_set[i, -1]
            confidence_of_pred = neural_net.forward_propagate(test_set[i, :-1].astype(float))[0][0]
            predicted_label = meta[meta.names()[-1]][1][0] if confidence_of_pred < 0.5 else meta[meta.names()[-1]][1][1]
            # print i, predicted_label, actual_label, confidence_of_pred

            if actual_label == predicted_label:
                correct_pred += 1

        return correct_pred * 1.0 / len(test_set)


if __name__ == '__main__':
    """
    The driver method which takes in command line arguments, trains the neural network using backpropagation
    algorithm, and predicts the output for each instance with some confidence
    :return: void
    """
    if len(sys.argv) != 5:
        print 'Usage: neuralnet trainfile num_folds learning_rate num_epochs'
        sys.exit(1)

    training_data_file_path = 'dataset/' + sys.argv[1]
    num_folds = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    num_epochs = int(sys.argv[4])

    np.random.seed(0)
    data, meta = arff.loadarff(training_data_file_path)
    data = np.asarray(data.tolist())
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=37)
    # np.random.shuffle(data)
    # train_data_size = int(len(data) * 0.9)
    # train_data = data[:train_data_size]
    # test_data = data[train_data_size:]

    # neural_net = NeuralNetwork(len(meta.names()) - 1, len(meta.names()) - 1, 1, learning_rate, num_epochs)
    # for _ in range(neural_net.epoch):
    # for i in range(len(train_data)):
    #     neural_net.train(train_data[i, :-1].astype(float), [0.0] if train_data[i, -1] == 'Rock' else [1.0])

    predicted_confidences = np.zeros(len(data))
    for train_indexes, test_indexes in skf.split(data[:, :-1].astype(float), data[:, -1]):
        X_train, X_test = data[train_indexes, :-1].astype(float), data[test_indexes, :-1].astype(float)
        y_train, y_test = data[train_indexes, -1], data[test_indexes, -1]
        neural_net = NeuralNetwork(len(meta.names()) - 1, len(meta.names()) - 1, 1, learning_rate, num_epochs)
        for _ in range(neural_net.epoch):
            for i in range(len(X_train)):
                neural_net.train(X_train[i], [0.0] if y_train[i] == 'Rock' else [1.0])
        fold_accuracy = neural_net.test_neural_net(data[test_indexes], test_indexes, predicted_confidences)
        print 'Fold accuracy:', fold_accuracy

    """
    train_accuracy = neural_net.test_neural_net(train_data)
    test_accuracy = neural_net.test_neural_net(test_data)
    print 'Train accuracy:', train_accuracy
    print 'Test accuracy:', test_accuracy
    """
