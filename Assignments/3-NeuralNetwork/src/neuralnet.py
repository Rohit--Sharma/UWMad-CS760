import sys
import numpy as np
from scipy.io import arff
from scipy.special import expit as sigmoid


class NeuralNetwork:
    """
    A class that holds the parameters of the neural network
    """
    def __init__(self, in_nodes, hidden_nodes, out_nodes, learn_rate, epoch):
        """
        Initialize the class variables
        :param in_nodes: int, number of input nodes of the neural network
        :param hidden_nodes: int, number of hidden nodes of the neural network
        :param out_nodes: int, number of output nodes of the neural network
        :param learn_rate: float, the learning rate of the neural network
        :param epoch: int, the number of epochs to use for training the neural network
        """
        self.num_input_nodes = in_nodes
        self.num_hidden_nodes = hidden_nodes
        self.num_output_nodes = out_nodes
        self.learning_rate = learn_rate
        self.epoch = epoch

        # Initialize the parameters for the edges of the neural network randomly between -1 to 1
        self.weights_input_hidden = np.random.uniform(-1.0, 1.0, (hidden_nodes, in_nodes + 1))  # add 1 for bias
        self.weights_hidden_output = np.random.uniform(-1.0, 1.0, (out_nodes, hidden_nodes + 1))  # add 1 for bias

    def train(self, input_vec, target_vec):
        """
        Forward propagate the input vector using the weights and compute the errors using cross entropy cost function.
        Then, backpropagate the errors to compute the gradient and update the weights to minimize the cost
        :param input_vec: list, the input vector to forward propagate
        :param target_vec: list, the output vector to compare the predictions and compute the errors
        :return: None
        """
        # input_vec and target_vec can be tuple, list or ndarray
        target_vec = np.array(target_vec, ndmin=2).T

        # Compute the activations
        activations = self.forward_propagate(input_vec)

        # Back propagate the errors
        output_errors = target_vec - activations['output_activation']
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)
        hidden_errors = hidden_errors * activations['hidden_activation'] * (1.0 - activations['hidden_activation'])

        # update the weights:
        self.weights_hidden_output += \
            self.learning_rate * np.dot(output_errors, activations['hidden_activation'].T)  # cross entropy cost func
        input_vec = np.array(input_vec, ndmin=2).T
        input_vec = np.concatenate((input_vec, [[1]]))  # Concatenate a row of 1 for bias
        self.weights_input_hidden += \
            self.learning_rate * np.dot(hidden_errors, input_vec.T)[:-1]  # skip the last row as there's no err for bias

    def forward_propagate(self, input_vec):
        """
        Compute the activations for the hidden layer and output layer and returns a dictionary
        :param input_vec: list, the input which has to be forward propagated
        :return: dict: the dictionary which returns the activations as lists for hidden and output layer
        """
        # input_vector can be tuple, list or ndarray

        input_vec = np.array(input_vec, ndmin=2).T
        input_vec = np.concatenate((input_vec, [[1]]))  # Concatenate a row of 1 for bias
        hidden_out_vec = np.dot(self.weights_input_hidden, input_vec)
        hidden_out_vec = sigmoid(hidden_out_vec)
        hidden_out_vec = np.concatenate((hidden_out_vec, [[1]]))

        output_vec = np.dot(self.weights_hidden_output, hidden_out_vec)
        output_vec = sigmoid(output_vec)

        return {'hidden_activation': hidden_out_vec, 'output_activation': output_vec}

    def test_neural_net(self, test_set, indices, predictions):
        """
        Forward propagates each row at an index from indices in the test_set and stores it in a list of predictions
          in the corresponding index.
        This is used to fill in specific predictions for the current fold in which it is called
        :param test_set: list, the list of rows of data for which predictions are to be made
        :param indices: list, the list of indices for which the predictions are to be computed
        :param predictions: list, the list of floats which are predictions to be filled. this value is reused in program
        :return: None
        """
        for _i in range(len(indices)):
            confidence_of_pred = \
                self.forward_propagate(test_set[indices[_i], :-1].astype(float))['output_activation'][0][0]
            predictions[indices[_i]] = confidence_of_pred


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

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=7)

    predicted_confidences = np.zeros(len(data))
    fold_numbers = np.zeros(len(data), dtype=int)
    curr_fold_num = 0
    for train_indexes, test_indexes in skf.split(data[:, :-1].astype(float), data[:, -1]):
        fold_numbers[test_indexes] = curr_fold_num
        curr_fold_num += 1

        X_train, X_test = data[train_indexes, :-1].astype(float), data[test_indexes, :-1].astype(float)
        y_train, y_test = data[train_indexes, -1], data[test_indexes, -1]
        neural_net = NeuralNetwork(len(meta.names()) - 1, len(meta.names()) - 1, 1, learning_rate, num_epochs)
        y_train = np.array(y_train, ndmin=2).T
        for _ in range(neural_net.epoch):
            train_data = np.concatenate((X_train, y_train), axis=1)
            np.random.shuffle(train_data)
            for i in range(len(train_data)):
                neural_net.train(train_data[i, :-1].astype(float),
                                 [0.0] if train_data[i, -1] == meta[meta.names()[-1]][1][0] else [1.0])

        neural_net.test_neural_net(data, test_indexes, predicted_confidences)

    for i in range(len(data)):
        predicted_label = meta[meta.names()[-1]][1][0] if predicted_confidences[i] < 0.5 else meta[meta.names()[-1]][1][
            1]
        actual_label = data[i, -1]
        print fold_numbers[i], predicted_label, actual_label, '{:.6f}'.format(predicted_confidences[i])
