from scipy.io import arff
import numpy as np
import pandas as pd
import sys
import math


def import_data(dataset_arff_path):
    """
    Importing the arff file
    :param dataset_arff_path: String, the path of the arff file to be imported
    :return: DataFrame and Metadata for the file content
    """
    data, meta = arff.loadarff(dataset_arff_path)
    return pd.DataFrame(data), meta


def sigmoid(input):
    return 1.0 / (1.0 + math.exp(-input))


def neuralnet_learn(dataset, input_layer_dim, hidden_layer_dim, output_layer_dim, num_folds, learning_rate, num_epochs):
    # np.random.seed(100)
    num_instances = dataset.shape[0]

    weights_layer_1 = np.random.uniform(-1.0, 1.0, (input_layer_dim, hidden_layer_dim))
    bias_layer_1 = np.random.uniform(-1.0, 1.0, (num_instances, hidden_layer_dim))
    weights_layer_2 = np.random.uniform(-1.0, 1.0, (hidden_layer_dim, output_layer_dim))
    bias_layer_2 = np.random.uniform(-1.0, 1.0, (num_instances, output_layer_dim))

    z_layer_1 = dataset.iloc[:, :-1].dot(weights_layer_1) + bias_layer_1
    activate_layer_1 = z_layer_1.applymap(lambda val: sigmoid(val))

    z_layer_2 = activate_layer_1.dot(weights_layer_2) + bias_layer_2
    activate_layer_2 = z_layer_2.applymap(lambda val: sigmoid(val))

    print activate_layer_2


def main():
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

    dataset, metadata = import_data(training_data_file_path)

    input_layer_dim = dataset.shape[1] - 1      # Excluded 1 for class attribute
    hidden_layer_dim = input_layer_dim
    output_layer_dim = 1

    neuralnet_learn(dataset, input_layer_dim, hidden_layer_dim, output_layer_dim, num_folds, learning_rate, num_epochs)


# Calling the main function
if __name__ == '__main__':
    main()
