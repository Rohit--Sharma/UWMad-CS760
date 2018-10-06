from scipy.io import arff
import numpy as np
import pandas as pd
import sys


def import_data(dataset_arff_path):
    """
    Importing the arff file
    :param dataset_arff_path: String, the path of the arff file to be imported
    :return: DataFrame and Metadata for the file content
    """
    data, meta = arff.loadarff(dataset_arff_path)
    return pd.DataFrame(data), meta


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


# Calling the main function
if __name__ == '__main__':
    main()
