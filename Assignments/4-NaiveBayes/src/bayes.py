import sys


def main():
    if len(sys.argv) != 4:
        print 'Usage: bayes <train-set-file> <test-set-file> <n|t>'
        sys.exit(1)

    training_data_file_path = 'dataset/' + sys.argv[1]
    testing_data_file_path = 'dataset/' + sys.argv[2]

    naive_bayes = True if sys.argv[3] == 'n' else False


if __name__ == '__main__':
    main()
