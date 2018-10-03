# Programming Assignment - 2
## ID3-Like Decision Tree Learner

### Usage:
The program can be started from the command line by passing in 3 arguments as follows:
```
$ dt-learn <train-set-file> <test-set-file> m
```
The training set and the testing set data should be passed in [ARFF file format](https://waikato.github.io/weka-wiki/arff_stable/). The data sets are to be placed in the dataset directory, from where the program directly picks them up.
The argument *m* indicates the number of training instances reaching the node for which leaf nodes are created.
