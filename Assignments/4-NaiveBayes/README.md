# Programming Assignment - 4
## Naive Bayes and Tree-Augmented Naive Bayes (TAN)

### Usage:
The program can be started from the command line by passing in 3 arguments as follows:
```
$ bayes <train-set-file> <test-set-file> <n|t>      # n: Naive Bayes, t: TAN
```
The training set and the testing set data should be passed in [ARFF file format](https://waikato.github.io/weka-wiki/arff_stable/). The data sets are to be placed in the dataset directory, from where the program directly picks them up.

#### Output Format:
The program outputs the following:
* The structure of the Bayes net by listing one line per variable in the following manner 
    * the name of the variable, 
    * the names of its parents in the Bayes net (for naive Bayes, this will simply be the 'class' variable for each other variable) separated by whitespace.
* One line for each instance in the test-set (in the same order as this file) indicating (i) the predicted class, (ii) the actual class, (iii) and the posterior probability of the predicted class (rounded to 12 digits after the decimal point).
* The number of the test-set examples that were correctly classified.