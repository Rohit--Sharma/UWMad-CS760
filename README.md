# CS 760: Machine Learning
## University of Wisconsin-Madison
### Fall 2018 - Programming Assignments

This repository consists of the programming assignments that were implemented as a part of CS760 course. They are written in `python` and targeted to run on UWMadison *Linux* workstations. 

Below are details of individual assignments:

1. Homework 2 - [ID3 Decision Tree](Assignments/2-DecisionTreeID3/README.md): Implementation of an ID3 Decision Tree learner for classification.

	i. To characterize the predictive accuracy of the learned tree as a function of the training set size, *Learning Curves* were plotted and analyzed for two data sets - *heart* and *diabetes*.
	ii. To investigate how predictive accuracy varies as a function of tree size, plots of how test-set accuracy varies with various values of *m* (stopping criteria) were analyzed.

2. Homework 3 - [Neural Network](Assignments/3-NeuralNetwork/README.md): A neural network with one hidden layer was trained with backpropagation to perform binary classification. n-fold *stratified* cross validation was used to train the network. The training was performed using *Stochastic Gradient Descent*, and a *Sigmoid* activation function was used.

	i. To analyze the performance of the neural network (on *sonar* data set), the following graphs were plotted: Accuracy vs Epochs, Accuracy vs Number of folds, and *ROC Curve*

3. Homework 4 - [Bayes Network](Assignments/4-NaiveBayes/README.md): Implementation of *Naive Bayes* and *Tree-Augmented Naive Bayes (TAN)* was done to solve a binary classification problem on discrete valued features. Laplace estimates with pseudocounts of 1 were used to estimate the probabilities. Structure of Bayes network was determined using TAN algorithm.
