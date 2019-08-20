import unittest
from src.neural_network.NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class NeuralNetworkBasicTest(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNetwork([3, 2, 1], 'sigmoid', 0.01)


class BuildNeuralNetworkTest(NeuralNetworkBasicTest):

    def runTest(self):

        self.assertEqual(self.nn.n_input, 3)
        self.assertEqual(self.nn.neural_network[0].n_neurons, 2)
        self.assertEqual(self.nn.neural_network[1].n_neurons, 1)


class NeuralNetworkAdvancedTest(unittest.TestCase):

    def setUp(self):

        self.nn = NeuralNetwork([4, 3, 3], 'sigmoid', 0.01)

        # Pre processing the data to train
        data = load_iris(return_X_y=True)
        X = data[0]
        parser = {
            0: np.array([1, 0, 0]),
            1: np.array([0, 1, 0]),
            2: np.array([0, 0, 1])
        }
        y = np.array([parser[val] for val in data[1]])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class TrainNeuralNetworkTest(NeuralNetworkAdvancedTest):

    def runTest(self):
        sc = StandardScaler()

        # Compute the mean and standard deviation based on the training data
        sc.fit(self.X_train)

        # Scale the training and test data to be of mean 0 and of unit variance
        x_train_std = sc.transform(self.X_train)
        x_test_std = sc.transform(self.X_test)

        # Fitting the training data
        self.nn.fit(x_train_std, self.y_train)
        self.nn.train(100, verbose=True)





