import sys
from os.path import dirname
import os

# Sets the PYTHON ENV PATHS
file_abs_path = os.path.abspath(dirname(__file__))
parent = os.path.abspath(os.path.join(file_abs_path, os.pardir))
parent_of_parent = os.path.abspath(os.path.join(parent, os.pardir))
sys.path.extend([parent, parent_of_parent])

from src.neural_network.metrics.Metrics import Metrics
from src.neural_network.preprocessing.Normalizer import Normalizer
from src.neural_network.NeuralNetwork import NeuralNetwork
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.neural_network.preprocessing.KFold import KFold

np.random.seed(42)

# Loads the dataset
data = load_iris(return_X_y=True)
X = data[0]

# One hot encoding and data set separation
y, classes = Metrics.one_hot_encoding(data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Normalization of the data set
normalizer = Normalizer()
x_train_std = normalizer.fit_transform(X_train, -1, 1)
x_test_std = normalizer.fit_transform(X_test, -1, 1)

# Creation of neural network architecture
nn = NeuralNetwork(architecture=[4, 15, 3], activation_function=['tanh', 'tanh'], learning_rate=.1)

# Weights and biases initialization
weights_1 = np.random.random(size=(15, 4))
biases_1 = np.random.random(size=(15, 1))
weights_2 = np.random.random(size=(3, 15))
biases_2 = np.random.random(size=(3, 1))

# Updates the parameters
parameters = [(weights_1, biases_1), (weights_2, biases_2)]
nn.update_weights(parameters)

# Fit and train the neural network
nn.fit(x_train_std, y_train)
nn.train(100, verbose=True)

# Gets the summary results
cm = nn.get_summary().get_confusion_matrix()
Metrics.plot_confusion_matrix(cm, classes)

"""
-------------- BONUS -------------
"""
kfold = KFold()
accuracy = []
rmse = []

# K-Fold validation
for train_index, test_index in kfold.fit_split(X_train, 6):

    nn = NeuralNetwork([4, 15, 3], 'sigmoid', 0.1)
    nn.fit(x_train_std, y_train)
    nn.train(100, verbose=True)

    summ = nn.get_summary()
    accuracy.append(summ.summary['accuracy'][-1])
    rmse.append(summ.summary['rmse'][-1])

print("Mean acc: {:.3f}, std acc: {:.3f}".format(np.mean(accuracy), np.std(accuracy)))
print("Mean loss: {:.3f}, std loss: {:.3f}".format(np.mean(rmse), np.std(rmse)))
