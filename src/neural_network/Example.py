import sys
from os.path import dirname
import os
import matplotlib.pyplot as plt

# Sets the PYTHON ENV PATHS
file_abs_path = os.path.abspath(dirname(__file__))
parent = os.path.abspath(os.path.join(file_abs_path, os.pardir))
parent_of_parent = os.path.abspath(os.path.join(parent, os.pardir))
sys.path.extend([parent, parent_of_parent])

from src.neural_network.metrics.Metrics import Metrics
from src.neural_network.preprocessing.Normalizer import Normalizer
from src.neural_network.metrics.Summary import Summary
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
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.7)
x_val, x_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5)

# Normalization of the data set
normalizer = Normalizer()
x_train_std = normalizer.fit_transform(X_train, -1, 1)
x_val_std = normalizer.fit_transform(x_val, -1, 1)
x_test_std = normalizer.fit_transform(x_test, -1, 1)

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
nn.fit_val(x_train_std, y_train, x_val_std, y_val)
nn.train(100, verbose=True)

# Testing the neural network
y_test_predicted = nn.predict(x_test_std)
test_summ = Summary()
test_summ.fit_and_add_step(y_test, y_test_predicted)
test_acc = test_summ.get_accuracy()

# Gets the summary results
summ = nn.get_summary_train()
cm = summ.get_confusion_matrix()
Metrics.plot_confusion_matrix(cm, classes)

# Gets the accuracy and loss
accuracy = summ.summary['accuracy']
mse = summ.summary['mse']

print("\nResults:\n")
print("Training accuracy: {:.3f}".format(accuracy[-1]))
print("Validation accuracy: {:.3f}".format(nn.get_summary_val().summary['accuracy'][-1]))
print("Testing accuracy: {:.3f}".format(test_acc))

fig, axs = plt.subplots(2, 1, figsize=(10, 5))
axs[0].plot(accuracy, label='Training accuracy')
axs[0].plot(nn.get_summary_val().summary['accuracy'], label='Validation accuracy')
axs[0].set(title='Accuracy vs iteration', ylabel='Accuracy', xlabel='Iteration')
axs[0].legend()

axs[1].plot(mse)
axs[1].set(title='MSE vs iteration', ylabel='MSE', xlabel='Iteration')

plt.show()

# """
# -------------- BONUS -------------
# """
# kfold = KFold()
# accuracy = []
# rmse = []
#
# # K-Fold validation
# for train_index, test_index in kfold.fit_split(X_train, 6):
#
#     nn = NeuralNetwork([4, 15, 3], 'sigmoid', 0.1)
#     nn.fit(x_train_std, y_train)
#     nn.train(100, verbose=True)
#
#     summ = nn.get_summary()
#     accuracy.append(summ.summary['accuracy'][-1])
#     rmse.append(summ.summary['rmse'][-1])
#
# print("Mean acc: {:.3f}, std acc: {:.3f}".format(np.mean(accuracy), np.std(accuracy)))
# print("Mean loss: {:.3f}, std loss: {:.3f}".format(np.mean(rmse), np.std(rmse)))
