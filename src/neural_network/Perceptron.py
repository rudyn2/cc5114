from src.neural_network.Neuron import Neuron
import numpy as np


__autor__ = "Rudy García Alvarado"


class Perceptron(Neuron):
    """
    This class provides the algorithms and tools to implement a Perceptron.
    """

    def __init__(self, n_input, activation_func, learning_rate):
        super(Perceptron, self).__init__(n_input, activation_func)
        self.learning_rate = learning_rate
        self.x = None
        self.y = None
        self.w = None
        self.b = None
        self.last_error = None

    def fit(self, x_train, y_train):
        """
        It receives the data to train and store it in the neuron. The x_train data must be 2d dimensional and the
        second dimension must be equals to n_input.
        of the neuron.
        :param x_train:                     Numpy array 2D
        :param y_train:                     Numpy array 1D
        """
        assert x_train.shape[1] == self.n_input, "This neuron can't fit the dimension of the input."
        self.x = x_train
        self.y = y_train

    def train(self):
        """
        This method trains a perceptron using a simple algorithm explained in the McCulloch-Pitts model of a
        perceptron. The weights are stored in the W attribute.
        :param:                             If True the learning curve will be plotted.
        """

        # Init a weights vector
        self.w = np.zeros(shape=(self.x.shape[1], 1))

        # Init the bias
        b = (np.random.random()*-1 + np.random.random())*2

        error = []
        # Training using perceptron algorithm
        for i in range(self.x.shape[0]):

            ex = self.x[i, :]
            desired_output = self.y[i]

            # Predicts the output
            predicted_output = np.matmul(ex, self.w) + b

            # Calculate the difference between real and predicted output
            diff = float(desired_output-predicted_output)
            error.append(diff)

            self.w += self.learning_rate*diff*(ex.reshape(self.w.shape[0], 1))
            b += self.learning_rate*diff

        self.b = b
        self.last_error = error

    def predict(self):
        """
        Executes a forward propagation in the network and outputs the result. This was designed to
        perform logical operations between just n operators.
        :return:                    A Numpy Array with the predicted values.
        """

        result = np.matmul(self.x, self.w) + self.b
        return np.array(result)

