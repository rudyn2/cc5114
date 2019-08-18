from src.neural_network.Neuron import Neuron

__autor__ = "Rudy García Alvarado"


class Perceptron(Neuron):
    """
    This class provides the algorithms and tools to implement a Perceptron.
    """

    def __init__(self, n_input, activation_func):
        super(Perceptron, self).__init__(n_input, activation_func)
        self.x = None
        self.y = None

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
        This train the neuron as a perceptron using a simple learning algorithm.
        :return:
        """
        pass
