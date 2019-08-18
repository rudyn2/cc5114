from src.neural_network.NeuronLayer import NeuronLayer
from src.neural_network.exceptions import InvalidArchitectureType


class NeuralNetwork:

    def __init__(self, architecture: list, activation_function: str):

        self._check_architecture_list(architecture)
        self.architecture = architecture
        self.n_input = architecture[0]
        self.neural_network = self._build_neural_network(architecture, activation_func=activation_function)
        self.x = None
        self.y = None

    @staticmethod
    def _check_architecture_list(architecture):

        if len(architecture) <= 1:
            raise InvalidArchitectureType

        for number in architecture:
            if type(number) != int:
                raise InvalidArchitectureType

    @staticmethod
    def _build_neural_network(architecture: list, activation_func: str):

        neural_network = []
        n_input = architecture[0]
        for n_neurons_index in range(1, len(architecture)):
            neural_network.append(NeuronLayer(n_input, architecture[n_neurons_index], activation_func=activation_func))
        return neural_network

    def fit(self, x_train, y_train):
        """
        Fits the data to the neural network.
        :param x_train:                     A 2D Numpy Array.
        :param y_train:                     A 1D Numpy Array with targets.
        """

        assert x_train.shape[1] == self.n_input, "The number of features of the input must be {}".format(self.n_input)
        assert y_train.shape[0] == x_train.shape[0], "The target vector must have length: {}".format(x_train.shape[0])

        self.x = x_train
        self.y = y_train










