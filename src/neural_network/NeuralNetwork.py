from src.neural_network.NeuronLayer import NeuronLayer
from src.neural_network.exceptions import InvalidArchitectureType
import numpy as np
import time
from src.neural_network.metrics.Summary import Summary
import matplotlib.pyplot as plt
plt.style.use('seaborn')


class NeuralNetwork:
    """
    This class implements the functionality needed to create and train a neural network using backpropagation.
    """

    def __init__(self, architecture: list, activation_function: str or list, learning_rate: float):

        self._check_architecture_list(architecture)
        self.architecture = architecture
        self.activation_functions = self._check_activation_function(activation_function)

        self.n_input = architecture[0]
        self.learning_rate = learning_rate
        self.neural_network = self._build_neural_network(architecture, activation_func=self.activation_functions)
        self.x = None
        self.y = None
        self.x_val = None
        self.y_val = None
        self.summary_train = Summary()
        self.summary_val = Summary()
        self.has_val = False

    @staticmethod
    def _check_architecture_list(architecture: list):
        """
        Checks that a valid architecture was introduced.
        :param architecture:                    A list with architecture parameters.
        """
        # Checks that the architecture list is not empty
        if len(architecture) <= 1:
            raise InvalidArchitectureType.InvalidArchitectureType

        for number in architecture:
            if type(number) != int:
                raise InvalidArchitectureType.InvalidArchitectureType

    def _check_activation_function(self, s: str or list):
        """
        Checks that a valid activation function was introduced.
        :param s:                               Name of activation function or list with activation functions.
        :return:                                Boolean. True if is a valid input.
        """
        if type(s) == str:
            return [s]*(len(self.architecture)-1)

        elif type(s) == list:
            # Checks that a every item is a valid string
            for func in s:
                if type(func) != str:
                    raise ValueError
            if len(s) != len(self.architecture)-1:
                raise InvalidArchitectureType
            return s

    @staticmethod
    def _build_neural_network(architecture: list, activation_func: str or list):
        """
        Builds the neural network using random weights initialization.
        :param architecture:                    The architecture of the neural network.
        :param activation_func:                 The activation function for the entire neural network.
        :return:
        """

        neural_network = []
        for n_neurons_index in range(1, len(architecture)):
            layer = NeuronLayer(architecture[n_neurons_index-1],
                                architecture[n_neurons_index],
                                activation_func[n_neurons_index-1])
            layer.random_init()
            neural_network.append(layer)
        return neural_network

    def _check_data(self, x, y):
        """
        Checks that given data is correct. Otherwise this method will raise AssertionError.
        :param x:                               Input data (can be training, validation or testing data) as a 2D
                                                Numpy array.
        :param y:                               Output data (can be training, validation or testing data) as a 2D or 1D
                                                Numpy array.
        :return:
        """

        # Assert for the training fitting
        assert x.shape[1] == self.n_input, "The number of features of the input must be {}".format(self.n_input)
        assert y.shape[0] == x.shape[0], "The target vector must have length: {}".format(x.shape[0])

        # Target length error
        msg = "The output of the NN must be equals to the target dimension."
        if len(y.shape) == 1:
            assert 1 == self.architecture[-1], msg
        else:
            assert y.shape[1] == self.architecture[-1], msg

    def fit_val(self, x_train, y_train, x_val, y_val):
        """
        Fits the data to the neural network.
        :param x_train:                     A 2D Numpy Array with training data.
        :param y_train:                     A 1D Numpy Array with targets of training data.
        :param x_val:                       A 2D Numpy Array with validation data.
        :param y_val:                       A 1D Numpy Array with targets of validation data.

        """

        # Checks the data
        self._check_data(x_train, y_train)
        self._check_data(x_val, y_val)

        self.has_val = True

        # Stores the data
        self.x = x_train
        self.y = y_train

        self.x_val = x_val
        self.y_val = y_val

    def fit(self, x_train, y_train):
        """
        Fits the data to the neural network.
        :param x_train:                     A 2D Numpy Array with training data.
        :param y_train:                     A 1D Numpy Array with targets of training data.

        """

        # Checks the data
        self._check_data(x_train, y_train)

        # Stores the data
        self.x = x_train
        self.y = y_train

    def forward(self, example):
        """
        It receives an example an executes a forward propagation in the neural network.
        :param example:                     A 1D Numpy Array.
        :return:                            A 1D Numpy Array.
        """

        actual_layer_output = self.neural_network[0].feed(example)
        for index in range(1, len(self.architecture)-1):
            actual_layer_output = self.neural_network[index].feed(actual_layer_output)
        result = actual_layer_output
        return result

    def backwards(self, error: np.ndarray, output_array: np.ndarray) -> np.ndarray:
        """
        Performs the backwards propagation into the neural network.
        :param error:                       The error between prediction and real value
        :param output_array:                The output array of the last forward propagation.
        :return:
        """

        # We define the deltas
        layers_index = list(range(1, len(self.architecture)-1))
        layers_index.reverse()

        layer_deltas = [self._output_delta(error, output_array)]

        # We iterate over the layers from right to left
        for layer_index in layers_index:
            prev_layer = self.neural_network[layer_index-1]
            actual_layer = self.neural_network[layer_index]
            deltas = []

            # We calculate the delta for each neuron of the previous layer using the actual layer
            for prev_neuron_position, neuron in enumerate(prev_layer.layer):

                # We use the actual layer weights and the last calculated deltas
                neuron_error = self._calculate_error(prev_neuron_position, actual_layer, layer_deltas[-1])
                neuron_delta = neuron_error*neuron.activation_func.get_derivative(neuron.last_value)
                deltas.append(neuron_delta)

            layer_deltas.append(np.array(deltas))

        layer_deltas.reverse()
        return np.array(layer_deltas)

    @staticmethod
    def _calculate_error(prev_pos_neuron: int, layer: NeuronLayer, deltas: np.ndarray):
        """
        It calculates the deltas in a layer induced by deltas in the following layer.
        :param prev_pos_neuron:             The position of the previous neuron whose its delta is being calculated.
        :param layer:                       A Layer.
        :return:                            A float value with calculated delta.
        """

        # Iteration over the neurons of the following layer
        total_adjust = 0
        for pos, neuron in enumerate(layer.layer):
            total_adjust += neuron.w[prev_pos_neuron]*deltas[pos]
        return total_adjust

    def _output_delta(self, error: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        This method calculates the delta of the output layer using the calculated error and neuron outputs.
        :param error:
        :param output:
        :return:
        """

        assert len(error) == len(output), "The error and output are not length equals. Something is wrong."

        output_delta = []
        for index, neuron in enumerate(self.neural_network[-1].layer):
            output_delta.append(error[index]*neuron.activation_func.get_derivative(output[index]))
        return np.array(output_delta)

    def bp_update(self, feed: np.ndarray, deltas: np.ndarray):
        """
        Updates the weights of the neural network using the calculated delta errors.
        :param feed:                        Input of the neural network.
        :param deltas:                      Last delta errors calculated.
        """

        for pos, layer in enumerate(self.neural_network):
            actual_deltas = deltas[pos]
            for delta_pos, neuron in enumerate(layer.layer):

                # We compute and do the update
                neuron_delta = actual_deltas[delta_pos]
                weight_update = self.learning_rate*neuron_delta*feed
                bias_update = self.learning_rate*neuron_delta
                neuron.update(neuron.w + weight_update, neuron.bias + bias_update)
            feed = layer.last_values

    def train(self, epochs: int, verbose=False):
        """
        It trains the neural network a total of "epochs" times. The verbose parameter allows to see some important
        information in the console.
        :param epochs:                              An int greater than 1.
        :param verbose:                             A Boolean.
        """

        assert self.x is not None and self.y is not None, "The data has not been fitted."
        assert type(epochs) == int and epochs >= 1, "The epochs is not a valid number."

        evolution_train_msg = "EPOCH: {} | MSE: {:.4f} | Train acc: {:.4f} | TIME: {:.2f} ms"
        evolution_val_msg = "EPOCH: {} | MSE: {:.4f} | Train acc: {:.4f} | Val acc {:.4f}| TIME: {:.2f} ms"
        total_time = time.time()
        for n_epoch in range(epochs):

            start_time = time.time()

            # a epoch in the entire data set
            for example_index in range(self.x.shape[0]):
                example = self.x[example_index, :]

                # Step 1 - Forward
                output = self.forward(example)
                error = self.y[example_index]-output

                # Step 2 - Backwards
                deltas = self.backwards(error, output)

                # Step 3 - Updates
                self.bp_update(example, deltas)

            self.summary_train.fit_and_add_step(self.y, self.predict(self.x))

            if self.has_val:
                self.summary_val.fit_and_add_step(self.y_val, self.predict(self.x_val))

            if verbose:
                if self.has_val:
                    print(evolution_val_msg.format(n_epoch+1,
                                                   self.summary_train.get_mse(),
                                                   self.summary_train.get_accuracy(),
                                                   self.summary_val.get_accuracy(),
                                                   (time.time()-start_time)/10**-3))
                else:
                    print(evolution_train_msg.format(n_epoch+1,
                                                     self.summary_train.get_mse(),
                                                     self.summary_train.get_accuracy(),
                                                     (time.time()-start_time)/10**-3))

        if verbose:
            print("Total Time: {:.2f} s".format((time.time()-total_time)))

    def get_summary_train(self) -> Summary:
        """
        Gets the training summary content of the last training procedure.
        :return:                        A numpy array.
        """
        return self.summary_train

    def get_summary_val(self) -> Summary:
        """
        Gets the validation summary content of the last training procedure.
        :return:                        A numpy array.
        """
        return self.summary_val

    def predict(self, batch) -> np.ndarray:
        """
        This method provides the functionality to predict the values per batch using the neural network.
        :return:                                A numpy array with the predicted values.
        """

        predicted_values = []

        # Make a prediction for every sample
        for example_index in range(batch.shape[0]):
            predicted_values.append(self.forward(batch[example_index, :]))

        return np.array(predicted_values)

    def update_weights(self, list_parameters: list):
        """
        It updates the neural network with a list of parameters. Each item of the list must be a tuple. Each
        tuple must have 2 Numpy arrays. The first one must contain the weights and the second one the biases. Each
        row of the weights must be a n-dimensional vector with the weights for each previous layer connection (e.g.
        for the first hidden layer each row has same length than number of features). The biases must be a 1-D
        numpy array with same length that number of neurons in the layer. The layers index are positional with the list
        of parameters (e.g. the first item of the list parameters corresponds to the weights and biases of the first
        hidden layer and so on).
        :param list_parameters:                 List with parameters.
        :return:
        """
        assert len(list_parameters) == len(self.architecture)-1

        for pos, item in enumerate(list_parameters):
            weights = item[0]
            biases = item[1]
            self.neural_network[pos].update(weights, biases)

