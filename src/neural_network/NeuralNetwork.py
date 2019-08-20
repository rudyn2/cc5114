from src.neural_network.NeuronLayer import NeuronLayer
from src.neural_network.exceptions import InvalidArchitectureType
from src.neural_network.metrics.Metrics import Metrics
import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use('seaborn')


class NeuralNetwork:

    def __init__(self, architecture: list, activation_function: str, learning_rate: float, threshold = 0.5):

        self._check_architecture_list(architecture)
        self.architecture = architecture
        self.n_input = architecture[0]
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.neural_network = self._build_neural_network(architecture, activation_func=activation_function)
        self.x = None
        self.y = None
        self.summary = None

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
        for n_neurons_index in range(1, len(architecture)):
            layer = NeuronLayer(architecture[n_neurons_index-1], architecture[n_neurons_index], activation_func)
            layer.random_init()
            neural_network.append(layer)
        return neural_network

    def fit(self, x_train, y_train):
        """
        Fits the data to the neural network.
        :param x_train:                     A 2D Numpy Array.
        :param y_train:                     A 1D Numpy Array with targets.
        """

        assert x_train.shape[1] == self.n_input, "The number of features of the input must be {}".format(self.n_input)
        assert y_train.shape[0] == x_train.shape[0], "The target vector must have length: {}".format(x_train.shape[0])

        # Target length error
        msg = "The output of the NN must be equals to the target dimension."
        if len(y_train.shape) == 1:
            assert 1 == self.architecture[-1], msg
        else:
            assert y_train.shape[1] == self.architecture[-1], msg

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

    def backwards(self, error, input_array, output_array):

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

            error = np.array(deltas)
            layer_deltas.append(error)

        layer_deltas.reverse()
        return layer_deltas


    @staticmethod
    def _calculate_error(prev_pos_neuron, layer, deltas):
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

    def _output_delta(self, error, output):
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

    def bp_update(self, feed, deltas):
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

    def train(self, epochs, verbose=False):
        """
        It trains the neural network a total of "epochs" times. The verbose parameter allows to see some important
        information in the console.
        :param epochs:                              An int greater than 1.
        :param verbose:                             A Boolean.
        """

        assert self.x is not None and self.y is not None, "The data has not been fitted."
        assert type(epochs) == int and epochs >= 1, "The epochs is not a valid number."

        summary = defaultdict(list)
        for n_epoch in range(epochs):

            start_time = time.time()

            # a epoch in the entire data set
            for example_index in range(self.x.shape[0]):
                example = self.x[example_index, :]

                # Step 1 - Forward
                output = self.forward(example)
                error = self.y[example_index]-output

                # Step 2 - Backwards
                deltas = self.backwards(error, example, output)

                # Step 3 - Updates
                self.bp_update(example, deltas)

            actual_prediction = self.predict()

            # Storing Epoch metrics
            rmse = Metrics.rms_loss(self.y, actual_prediction)
            summary['accuracy'].append(Metrics.accuracy(self.y, actual_prediction))
            summary['rmse'].append(rmse)

            if verbose:
                print("EPOCH: {} | RMSE: {:.2f} | TIME: {:.2f} ms".format(n_epoch+1, rmse, (time.time()-start_time)/10**-3))

        self.summary = summary

    def get_accuracy(self):
        """
        Gets the accuracy curve points for the last training procedure.
        :return:                        A numpy array.
        """
        return self.summary['accuracy']

    def get_rmse(self):
        """
        Gets the accuracy curve points for the last training procedure.
        :return:                        A numpy array.
        """
        return self.summary['rmse']

    def predict(self):
        """
        This method provides the functionality to predict the values per target using the neural network.
        :return:                                A numpy array with the predicted values.
        """

        predicted_values = []

        # Make a prediction for every sample
        for example_index in range(self.x.shape[0]):
            predicted_values.append(self.forward(self.x[example_index, :]))

        return np.array(predicted_values)


np.random.seed(42)

nn = NeuralNetwork([4, 3, 3], 'sigmoid', 0.1)

# Pre processing the data to train
data = load_iris(return_X_y=True)
X = data[0]
y = Metrics.one_hot_encoding(data[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()

x_train_std = sc.fit_transform(X_train)
x_test_std = sc.fit_transform(X_test)

nn.fit(x_train_std, y_train)
nn.train(150, verbose=True)
y_predicted = nn.predict()

# cm = Metrics.confusion_matrix(y_train, y_predicted)

plt.plot(nn.get_rmse())
plt.title('RMSE vs Epochs')
plt.show()


# fake_data = np.random.randint(0, 2, size=(1000, 2))
# real_and = np.logical_and(fake_data[:, 0], fake_data[:, 1])
# real_and = np.array([1 if x else 0 for x in real_and])

# nn = NeuralNetwork([2, 1], 'sigmoid', 0.01)
# nn.fit(fake_data, real_and)
# nn.train(100, verbose=True)

# accuracy = nn.get_accuracy()

# plt.plot(accuracy)
# plt.xlabel("# Epochs")
# plt.ylabel("Accuracy")
# plt.show()









