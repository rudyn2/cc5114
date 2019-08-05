import numpy as np
from abc import ABC, abstractmethod


class Perceptron(ABC):

    def __init__(self):

        self.X = None
        self.X_bias = None
        self.t = None
        self.W = None

    def fit(self, x):
        """
        It receives the data and store it into the fields of the class.
        :param x:                   Numpy Array with the data
        :return:
        """
        self.X = x

        # Add bias to the train matrix
        ones = np.ones(shape=(self.X.shape[0], 1))
        self.X_bias = np.hstack([ones, self.X])

    def train(self, t):
        """
        This method trains a perceptron using a simple algorithm explained in the McCulloch-Pitts model of a
        perceptron. The weights are stored in the W attribute.
        :param t:                   Numpy array with desired outputs.
        """

        self.t = t
        # Init a weights vector
        w = np.zeros(shape=(self.X.shape[1]+1, 1))

        # Training using perceptron algorithm
        for i in range(self.X.shape[0]):
            ex = self.X_bias[i, :]
            ex_target = t[i]

            if (np.matmul(ex, w) >= 0 and ex_target == 1) or (np.matmul(ex, w) < 0 and ex_target == 0):
                pass
            else:
                factor = -1 if ex_target == 0 else ex_target
                adjust = factor*ex
                adjust = adjust.reshape(self.X_bias.shape[1], 1)
                w = np.add(w, adjust)

        self.W = w

    def predict(self, verbose=False):
        """
        Executes a forward propagation in the network and outputs the result. This was designed to
        perform logical operations between just n operators.
        :return:                    A Numpy Array with the predicted values.
        """

        result = np.matmul(self.X_bias, self.W) >= 0
        result = [1 if value else 0 for value in result]

        if verbose:
            print("\nTraining data shape: {}".format(self.X.shape))
            print("Total cases: {}\nTrue positives: {}".format(len(self.t), np.sum(result == self.t)))

        return np.array(result)

    def predict_new(self, new):

        pair = np.hstack([1, np.array(new)]).reshape((len(new)+1, 1)).T
        return 1 if np.matmul(pair, self.W) >= 0 else 0


class AND(Perceptron):

    def __init__(self):
        super().__init__()
        # self.W = np.array([[-1], [0.5], [0.5]])


class OR(Perceptron):

    def __init__(self):
        super().__init__()
        # self.W = np.array([[-0.5], [0.5], [0.5]])


class NAND(Perceptron):

    def __init__(self):
        super().__init__()
        # self.W = np.array([[1], [-0.501], [-0.5]])


class Adder:
    """
    This class provides the functionality to realize a binary sum operation using a nand perceptron.
    """

    def __init__(self, nand):
        """
        Constructor for the adder class.
        :param nand:
        """
        self._nand = nand

    def sum(self, x, y):
        """
        Return the sum and carry bits between 2 operators.
        :param x:                   A single bit (0 or 1)
        :param y:                   A single bit (0 or 1)
        :return:                    A tuple.
                                        1) The sum between 0 and 1 (0 or 1)
                                        2) The carry bit of the sum operation (0 or 1)
        """
        assert self._nand.X.shape[1] == 2, "The provided nand perceptron was trained for a higher dimensional data"
        aux = self._nand.predict_new([x, y])
        sum = self._nand.predict_new([self._nand.predict_new([x, aux]), self._nand.predict_new([y, aux])])
        carry = self._nand.predict_new([aux, aux])
        return sum, carry


fake_data = np.random.randint(0, 2, size=(1000, 2))
real_and = np.logical_and(fake_data[:, 0], fake_data[:, 1])
real_and = np.array([1 if x else 0 for x in real_and])
real_or = np.logical_or(fake_data[:, 0], fake_data[:, 1])
real_nand = np.logical_not(real_and)

p_nand = NAND()
p_or = OR()
p_and = AND()

# Using an AND perceptron
p_and.fit(fake_data)
p_and.train(real_and)
result = p_and.predict(verbose=True)

p_or.fit(fake_data)
p_or.train(real_or)
p_or.predict(verbose=True)

p_nand.fit(fake_data)
p_nand.train(real_nand)
p_nand.predict(verbose=True)

adder = Adder(p_nand)
a, b = adder.sum(0, 0)

print("\nThe sum bit was {} and the carry bit {}".format(a, b))
