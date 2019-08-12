import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Perceptron(ABC):

    def __init__(self):

        self.X = None
        self.X_bias = None
        self.t = None
        self.b = None
        self.W = None

    def fit(self, x, t):
        """
        It receives the data and store it into the fields of the class.
        :param x:                   Numpy Array with the data
        :param t:                   Numpy array with the target data
        :return:
        """
        self.X = x
        self.t = t

    def train(self, plot_lc=False):
        """
        This method trains a perceptron using a simple algorithm explained in the McCulloch-Pitts model of a
        perceptron. The weights are stored in the W attribute.
        """

        # init learning rate
        lr = 0.1

        # Init a weights vector
        self.W = np.zeros(shape=(self.X.shape[1], 1))

        # Init the bias
        b = (np.random.random()*-1 + np.random.random())*2

        # Training using perceptron algorithm
        for i in range(self.X.shape[0]):

            ex = self.X[i, :]

            desired_output = self.t[i]
            predicted_output = np.matmul(ex, self.W) + b
            diff = float(desired_output-predicted_output)

            self.W = self.W + lr*diff*(ex.reshape(self.W.shape[0], 1))
            b += lr*diff

        self.b = b

    def learning_curve(self):
        """
        Generates the learning curve.
        :return:                        A numpy array with the learning rate over n iterations.
        """
        aux_x = self.X
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.t, test_size=0.33, random_state=42)
        learning_curve = []

        for i in range(1, x_train.shape[0]):
            self.X = x_train[:i, :]
            self.train(plot_lc=False)
            result = self.predict()
            learning_curve.append(sum(result)/len(result))

        self.X = aux_x

        return learning_curve

    def predict(self, verbose=False):
        """
        Executes a forward propagation in the network and outputs the result. This was designed to
        perform logical operations between just n operators.
        :return:                    A Numpy Array with the predicted values.
        """

        result = np.matmul(self.X, self.W) + self.b >= 0
        result = [1 if value else 0 for value in result]

        if verbose:
            print("\nTraining data shape: {}".format(self.X.shape))
            print("Total cases: {}\nTrue positives: {}".format(len(self.t), np.sum(result == self.t)))

        return np.array(result)


length = 200
data = np.random.rand(length, 2)*2
target = data[:, 0]*2 >= data[:, 1]
line_points = data[:, 0]*2

blue = data[target]
red = data[~target]

plt.xlim((0, 2))
plt.ylim((0, 2))
plt.plot(data[:, 0], line_points)
plt.scatter(blue[:, 0], blue[:, 1], c='b')
plt.scatter(red[:, 0], red[:, 1], c='r')
plt.show()

p = Perceptron()
p.fit(data, target)
c = p.learning_curve()
plt.plot(c)
plt.show()
# p.train()
print("")
