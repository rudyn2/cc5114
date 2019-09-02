import numpy as np


class Splitter:
    """
    Train test split Class.
    """

    def __init__(self):
        """
        Constructor for Splitter Class.
        """
        self.x = None
        self.y = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits the data to the Splitter object.
        """
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def transform(self, test_size, random_seed=42):
        """
        Splits the data in 4 accordingly to the test size.
        :param test_size:                           Float value with test size.
        :param random_seed:                         Random seed for numpy.
        :return:                                    A Tuple.
                                                        1) X train array with 1-test size of total data.
                                                        2) X test array with test size of total data.
                                                        3) Y train array with 1-test size of total data.
                                                        4) Y test array with test size of total data.
        """

        assert type(test_size) == float, "The test size must be a float value"
        assert 0 <= test_size <= 1, "The test size mus be between 0 and 1."

        np.random.seed(random_seed)

        # Both arrays has same length so we shuffle the indexes
        indexes = np.arange(self.x.shape[0])
        np.random.shuffle(indexes)
        self.x = self.x[indexes]
        self.y = self.y[indexes]
        sep_index = int((1 - test_size)*self.x.shape[0])

        return self.x[:sep_index, :], self.x[sep_index+1:, :], self.y[:sep_index, :], self.y[sep_index+1:, :]

    def fit_transform(self, x, y, test_size):
        """
        Fits and transforms the data.
        :param x:                                   Numpy array with X data to split.
        :param y:                                   Numpy array with y data to split.
        :param test_size:                           Float value with test size.
        :return:                                    A Tuple.
                                                        1) X train array with 1-test size of total data.
                                                        2) X test array with test size of total data.
                                                        3) Y train array with 1-test size of total data.
                                                        4) Y test array with test size of total data.
        """
        self.fit(x, y)
        return self.transform(test_size)


from sklearn.datasets import load_iris
from src.neural_network.metrics.Metrics import Metrics

np.random.seed(42)

# Loads the dataset
data = load_iris(return_X_y=True)
X = data[0]

# One hot encoding and data set separation
y, classes = Metrics.one_hot_encoding(data[1])
X_train, X_test, y_train, y_test = Splitter().fit_transform(X, y, 0.3)