import numpy as np


class Normalizer:

    def __init__(self):
        self.data = None

    def fit(self, data):

        assert type(data) == np.ndarray, "The data must be a numpy array"
        self.data = data

    def transform(self, n_low, n_high):
        """
        Transforms the data.
        :param n_low:
        :param n_high:
        :return:
        """

        assert self.data is not None, "The data has not been fitted."

        data = self.data.copy()
        for column_index in range(data.shape[1]):

            column = data[:, column_index]
            max_value = np.max(column)
            min_value = np.min(column)
            data[:, column_index] = (column-min_value)*(n_high-n_low)/(max_value-min_value) + n_low

        return data

    def fit_transform(self, data, n_low, n_high):
        """
        Fits and transforms the data.
        :param data:                            Data to fit.
        :param n_low:                           Minimal value required after transformation.
        :param n_high:                          Maximum value required after transformation.
        :return:                                The transformed data.
        """
        self.fit(data)
        return self.transform(n_low, n_high)
