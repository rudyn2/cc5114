import numpy as np


class Metrics:

    @staticmethod
    def accuracy(real_y, predicted_y):
        """
        This method provides the functionality to calculate the accuracy between a prediction and a real set of values.
        :param real_y:                              A numpy array with real values.
        :param predicted_y:                         A numpy array with predicted values.
        :return:                                    A single float value.
        """

        assert len(real_y) == len(predicted_y), "The length must be equal."
        # predicted_y= Metrics._parse_sigmoid(predicted_y)

        n = predicted_y.shape[0]
        total = 0
        for single_output_index in range(n):
            if predicted_y[single_output_index].all() == real_y[single_output_index].all():
                total += 1

        # Returns the ratio of corrected samples over all the samples
        return total/n

    @staticmethod
    def rms_loss(real_y, predicted_y):
        """
        This method provides the functionality to calculate the Root Mean Square error between the predicted and real
        values.
        :param real_y:                              A numpy array with real values.
        :param predicted_y:                         A numpy array with predicted values.
        :return:                                    A single float value metric.
        """
        n = predicted_y.shape[0]
        total_loss = 0
        for single_output_index in range(n):
            total_loss += Metrics._euclidean_distance(real_y[single_output_index], predicted_y[single_output_index])**2
        return np.sqrt(total_loss/n)

    @staticmethod
    def confusion_matrix(real_y, predicted_y):
        """
        Gets the confusion matrix assuming that the data has one hot encoding. Each row correspond to a single output.
        :param real_y:                              An array with one hot encoding.
        :param predicted_y:                         Other array with one hot encoding.
        :return:                                    A numpy array matrix.
        """

        assert real_y.shape[1] == predicted_y.shape[1], "The second dimension of input arrays must be equal."

        n_classes = real_y.shape[1]
        cm = np.zeros(shape=(n_classes, n_classes))

        for row in range(n_classes):
            for column in range(n_classes):

                # We create the predicted and expected codes
                code = np.zeros(shape=(1, n_classes))
                predicted_code = code[row]
                real_code = code[row]

                predicted_code[row] = 1
                real_code[column] = 1

    @staticmethod
    def _euclidean_distance(x, y):
        """
        Calculates the euclidean distance between 2 N-Dimensional data points.
        :param x:                                   A 1D Numpy array.
        :param y:                                   A 1D Numpy array.
        :return:                                    A single float value.
        """
        assert len(x) == len(y)
        return np.sqrt(np.sum((x-y)**2))

    @staticmethod
    def one_hot_encoding(array: np.ndarray) -> np.ndarray:

        assert array.ndim == 1, "The array provided must be one dimensional"

        # First we detect the classes
        classes = np.unique(array)
        total_classes = len(classes)

        codes = []
        for element in array:
            code = np.zeros(total_classes)
            code[classes == element] = 1
            codes.append(code)

        encoded_data = np.array(codes)
        return encoded_data

    @staticmethod
    def _parse_sigmoid(array, threshold=0.5):
        """
        Parse an array to one hot encoding transformation.
        :param array:
        :param threshold:
        :return:
        """
        mask = array >= threshold
        array[mask] = 1
        array[~mask] = 0
        return array

