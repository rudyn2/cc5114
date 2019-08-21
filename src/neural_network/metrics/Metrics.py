import numpy as np


class Metrics:

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

        for class_i in range(n_classes):

            # Selects the i-class indexes
            mask = real_y[:, class_i] == 1
            # Selects the predicted elements with the mask
            predicted_i_class = predicted_y[mask]

            for class_j in range(n_classes):

                # Selects class j
                inter = predicted_i_class[predicted_i_class[:, class_j] == 1]
                cm[class_i, class_j] += inter.shape[0]

        return cm

    @staticmethod
    def accuracy(confusion_matrix):
        return np.trace(confusion_matrix)/np.sum(confusion_matrix)

    @staticmethod
    def filter_class(data, row):
        """
        Filters a dataset with one hot encoding values and selects all the examples which in its "row" position has a 1.
        :param real_y:                              Numpy array with one hot encoding.-
        :param row:                                 Position of 1 (it describes the class).
        :return:                                    The filtered numpy array
        """

        results = []
        for example_index in range(data.shape[0]):
            # Extract the example
            example = data[example_index, :]
            if example[row] == 1:
                results.append(example)

        mask = data[:, row] == 1

        return np.array(results)

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

