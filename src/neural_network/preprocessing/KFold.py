import numpy as np


class KFold:
    """
    This class provides the functionality to split a dataset ir order to perform k-cross validation.
    """

    def __init__(self):
        """
        Constructor for the KFold class.
        """
        self.data = None

    def fit(self, data: np.ndarray):
        """
        Fits the data to actual object.
        :param data:                                Data to split.
        """
        self.data = data

    def split(self, n_fold: int, random_state=42) -> list:
        """
        This method returns a list of tuples where you have the index for training and testing.
        :param n_fold:                              Number of splits.
        :param random_state:                        Numpy random state (int)
        :return:                                    A list of tuple. Each tuple:
                                                        1) Training index
                                                        2) Testing index
        """

        assert self.data is not None, "You must fit the data first."
        assert type(n_fold) == int and n_fold > 0, "The number of k-folds must be a positive integer."
        assert n_fold < self.data.shape[0], "The number of splits must be greater than the amount of data."

        # Shuffle the data
        np.random.shuffle(self.data)
        total_indexes = np.array(range(self.data.shape[0]))

        # Generates the indexes limits
        index_limits = np.linspace(0, self.data.shape[0], n_fold+1)
        indexes_tuples = []
        for index_limit_pos in range(len(index_limits)-1):
            lower_index = int(round(index_limits[index_limit_pos]))
            upper_index = int(round(index_limits[index_limit_pos+1])-1)
            print("Fold: {}, Lower: {}, Upper: {}".format(index_limit_pos + 1, lower_index, upper_index))

            testing_index = total_indexes[lower_index: upper_index]
            training_index_lower = total_indexes[:lower_index]
            training_index_upper = total_indexes[upper_index:]
            training_index = np.concatenate([training_index_lower, training_index_upper])
            indexes_tuples.append((training_index, testing_index))

        return indexes_tuples

    def fit_split(self, data: np.ndarray, n_fold: int, random_state=42) -> list:
        """
        Fits and splits the data
        :param data:                                Data to split.
        :param n_fold:                              Number of splits.
        :param random_state:                        Numpy random state.
        :return:                                    The same as split method.
        """
        self.fit(data)
        return self.split(n_fold, random_state=random_state)
