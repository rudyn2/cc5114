from abc import abstractmethod


class Fitness:
    """
    Abstract class for Fitness functions.
    """

    def __init__(self):
        pass

    @abstractmethod
    def eval(self, individual):
        """
        This method must take any individual and evaluate it using some user defined metric. That's
        all, just take the individual, evaluate it and returns the score.
        :param individual:
        :return:
        """
        pass
