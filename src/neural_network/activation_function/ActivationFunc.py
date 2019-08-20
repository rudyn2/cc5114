from abc import abstractmethod


class ActivationFunc:

    def __init__(self):
        pass

    @abstractmethod
    def get_value(self, x):
        pass

    @abstractmethod
    def get_derivative(self, x):
        pass



