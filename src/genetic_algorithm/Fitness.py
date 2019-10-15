from abc import abstractmethod


class Fitness:

    def __init__(self):
        pass

    @abstractmethod
    def eval(self, individual):
        pass
