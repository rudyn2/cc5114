from abc import abstractmethod

from src.genetic_algorithm.Fitness import Fitness


class Individual:

    def __init__(self, length_gen: int):
        self.length_gen = length_gen

    def eval(self, fitness_function: Fitness) -> float:
        return fitness_function.eval(self)

    @abstractmethod
    def cross_over(self, other):
        pass

    @abstractmethod
    def mutate(self, gen_mutation_rate):
        pass

    @abstractmethod
    def get_gen(self):
        pass

    @abstractmethod
    def fit_gen(self, gen):
        pass
