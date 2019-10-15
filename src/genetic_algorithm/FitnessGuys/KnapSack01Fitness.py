from src.genetic_algorithm.Fitness import Fitness
import numpy as np


class KnapSackFitness_01(Fitness):

    def __init__(self, weights: list, values: list, max_capacity: int):
        super().__init__()
        assert len(weights) == len(values), "The values and weights parameters must have same length."
        self.weights = weights
        self.max_capacity = max_capacity
        self.values = values

    def eval(self, individual):
        selected_elements = individual.get_gen()
        if np.sum(np.array(self.weights)*np.array(selected_elements)) > self.max_capacity:
            return .0
        return np.sum(np.array(self.values)*np.array(selected_elements))
