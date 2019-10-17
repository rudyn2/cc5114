from src.genetic_algorithm.Fitness import Fitness
import numpy as np


class KnapSackFitness_01(Fitness):
    """
    Fitness Function Class for the 0-1 Knapsack problem.
    """

    def __init__(self, weights: list, values: list, max_capacity: int):
        """
        Generic constructor for the KnapSackFitness_01 class.
        :param weights:                             The weights of each kind of item.
        :param values:                              The values of each kind of item.
        :param max_capacity:                        Max amount of total weight supported.
        """
        super().__init__()
        assert len(weights) == len(values), "The values and weights parameters must have same length."
        self.weights = weights
        self.max_capacity = max_capacity
        self.values = values

    def eval(self, individual):
        """
        This method evaluates some individual and returns a score. The score is associated to the 0-1 Knapsack Problem.
        So, the method will return the dot product of the elements values and the selected elements array (this array will
        contains 1 at the i-esima position if the item i-esimo is selected, otherwise returns 0) if the sum of the
        weights is less than the max capacity supported. Else, it will return 0 (because the thief cant carry out all
        that items together).
        :param individual:                              Some 01KnapSackIndividual.
        :return:                                        The score.
        """
        selected_elements = individual.get_gen()
        if np.sum(np.array(self.weights)*np.array(selected_elements)) > self.max_capacity:
            return .0
        return np.sum(np.array(self.values)*np.array(selected_elements))
