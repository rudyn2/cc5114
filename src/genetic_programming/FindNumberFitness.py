from Fitness import Fitness
from AstIndividual import AstIndividual
import numpy as np


class FindNumberFitness(Fitness):

    def __init__(self, target_number: float):
        super().__init__()
        self.target_number = target_number

    def eval(self, individual):
        assert isinstance(individual, AstIndividual), "The individual to evaluate must be an Abstract Syntax Tree"
        tree_to_eval = individual.get_gen()
        punishment = 0 if tree_to_eval.is_pure() else 0.5
        if tree_to_eval.get_depth() == 0:
            return np.exp(-(((tree_to_eval.eval()) - self.target_number)/self.target_number)**2)
        return np.exp(-((tree_to_eval.eval() - self.target_number)/self.target_number)**2) + 1/tree_to_eval.get_depth() - punishment
