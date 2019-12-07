from Fitness import Fitness
from AstIndividual import AstIndividual


class FindNumberFitness(Fitness):

    def __init__(self, target_number: float):
        super().__init__()
        self.target_number = target_number

    def eval(self, individual):
        assert isinstance(individual, AstIndividual), "The individual to evaluate must be an Abstract Syntax Tree"
        tree_to_eval = individual.get_gen()
        try:
            return -(abs(self.target_number - tree_to_eval.eval(feed_dict={'values': []})) + tree_to_eval.get_depth())
        except ZeroDivisionError:
            return -10**12
