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
            punishment = 0 if tree_to_eval.is_pure(feed_dict={'values': []}) else -10**10
            return -(abs(self.target_number - tree_to_eval.eval(feed_dict={'values': []})) + tree_to_eval.get_depth()) + punishment
        except ZeroDivisionError:
            return -10**12
