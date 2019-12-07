from Fitness import Fitness
from AstIndividual import AstIndividual


class FindNumberFitness(Fitness):

    def __init__(self, target_number: float):
        super().__init__()
        self.target_number = target_number

    def eval(self, individual):
        assert isinstance(individual, AstIndividual), "The individual to evaluate must be an Abstract Syntax Tree"
        tree_to_eval = individual.get_gen()
        punishment = 1 if tree_to_eval.is_pure(feed_dict={'values': []}) else 100
        return -(abs(self.target_number - tree_to_eval.eval(feed_dict={'values': []})) + tree_to_eval.get_depth())*punishment
