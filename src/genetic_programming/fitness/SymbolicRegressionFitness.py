from Fitness import Fitness
from AstIndividual import AstIndividual


class SymbolicRegressionFitness(Fitness):

    def __init__(self, target_data: dict):
        super().__init__()
        self.target_data = target_data

        # check that the variables has the same amount of data
        assert len(set([len(item) for item in target_data.values()])) == 1, "Los datos no poseen las mismas dimensiones"
        assert "values" in target_data.keys(), "Debes introducir los valores de la función a evaluar en la" \
                                               "llave values"

    def eval(self, individual):
        assert isinstance(individual, AstIndividual), "The individual to evaluate must be an Abstract Syntax Tree"
        tree_to_eval = individual.get_gen()
        difference = 0
        for idx in range(len(self.target_data['values'])):
            # build the feed dict with all the variables
            feed_dict = {}
            for var in self.target_data.keys():
                if var == 'values':
                    continue
                feed_dict[var] = self.target_data[var][idx]
            difference += abs(tree_to_eval.eval(feed_dict)-self.target_data['values'][idx])
        return -difference
