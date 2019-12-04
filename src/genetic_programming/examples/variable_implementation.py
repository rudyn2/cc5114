import abstract_syntax_tree
from arboles import *
from fitness.SymbolicRegressionFitness import SymbolicRegressionFitness
import numpy as np
from AstIndividualGenerator import AstIndividualGenerator
from GeneticEngine import GeneticEngine

random.seed(31)

max_depth = 5
allowed_functions = [AddNode, SubNode, MultNode]
allowed_terminals = list(range(-10, 11))
allowed_terminals.extend(["x"]*len(allowed_terminals))

x_domain = np.linspace(-100, 100, 201)
regression_fitness = SymbolicRegressionFitness(target_data={
    'x': x_domain,
    'values': x_domain*x_domain + x_domain - 6
})
ast_gen = AstIndividualGenerator(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)
ga = GeneticEngine(population_size=100,
                   gen_size=max_depth,
                   mutation_rate=0.6,
                   gen_mutation_rate=0.3,
                   elitism_rate=0.6,
                   fitness_function=regression_fitness,
                   individual_generator=ast_gen,
                   max_iter=5)
ga.run(mode='fitness_threshold', fitness_threshold=-1)
ga.plot_evolution()
