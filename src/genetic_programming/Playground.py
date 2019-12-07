from AstIndividualGenerator import AstIndividualGenerator
from fitness.FindNumberFitness_v2 import FindNumberFitness
from arboles import *
from GeneticEngine import GeneticEngine

random.seed(42)

allowed_functions = [AddNode, SubNode, MultNode]
allowed_terminals = [25, 7, 8, 100, 4, 2]
max_depth = 5

find_number_fitness = FindNumberFitness(target_number=65346)
ast_gen = AstIndividualGenerator(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)
ga = GeneticEngine(population_size=200,
                   gen_size=max_depth,
                   mutation_rate=0.8,
                   gen_mutation_rate=0.3,
                   elitism_rate=0.7,
                   fitness_function=find_number_fitness,
                   individual_generator=ast_gen,
                   max_iter=10)


ga.run(mode='max_iter', verbose=True)
ga.plot_evolution()
best_tree = ga.get_best().tree
result = best_tree.eval(feed_dict={'values': []})
print(f"Number found: {result}")
print(f"Depth: {best_tree.get_depth()}")


