from AstIndividualGenerator import AstIndividualGenerator
from fitness.FindNumberFitness_v2 import FindNumberFitness
from arboles import *
from GeneticEngine import GeneticEngine

random.seed(42)

target_number = 65346
allowed_functions = [AddNode, SubNode, MultNode]
allowed_terminals = [25, 7, 8, 100, 4, 2]
max_depth = 4

find_number_fitness = FindNumberFitness(target_number=target_number)
ast_gen = AstIndividualGenerator(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)
ga = GeneticEngine(population_size=200,
                   gen_size=max_depth,
                   mutation_rate=0.7,
                   gen_mutation_rate=0.3,
                   elitism_rate=0.7,
                   fitness_function=find_number_fitness,
                   individual_generator=ast_gen,
                   max_iter=10)

# If automatic is True, the GP algorithm is executed several times until it reaches the "max difference allowed" and
# "max depth allowed" criterion. Each execution of the algorithm will stop after the maximum iteration parameter
# specified in the Genetic Engine initialization.
automatic = False
max_difference_allowed = 346
max_depth_allowed = 4

if automatic:
    idx = 0
    trees = []
    while idx <= 1:
        ga.run(mode='max_iter', verbose=False)
        best_tree = ga.get_best().tree
        result = best_tree.eval(feed_dict={'values': []})
        ga.plot_evolution(y_scale=True)
        if abs(result-65346) == max_difference_allowed and best_tree.get_depth() <= max_depth_allowed:
            print(f"Number found: {result}")
            print(f"Depth: {best_tree.get_depth()}")
            break
        idx += 1
        print(f"Iteration: {idx} | Result: {result} | Depth: {best_tree.get_depth()}")
        trees.append(best_tree)
else:
    ga.run(mode='fitness_threshold', fitness_threshold=-459, verbose=True)
    best_tree = ga.get_best().tree
    result = best_tree.eval(feed_dict={'values': []})
    ga.plot_evolution(y_scale=True)
    print(f"Number found: {result}")
    print(f"Depth: {best_tree.get_depth()}")



