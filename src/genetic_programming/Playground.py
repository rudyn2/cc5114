from AstIndividual import AstIndividual
from PopManager import PopManager
from AstIndividualGenerator import AstIndividualGenerator
from FindNumberFitness import FindNumberFitness
from arboles import *
from GeneticEngine import GeneticEngine

# random.seed(42)

allowed_functions = [AddNode, SubNode, MultNode, MaxNode]
allowed_terminals = [25, 7, 8, 100, 4, 2]
max_depth = 3

ast_gen = AstIndividualGenerator(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)
ast_1 = ast_gen(max_depth=3)
ast_2 = ast_gen(max_depth=3)

child = ast_1.cross_over(ast_2)
child_gen = child.get_gen()
print(child_gen)
print(f"Depth: {child_gen.get_depth()}")
print(f"Is Pure: {child_gen.is_pure()}")

# find_number_fitness = FindNumberFitness(target_number=65346)
# ast_gen = AstIndividualGenerator(allowed_functions=allowed_functions, allowed_terminals=allowed_terminals)
# ga = GeneticEngine(population_size=100,
#                    gen_size=max_depth,
#                    mutation_rate=0.8,
#                    gen_mutation_rate=0.3,
#                    elitism_rate=0.5,
#                    fitness_function=find_number_fitness,
#                    individual_generator=ast_gen,
#                    max_iter=20)
# ga.run(mode='max_iter')
# ga.plot_evolution()
# print(f"Number found: {ga.get_best().tree.eval()}")
