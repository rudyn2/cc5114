from src.genetic_algorithm.GeneticEngine import GeneticEngine
from src.genetic_algorithm.IndividualGuys import KnapSack01Individual
from src.genetic_algorithm.FitnessGuys import KnapSack01Fitness
from src.genetic_algorithm.Utils import plot_hotmap

import numpy as np

np.random.seed(42)
weights = [12, 2, 1, 1, 4]
values = [4, 2, 2, 1, 10]
max_capacity = 15
bits_fitness = KnapSack01Fitness.KnapSackFitness_01(weights=weights,
                                                    values=values,
                                                    max_capacity=max_capacity)
ga = GeneticEngine(population_size=25,
                   gen_size=len(weights),
                   mutation_rate=0.2,
                   gen_mutation_rate=0.2,
                   elitism_rate=0.1,
                   fitness_function=bits_fitness,
                   individual_generator=KnapSack01Individual.KnapSack01Individual,
                   max_iter=50)
# ga.run(mode='max_iter')
# ga.plot_evolution()
plot_hotmap(gen_size=len(weights),
            fitness_function=bits_fitness,
            elitism_rate=0.2,
            mode='fitness_threshold',
            individual_generator=KnapSack01Individual.KnapSack01Individual,
            fitness_threshold=13)
