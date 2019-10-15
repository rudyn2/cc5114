from src.genetic_algorithm.GeneticEngine import GeneticEngine
from src.genetic_algorithm.IndividualGuys import KnapSack01Individual
from src.genetic_algorithm.FitnessGuys import KnapSack01Fitness
from src.genetic_algorithm.Utils import plot_hotmap

weights = [12, 2, 1, 1, 4]
values = [4, 2, 2, 1, 10]
max_capacity = 20
bits_fitness = KnapSack01Fitness.KnapSackFitness_01(weights=weights,
                                                    values=values,
                                                    max_capacity=max_capacity)
ga = GeneticEngine(population_size=30,
                   gen_size=len(weights),
                   mutation_rate=0.5,
                   gen_mutation_rate=0.2,
                   elitism_rate=0.1,
                   fitness_function=bits_fitness,
                   individual_generator=KnapSack01Individual.KnapSack01Individual,
                   max_iter=30)
# ga.run()
# ga.plot_evolution()
plot_hotmap(len(weights),
            bits_fitness,
            KnapSack01Individual.KnapSack01Individual)
