from src.genetic_algorithm.GeneticEngine import GeneticEngine
from src.genetic_algorithm.IndividualGuys import KnapSack01Individual
from src.genetic_algorithm.FitnessGuys import KnapSack01Fitness

weights = [12, 2, 1, 1, 4]
values = [4, 2, 2, 1, 10]
max_capacity = 16
bits_fitness = KnapSack01Fitness.KnapSackFitness_01(weights=weights,
                                                    values=values,
                                                    max_capacity=max_capacity)
ga = GeneticEngine(population_size=100,
                   gen_size=len(weights),
                   mutation_rate=0.8,
                   gen_mutation_rate=0.2,
                   elitism_rate=0.05,
                   fitness_function=bits_fitness,
                   individual_generator=KnapSack01Individual.KnapSack01Individual,
                   max_iter=20)
ga.run()
ga.plot_evolution()
