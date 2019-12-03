from src.genetic_algorithm.FitnessGuys import WordFitness
from src.genetic_algorithm.GeneticEngine import GeneticEngine
from src.genetic_algorithm.IndividualGuys import WordIndividual

# Execute the GA algorithm for the word exercise
word_to_find = 'ElbromaS'
w_fitness = WordFitness.WordFitness(word_to_find)
ga = GeneticEngine(population_size=100,
                   gen_size=len(word_to_find),
                   mutation_rate=0.8,
                   gen_mutation_rate=0.2,
                   elitism_rate=0.2,
                   fitness_function=w_fitness,
                   individual_generator=WordIndividual.WordIndividual,
                   max_iter=20)

ga.run(mode='max_iter')
ga.plot_evolution()