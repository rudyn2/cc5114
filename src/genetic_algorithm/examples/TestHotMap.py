from src.genetic_algorithm.Utils import plot_hotmap
from src.genetic_algorithm.IndividualGuys import WordIndividual
from src.genetic_algorithm.FitnessGuys import WordFitness


word_to_find = 'Elbroma'
w_fitness = WordFitness.WordFitness(word_to_find)
plot_hotmap(gen_size=len(word_to_find),
            fitness_function=w_fitness,
            individual_generator=WordIndividual.WordIndividual)
