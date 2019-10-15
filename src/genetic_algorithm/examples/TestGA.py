from src.genetic_algorithm.FitnessGuys import WordFitness, BitsSequenceFitness
from src.genetic_algorithm.GeneticEngine import GeneticEngine
from src.genetic_algorithm.IndividualGuys import WordIndividual, BitsSequenceIndividual

# Execute the GA algorithm for the word exercise
w_fitness = WordFitness.WordFitness('Elbromas')
ga = GeneticEngine(population_size=100,
                   gen_size=8,
                   mutation_rate=0.8,
                   gen_mutation_rate=0.2,
                   elitism_rate=0.1,
                   fitness_function=w_fitness,
                   individual_generator=WordIndividual.WordIndividual,
                   max_iter=20)

ga.run()
ga.plot_evolution()
print("----------------------------")

# Execute the GA algorithm for the sequence of bits exercise
target_sequence = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
bits_fitness = BitsSequenceFitness.BitsSequenceFitness(target_sequence)
ga = GeneticEngine(population_size=100,
                   gen_size=len(target_sequence),
                   mutation_rate=0.8,
                   gen_mutation_rate=0.2,
                   elitism_rate=0.1,
                   fitness_function=bits_fitness,
                   individual_generator=BitsSequenceIndividual.BitsSequenceIndividual,
                   max_iter=20)
ga.run()
ga.plot_evolution()
