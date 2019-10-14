import numpy as np


def fitness(individual_gen: np.ndarray, target_gen: np.ndarray):
    return np.sum(individual_gen == target_gen)


def sequence_bits_generator(size: (int, int)):
    return np.random.randint(0, 2, size=size)


def roulette_selection(scores: np.ndarray, population: np.ndarray, target_gen: np.ndarray, fitness_function) -> np.ndarray:
    assert population.shape[1] == target_gen.shape[0], "The gen length are not equals"

    # Calculate overall fitness
    overall_fitness = np.sum(scores)

    # Lets play the roulette game!
    # Go through the population and sum fitness's. When the sum s is greater than r,
    # stop and return the individual where you are.
    r = np.random.randint(0, int(overall_fitness))
    s = 0
    selected = 0
    for individual_index in population.shape[0]:
        if s >= r:
            selected = individual_index
            break
        s += scores[individual_index]
    return population[selected, :]


def cross_over(individual_a: np.ndarray, individual_b: np.ndarray) -> np.ndarray:
    pass


def find_sequence_bits(target: np.ndarray,
                       population_shape: (int, int),
                       max_iter: int,
                       fitness_function,
                       individual_generator):
    assert target.shape[0] == population_shape[1]

    # Create the population
    population = individual_generator(population_shape)

    # Calculate the scores
    scores = np.ndarray([fitness_function(population[idx, :], target) for idx in range(population.shape[0])])

    for _ in range(max_iter):
        # Solution found?
        if np.max(scores) == target.shape[0]:
            break

        # Selection

        # Crossover

        # Mutation






