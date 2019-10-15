import numpy as np


def fitness(individual_gen: np.ndarray, target_gen: np.ndarray):
    return np.sum(individual_gen == target_gen)


def sequence_bits_generator(size):
    return np.random.randint(0, 2, size=size)


def roulette_selection(scores: np.ndarray, population: np.ndarray, target_gen: np.ndarray) -> np.ndarray:
    assert population.shape[1] == target_gen.shape[0], "The gen length are not equals"

    # Calculate overall fitness
    overall_fitness = np.sum(scores)

    # Lets play the roulette game!
    # Go through the population and sum fitness's. When the sum s is greater than r,
    # stop and return the individual where you are.
    r = np.random.randint(0, int(overall_fitness))
    s = 0
    selected = 0
    for individual_index in range(population.shape[0]):
        if s >= r:
            selected = individual_index
            break
        s += scores[individual_index]
    return population[selected, :]


def tournament_selection(scores: np.ndarray, population: np.ndarray, target_gen: np.ndarray):

    mask = np.random.randint(0, len(scores), size=int(0.2*len(scores)))
    choosen = population[mask, :]
    sub_scores = scores[mask]
    try:
        return choosen[np.argmax(sub_scores), :]
    except ValueError:
        pass


def cross_over(individual_a: np.ndarray, individual_b: np.ndarray) -> np.ndarray:

    # Selects a random point
    separator = np.random.randint(0, individual_a.shape[0])
    left_gen_half = individual_a[:separator]
    right_gen_half = individual_b[separator:]
    child = np.concatenate([left_gen_half, right_gen_half])
    return child


def calculate_population_scores(population: np.ndarray, target: np.ndarray, fitness_function):

    scores = []
    for idx in range(population.shape[0]):
        scores.append(fitness_function(target, population[idx, :]))
    return np.array(scores)


def find_sequence_bits(target: list,
                       population_size: int,
                       max_iter: int,
                       fitness_function,
                       population_generator):

    target = np.array(target)

    # Create the initial population
    population = population_generator((population_size, len(target)))

    # Calculate the scores
    scores = calculate_population_scores(population, target, fitness_function)
    generation = 1
    solution_found = False
    for _ in range(max_iter):

        # Solution found?
        if np.max(scores) == target.shape[0]:
            solution_found = True
            break

        # First we perform elitism, 10% of the best goes to next generation
        scores_idxs = [(idx, score) for idx, score in enumerate(scores)]
        scores_idxs.sort(key=lambda x: x[1])
        scores_idxs = list(reversed(scores_idxs))
        best_individuals = np.array([population[element[0], :] for element in scores_idxs[:int(0.1*len(scores_idxs))]])

        # Selection: Tournament
        total_mutation = population_size - best_individuals.shape[0]
        new_generation = []
        for _ in range(total_mutation):
            parent_a = tournament_selection(scores, population, target)
            parent_b = tournament_selection(scores, population, target)

            # Applying cross over
            new_individual = cross_over(parent_a, parent_b)
            new_generation.append(new_individual)

        population = np.concatenate([best_individuals, np.array(new_generation)])
        scores = calculate_population_scores(population, target, fitness_function)

        print(f"Generation {generation}, best score: {np.max(scores)}, worst score: {np.min(scores)}, mean score: {np.mean(scores)}")
        generation += 1

    if solution_found:
        best_match = population[np.argmax(scores), :]
        print(f"Solution found at generation {generation}")
        return best_match


sequence = [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
solution = find_sequence_bits(sequence, 100, 10000, fitness, sequence_bits_generator)


