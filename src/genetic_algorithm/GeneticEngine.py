from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.genetic_algorithm.Individual import Individual
from src.genetic_algorithm.PopManager import PopManager


class GeneticEngine:
    """
    Genetic Engine Class for a simple Genetic Algorithm implementation.
    """

    def __init__(self, population_size: int,
                 gen_size: int,
                 mutation_rate: float,
                 gen_mutation_rate: float,
                 elitism_rate: float,
                 fitness_function,
                 individual_generator,
                 tournament_selection_proportion: float = 0.2,
                 max_iter: int = 100):
        """
        Generic constructor for the GeneticEngine Class.
        :param population_size:                         The amount of individuals for the population (int).
        :param gen_size:                                The length of each chromosome. E.g., if it the chromosomes
                                                        are words will be the number of letters in the word.
        :param mutation_rate:                           The probability that some individual will mutate.
        :param gen_mutation_rate:                       The amount of genes that will change in each chromosome. E.g,
                                                        if a word has 10 letters, a mutation rate of 0.2 means that
                                                        20% (2 letters) will mutate.
        :param elitism_rate:                            The proportion of the best of some population that will pass
                                                        to the next generation. E.g., an elitism rate of 0.2 means
                                                        that best 20% of the population will go straight to the
                                                        next generation.
        :param fitness_function:                        An instance of the Fitness Class. This object must implement
                                                        the evaluation procedure.
        :param individual_generator:                    An instance of the Individual Class. This object must have
                                                        the logic for some individual in the context of some problem.
        :param tournament_selection_proportion:         The proportion of individuals of some population that will
                                                        be randomly selected. From these, the best will be extracted.
        :param max_iter:                                Maximum number of iterations that the algorithm can pass.
        """

        self.population_size = population_size
        self.gen_size = gen_size
        self.mutation_rate = mutation_rate
        self.gen_mutation_rate = gen_mutation_rate
        self.elitism_rate = elitism_rate
        self.fitness = fitness_function
        self.individual_generator = individual_generator
        self.tournament_selection_proportion = tournament_selection_proportion
        self.max_iter = max_iter

        self.summary = None
        self.population = None

    def tournament_selection(self) -> Individual:
        """
        Performs tournament selection over the population.
        :return:
        """
        random_choices = self.population.select_random(self.tournament_selection_proportion)
        temp_manager = PopManager(self.fitness)
        temp_manager.fit(random_choices)
        return temp_manager.select_best()

    def eval_condition(self, mode: str, **kwargs):
        """
        Evaluation condition. It receives a mode as a string and evaluates a condition using the keyword
        arguments. This method was created to support futures new more complex conditions.
        :param mode:                                    Mode of operations.
                                                            max_iter: The algorithm will stop when a given amount
                                                            of iterations is reached. The maximum number of iterations
                                                            is the n_iter parameter given in the GA constructor,
                                                            fitness_threshold: The algorithm will stop when
                                                            the mean score reach some user-defined value specified
                                                            in the keyword arguments as fitness_threshold.
        :return:
        """
        if mode == 'max_iter':
            assert 'iter_actual' in list(kwargs.keys()), \
                "For max iter condition the actual iteration must be provided as keyword parameter: 'iter_actual'"

            return kwargs['iter_actual'] <= self.max_iter
        elif mode == 'fitness_threshold':
            assert 'fitness_threshold' in list(kwargs.keys()) and 'fitness_actual' in list(kwargs.keys()), \
                "For fitness threshold condition the actual fitness and fitness threshold " \
                "must be provided as keyword parameter: 'fitness_threshold' and 'fitness_actual'"
            return kwargs['fitness_actual'] <= kwargs['fitness_threshold']
        else:
            raise ValueError("No valid condition mode")

    def run(self, mode: str, verbose=True, **kwargs):
        """
        Runs the Genetic Algorithm.
        :return:
        """

        # Create the initial population
        self.population = PopManager(self.fitness)
        self.population.create(individual_class=self.individual_generator,
                               population_size=100,
                               length_gen=self.gen_size)

        # Calculate the scores
        scores = self.population.calculate_scores()
        mean_fitness = np.mean(scores)

        # Adding actual fitness to kwargs
        kwargs['fitness_actual'] = mean_fitness
        kwargs['iter_actual'] = 0
        generation = 0

        # Summary dict
        summary = defaultdict(list)
        while self.eval_condition(mode, **kwargs) or generation <= self.max_iter:

            # Elitism
            new_generation = self.population.select_rate_best(self.elitism_rate)

            for _ in range(self.population_size):
                parent_a = self.tournament_selection()
                parent_b = self.tournament_selection()

                # Applying cross over
                new_individual = parent_a.cross_over(parent_b)

                # Applying mutation
                r = np.random.rand()
                if r <= self.mutation_rate:
                    new_individual.mutate(self.gen_mutation_rate)

                new_generation.append(new_individual)

            self.population.fit(new_generation)
            # ------------------------------------------------------

            scores = self.population.calculate_scores()
            kwargs['fitness_actual'] = np.max(scores)
            kwargs['iter_actual'] = generation

            # Summary of the progress
            summary['best_scores'].append(np.max(scores))
            summary['worst_scores'].append(np.min(scores))
            summary['mean_scores'].append((np.mean(scores)))

            if verbose:
                print(f"Generation {generation}, best score: {summary['best_scores'][-1]}, "
                      f"worst score: {summary['worst_scores'][-1]}, mean score: {summary['mean_scores'][-1]}")
            generation += 1

        if verbose:
            print(f"Best individual of last generation: {self.population.select_best()}")
        self.summary = summary

    def plot_evolution(self, y_scale=False):
        """
        Generates a plot of the generation's evolution.
        """

        generations = list(range(1, len(self.summary['mean_scores'])+1))
        plt.plot(generations, self.summary['best_scores'], label='best')
        plt.plot(generations, self.summary['worst_scores'], label='worst')
        plt.plot(generations, self.summary['mean_scores'], label='mean')
        plt.title('Generation evolution')
        plt.xlabel('# Generation')
        plt.ylabel('Score')
        if y_scale:
            plt.yscale('symlog')
        plt.legend()
        plt.show()

    def get_summary(self):
        """
        Returns the generation scores per generation.
        :return:
        """
        return self.summary

    def get_best(self):
        """
        Return the best individual of the last generation.
        :return:                    Individual
        """
        return self.population.select_best()



